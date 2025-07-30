# %%

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv(override=True)


def create_system_prompt_with_dynamic_fewshooter(
    yaml_path: Path, system_prompt: str, input_key: str, output_key: str
) -> ChatPromptTemplate:
    """Create a system prompt with a dynamic few-shot prompt."""
    sample_queries = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # Construir ejemplos a partir del YAML
    examples = [
        {"input": item[input_key], "output": item[output_key].strip()}
        for item in sample_queries
    ]

    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = InMemoryVectorStore.from_texts(
        to_vectorize, embeddings, metadatas=examples
    )
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input"],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # In this case, each example will become 2 messages:
        # 1 human, and 1 ai
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class CypherQuery(BaseModel):
    """Cypher query agent."""

    cypher_query: str = Field(description="The Cypher query ready to be executed.")


sample_queries_path = Path(__file__).with_name("sample_queries.yaml")

chain_for_cypher_query = create_system_prompt_with_dynamic_fewshooter(
    sample_queries_path,
    "You are an expert Cypher query writer.",
    "pregunta",
    "cypher_query",
) | llm.with_structured_output(CypherQuery)


PROMPT_FOR_GENERATE_QUERIES = """
Based on the user question, return three (3) queries useful to retrieve documents in parallel.
Queries should expand/enrich the semantic space of the user question.
"""
# TEMPLATE_FOR_GENERATE_QUERIES = ChatPromptTemplate.from_template(
#     PROMPT_FOR_GENERATE_QUERIES
# )
# chain_for_generate_queries = TEMPLATE_FOR_GENERATE_QUERIES | llm.with_structured_output(
#     GeneratedQueries, method="function_calling"
# )
sample_generated_answers_path = Path(__file__).with_name(
    "sample_generated_answers.yaml"
)

chain_for_questions_generation = create_system_prompt_with_dynamic_fewshooter(
    sample_generated_answers_path,
    PROMPT_FOR_GENERATE_QUERIES,
    "pregunta",
    "generated_queries",
) | llm.with_structured_output(GeneratedQueries)
