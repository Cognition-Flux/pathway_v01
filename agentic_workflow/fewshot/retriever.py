"""Module for retrieving few-shot examples."""

# %%
from langchain_community.retrievers import BM25Retriever

from agentic_workflow.fewshot.loader import load_examples


def format_example(row):
    """Format an example into a string."""
    question = row["human"]
    answer = row["ai-assistant"]
    return f"""<problem>
                {question}
                </problem>
                <solution>
                {answer}
                </solution>"""


# Load examples from YAML file
examples = load_examples()

retriever = BM25Retriever.from_texts([format_example(row) for row in examples])

if __name__ == "__main__":
    top_k = 2

    examples_str = "\n".join(
        [
            doc.page_content
            for doc in retriever.invoke("fármacos y medicamentos")[:top_k]
        ]
    )
    print(examples_str)
