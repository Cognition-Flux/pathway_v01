"""
Simple parallel retriever with metadata filtering using Hybrid Search.
"""

# %%
import os
from typing import Dict, Literal, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from dev.langgraph_basics.simple_hybrid_search_w_metadata_filtering import (
    retriever,
)

load_dotenv(override=True)


class MetadataFilterFields(BaseModel):
    """
    Schema for filtering metadata.
    All fields are optional.
    """

    enzyme: Optional[
        Literal[
            "HK",
            "PFK1",
            "PK",
            "CS",
            "IDH",
            "AKGDH",
            "SDH",
            "GAPDH",
            "MDH",
            "SSADH",
        ]
    ] = Field(
        default=None,
        description=(
            "The enzyme name. The names are HK: hexokinase, "
            "PFK1: phosphofructokinase-1, PK: pyruvate kinase, "
            "CS: citrate synthase, IDH: isocitrate dehydrogenase, "
            "AKGDH: alpha-ketoglutarate dehydrogenase, "
            "SDH: succinate dehydrogenase, "
            "GAPDH: glyceraldehyde-3-phosphate dehydrogenase, "
            "MDH: malate dehydrogenase, "
            "SSADH: succinate semialdehyde dehydrogenase."
        ),
    )
    subsystem: Optional[Literal["glycolysis", "TCA"]] = Field(
        default=None, description="The metabolic subsystem."
    )
    # substrates: Optional[
    #     List[
    #         Literal[
    #             "Glc",
    #             "ATP",
    #             "F6P",
    #             "PEP",
    #             "ADP",
    #             "AcCoA",
    #             "OAA",
    #             "IsoCit",
    #             "NAD+",
    #             "aKG",
    #             "CoA",
    #             "Suc",
    #             "Q",
    #             "G3P",
    #             "Pi",
    #             "NADH",
    #             "Mal",
    #             "SSA",
    #         ]
    #     ]
    # ] = Field(default=None, description="A list of substrates for the enzyme.")
    # products: Optional[
    #     List[
    #         Literal[
    #             "G6P",
    #             "ADP",
    #             "F1,6BP",
    #             "Pyr",
    #             "ATP",
    #             "Cit",
    #             "aKG",
    #             "CO2",
    #             "NADH",
    #             "SucCoA",
    #             "Fum",
    #             "QH2",
    #             "1,3-BPG",
    #             "NAD+",
    #             "OAA",
    #             "Suc",
    #         ]
    #     ]
    # ] = Field(default=None, description="A list of products for the enzyme.")
    reversible: Optional[bool] = Field(
        default=None, description="Whether the reaction is reversible."
    )
    flux: Optional[float] = Field(default=None, description="The metabolic flux value.")


class FinalFilter(BaseModel):
    """
    A wrapper for the final filter dictionary.
    The LLM should return a JSON object with a 'filter' key.
    """

    filter: Dict = Field(
        description="The final filter dictionary to be used in the retriever."
    )


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
PROMPT_FOR_FILTER_FIELDS = """
Based on the user's question, extract the relevant filtering criteria into a JSON object that conforms to the `MetadataFilterFields` schema.

**Instructions:**
- All fields in the schema are optional.
- Only populate fields that are explicitly mentioned in the user's query.
- If a field is not mentioned in the query, it MUST be omitted or set to `None`. Do not infer or guess values.

**User Question:**
{query}
"""
TEMPLATE_FOR_FILTER_FIELDS = ChatPromptTemplate.from_template(PROMPT_FOR_FILTER_FIELDS)
chain_for_filter_fields = TEMPLATE_FOR_FILTER_FIELDS | llm.with_structured_output(
    MetadataFilterFields, method="function_calling"
)
chain_for_filter_fields.invoke({"query": "enzimas del TCA que usan NADH"})

PROMPT_FOR_FILTER_GENERATION = """
You are an expert at creating database filters for a vectorstore based on natural language queries.
Your mission is to construct a precise and valid metadata filter based on a user's question and a set of pre-extracted, relevant fields and their values.

--------------------
**Filter Syntax Documentation**

Use the following operators to construct the filter.

| Filter | Description | Supported types |
| :--- | :--- | :--- |
| `$eq` | Matches values that are equal to a specified value. | `Number`, `string`, `boolean` |
| `$ne` | Matches values that are not equal to a specified value. | `Number`, `string`, `boolean` |
| `$gt` | Matches values that are greater than a specified value. | `Number` |
| `$gte` | Matches values that are greater than or equal to a specified value. | `Number` |
| `$lt` | Matches values that are less than a specified value. | `Number` |
| `$lte` | Matches values that are less than or equal to a specified value. | `Number` |
| `$in` | Matches values that are in a specified array. | `string[]`, `number[]` |
| `$nin` | Matches values that are not in a specified array. | `string[]`, `number[]` |
| `$exists` | Matches documents that contain the specified field. | `boolean` |
| `$and` | Joins clauses with a logical AND. All conditions must be met. | `Filter[]` |
| `$or` | Joins clauses with a logical OR. At least one condition must be met. | `Filter[]` |
--------------------

**Instructions**

1.  **Analyze the Query**: Carefully examine the "User Question" to understand the core intent and the logical connections (e.g., AND, OR) between different criteria.
2.  **Use Extracted Fields**: Base your filter exclusively on the keys provided in the "Extracted Relevant Fields" JSON. The values in this object are hints derived from the user's query.
3.  **Apply Operators**: Select the appropriate operators from the documentation to translate the query's logic into a filter.
    - For array-based fields, use `$in` to check for the presence of one or more items.
    - If a concept could logically appear in multiple fields, construct an `$or` condition to check all of them.
4.  **No Hallucinations**: Do NOT invent new field names or values. Adhere strictly to the provided "Extracted Relevant Fields."
5.  **Strict Output Format**: Your response MUST be a valid JSON object with a single root key "filter", as specified below.

--------------------

**Output Format (MANDATORY)**

Your response must be a JSON object with a single "filter" key.

```json
{{
  "filter": {{ /* Your generated filter object goes here */ }}
}}
```
--------------------

**Example (for guidance only)**

*   **User Question**: "Search for items in 'category_A' of type 'type_1' or 'type_2', with a value over 100."
*   **Extracted Relevant Fields**:
    ```json
    {{
      "category": "category_A",
      "type": ["type_1", "type_2"],
      "value": 100
    }}
    ```
*   **Generated Filter**:
    ```json
    {{
      "filter": {{
        "$and": [
          {{ "category": {{ "$eq": "category_A" }} }},
          {{ "type": {{ "$in": ["type_1", "type_2"] }} }},
          {{ "value": {{ "$gt": 100 }} }}
        ]
      }}
    }}
    ```
--------------------

**Now, create the filter for the following request:**

**User Question:**
{query}

**Extracted Relevant Fields (JSON):**
{extracted_fields}
"""

# Re-build the generation chain with the updated prompt
TEMPLATE_FOR_FILTER_GENERATION = ChatPromptTemplate.from_template(
    PROMPT_FOR_FILTER_GENERATION
)
chain_for_filter_generation = (
    TEMPLATE_FOR_FILTER_GENERATION
    | llm.with_structured_output(FinalFilter, method="function_calling")
)

# %%
if __name__ == "__main__":
    USER_QUERY = "enzimas del ciclo de Krebs que usan NAD y son reversibles"
    print(f"User Query: {USER_QUERY}\n")

    # Step 1: Extract relevant fields
    extracted_fields = chain_for_filter_fields.invoke({"query": USER_QUERY})
    print(f"Extracted Fields:\n{extracted_fields}\n")

    # Step 2: Generate the final filter
    extracted_json = extracted_fields.model_dump_json(indent=2)
    final_filter_obj = chain_for_filter_generation.invoke(
        {
            "query": USER_QUERY,
            "extracted_fields": extracted_json,
        }
    )
    print(f"Generated Filter Object:\n{final_filter_obj.filter}\n")

    results = retriever.invoke(USER_QUERY, filter=final_filter_obj.filter)
    for res in results:
        score = res.metadata.get("score", "N/A")
        print(f"* [score: {score:.3f}] {res.page_content} [{res.metadata}]")
