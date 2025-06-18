"""Module for retrieving few-shot examples."""

# %%
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever

# pylint: disable=import-error,wrong-import-position
load_dotenv(override=True)

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
os.chdir(package_root)
from tpp_agentic_graph_v4.fewshot.loader import load_examples  # noqa: E402


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
        [doc.page_content for doc in retriever.invoke("como me puedes ayudar")[:top_k]]
    )
    print(examples_str)
