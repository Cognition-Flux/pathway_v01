"""Module for reducer functions used in the LangGraph state."""

# %%
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document

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


# Custom reducer for steps to ensure proper step handling
def steps_reducer(left: list[str], right: list[str]) -> list[str]:
    """Ensure proper handling of step updates.

    This function is designed to handle three cases:
    1. Set steps (replace) - when right list has the 'REPLACE_STEPS' marker
    2. Update steps (refine) - when right list has steps to replace the current ones
    3. Remove a step (default) - when right is None or empty, remove the first step

    Args:
        left: Current list of steps
        right: New list of steps or update instruction

    Returns:
        Updated list of steps

    """
    # Handle None values
    if left is None:
        left = []

    if right is None or len(right) == 0:
        # Default behavior: remove the first step (if any)
        return left[1:] if left else []

    # Special marker for complete replacement of steps
    if len(right) > 0 and right[0] == "REPLACE_STEPS":
        # Replace all steps with the new ones (excluding the marker)
        return right[1:] if len(right) > 1 else []

    # Default: update/replace with the new steps
    return right


def web_results_reducer(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | Literal["delete"] | None,
) -> list[dict[str, Any]]:
    """Reduce and process web search results, handling duplicates.

    This reducer is designed to handle web search results from multiple queries,
    combining them while eliminating duplicates based on URL.

    The expected structure for each item is:
    {
        'query': str,  # The original search query
        'results': [   # List of search results for this query
            {
                'title': str,
                'url': str,
                'content': str,
                'score': float,
                'raw_content': str (optional)
            },
            ...
        ]
    }

    Args:
        existing: The existing web search results in the state, if any
        new: The new web search results to process, or "delete" to clear all results

    Returns:
        A combined list of web search results with duplicates removed
    """
    if new == "delete":
        return []

    existing_list = existing or []
    if not new:
        return existing_list

    existing_results_map: dict[str, bool] = {}
    for item in existing_list:
        query = item.get("query", "")
        if "results" in item and isinstance(item["results"], list):
            for result in item["results"]:
                if "url" in result:
                    key = f"{query}:{result['url']}"
                    existing_results_map[key] = True

    combined_results = list(existing_list)

    for item_new in new:
        query = item_new.get("query", "")

        if "error" in item_new:
            combined_results.append(item_new)
            continue

        if "results" in item_new and isinstance(item_new["results"], list):
            filtered_item_results = []
            for result in item_new["results"]:
                if "url" in result:
                    key = f"{query}:{result['url']}"
                    if key not in existing_results_map:
                        filtered_item_results.append(result)
                        existing_results_map[key] = True

            if filtered_item_results:
                combined_results.append(
                    {"query": query, "results": filtered_item_results}
                )

    return combined_results


def _generate_element_id(page_content: str) -> str:
    """Generate an element_id for a document based on page_content."""
    md5_hash = hashlib.md5(
        page_content.encode(),
        usedforsecurity=False,
    ).hexdigest()
    return md5_hash


def reduce_docs(
    existing: list[Document] | None,
    new: list[Document] | list[dict[str, Any]] | list[str] | str | Literal["delete"],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a list
    of Document objects.
    It performs several key functions:
    1. Deletes existing documents if the 'delete' command is received
    2. Creates new Document objects from strings or dictionaries
    3. Combines new documents with existing ones, avoiding duplicates by
       tracking element_ids
    4. Ensures all Documents have a unique element_id for identification

    This function is used as an annotation for Document lists in state management to
    handle document reductions in graph-based workflows.

    Args:
        existing: The existing docs in the state, if any
        new: The new input to process, which can be a list of Documents, dictionaries,
             strings, a single string, or the literal "delete"

    Returns:
        A combined list of Document objects with duplicates removed

    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []

    if isinstance(new, str):
        doc = Document(
            page_content=new,
            metadata={"element_id": _generate_element_id(new)},
        )
        return [*existing_list, doc]

    new_docs_to_add = []
    if isinstance(new, list):
        existing_ids = {
            doc.metadata.get("element_id")
            for doc in existing_list
            if doc.metadata.get("element_id")
        }

        for item in new:
            item_id = None
            doc_to_add = None

            if isinstance(item, str):
                item_id = _generate_element_id(item)
                if item_id not in existing_ids:
                    doc_to_add = Document(
                        page_content=item,
                        metadata={"element_id": item_id},
                    )

            elif isinstance(item, dict):
                page_content = item.get("page_content", "")
                metadata = item.get("metadata", {})
                item_id = metadata.get("element_id") or _generate_element_id(
                    page_content
                )

                if item_id not in existing_ids:
                    doc_to_add = Document(
                        page_content=page_content,
                        metadata={**metadata, "element_id": item_id},
                    )

            elif isinstance(item, Document):
                item_id = item.metadata.get("element_id")
                if not item_id:
                    item_id = _generate_element_id(item.page_content)
                    doc_to_add = Document(
                        page_content=item.page_content,
                        metadata={**item.metadata, "element_id": item_id},
                    )

                elif item_id not in existing_ids:
                    doc_to_add = item

            if doc_to_add and item_id:
                new_docs_to_add.append(doc_to_add)
                existing_ids.add(item_id)

    return existing_list + new_docs_to_add


def merge_reasoning(
    existing: str | list[str] | None, new: str | list[str]
) -> str | list[str]:
    """Merge reasoning strings from multiple nodes.

    This function concatenates new reasoning information with existing reasoning,
    adding a delimiter between them to maintain readability. It can handle both
    string and list of strings.

    Args:
        existing: The existing reasoning string or list of strings, if any
        new: The new reasoning to add, which can be a string or list of strings

    Returns:
        A concatenated string with all reasoning information, or a list of messages

    """
    # Convert lists to strings to avoid showing raw Python list representation
    if isinstance(new, list):
        new = "\n".join(new)

    if isinstance(existing, list):
        existing = "\n".join(existing)

    # Both are now strings or None, handle as before
    if not existing:
        return new
    if not new:
        return existing

    # Add a delimiter to separate different reasoning chunks
    return f"{existing}\n\n---\n\n{new}"


# NEW: Reducer for executed steps


def executed_steps_reducer(
    existing: list[Any] | None,
    new: list[Any] | Literal["delete"] | None,
) -> list[Any]:
    """Append new executed steps to the existing list.

    This reducer safely concatenates lists of executed steps coming from
    different nodes. It supports three operations:

    1. Delete – if *new* is the literal ``"delete"`` the current list is
       cleared.
    2. No-op  – if *new* is ``None`` it returns the current list untouched.
    3. Append – otherwise it appends the *new* steps to the current list,
       avoiding duplicates while preserving order.
    """
    # Handle delete instruction
    if new == "delete":
        return []

    # Normalise *existing* and *new*
    existing_list = existing or []
    new_list: list[Any] = new or []

    # Fast-path when there is nothing to append
    if not new_list:
        return existing_list

    # Build a set of already seen items (by identity) to avoid duplicates.
    # We use ``id`` because OneStep instances are not hashable by default.
    seen_ids = {id(item) for item in existing_list}

    appended: list[Any] = []
    for item in new_list:
        if id(item) not in seen_ids:
            appended.append(item)
            seen_ids.add(id(item))

    return existing_list + appended


# NEW: Reducer for scratchpad messages


def scratchpad_reducer(
    existing: list[Any] | None,
    new: list[Any] | Any | Literal["delete"] | None,
) -> list[Any]:
    """Accumulate messages in ``scratchpad`` reliably.

    This reducer is intentionally forgiving with respect to the *new* payload:

    * ``"delete"`` → clears the scratchpad.
    * ``None``      → leaves the scratchpad untouched.
    * *single item* → wraps it in a list and appends it.
    * *list*        → appends every element of the list.

    It removes duplicates across the *whole* scratchpad based on the
    ``content`` attribute (when present) or falls back to the object itself
    (useful when *msg* is a plain string). Order is preserved.
    """

    # 1️⃣ Delete everything if explicitly requested
    if new == "delete":
        return []

    # 2️⃣ Normalise current and incoming data
    existing_list: list[Any] = existing or []

    if new is None:
        new_list: list[Any] = []
    elif isinstance(new, list):
        new_list = new
    else:  # Accept single BaseMessage / str / any object
        new_list = [new]

    # Early-exit when nothing to append
    if not new_list:
        return existing_list

    # 3️⃣ Build a set with already present *contents* for fast look-ups
    existing_contents = {getattr(msg, "content", msg) for msg in existing_list}

    # 4️⃣ Append unique items while preserving order
    output: list[Any] = existing_list.copy()
    for msg in new_list:
        content_key = getattr(msg, "content", msg)
        if content_key not in existing_contents:
            output.append(msg)
            existing_contents.add(content_key)

    return output


# %%
