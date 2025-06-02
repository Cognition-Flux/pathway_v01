"""Web search tool implementation using Tavily Search API to perform
searches with customizable parameters, including domain filtering and
customizable parameters, including domain filtering and
multiple query support.
"""

# %%
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables
load_dotenv(override=True)

# Scientific and academic domains to filter search results
#
# The list is loaded from ``domains.yaml`` located in the same directory
# to allow easy customization without touching the code.


def _load_domains() -> list[str]:
    """Load the list of scientific domains from the YAML file.

    The YAML file can contain either a bare list or a mapping with the key
    ``domains`` holding the list. If the file is not present or any error
    occurs while parsing, a RuntimeError is raised to make the issue
    explicit at import-time.
    """
    yaml_path = Path(__file__).with_name("domains.yaml")

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Domain file not found: {yaml_path}. Please create it or adjust the path."
        )

    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error parsing domain YAML file: {exc}") from exc

    # The YAML can be either a list or a dict with "domains" key.
    domains_list: list[str]
    if isinstance(data, list):
        domains_list = [str(item) for item in data]
    elif isinstance(data, dict) and "domains" in data:
        raw_list = data["domains"]
        if not isinstance(raw_list, list):
            raise ValueError("The 'domains' key must contain a list.")
        domains_list = [str(item) for item in raw_list]
    else:
        raise ValueError(
            "YAML file must contain either a list or a mapping with a 'domains' key."
        )

    if not domains_list:
        raise ValueError("Domain list loaded from YAML is empty.")

    return domains_list


# Load domains at import-time to fail fast if configuration is missing
domains = _load_domains()

# PARAMETERS
max_results = 2
include_raw_content = True
include_domains = domains
include_images = False  # No incluir imágenes en los resultados
include_answer = False  # No incluir una respuesta generada por IA
search_depth = "advanced"  # Basic or advanced search depth
time_range = "year"  # Rango de tiempo de los resultados (último año)
language = "en"  # Idioma de los resultados


def create_search_tool() -> TavilySearchResults:
    """Create and configure a TavilySearchResults tool with the specified parameters.

    Returns:
        TavilySearchResults: Configured search tool instance
    """
    return TavilySearchResults(
        max_results=max_results,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        include_images=include_images,
        include_answer=include_answer,
        search_depth=search_depth,
    )


def search_multiple_queries(queries: list[str]) -> list[dict[str, Any]]:
    """Perform searches for multiple queries using the configured search tool.

    Args:
        queries: List of search queries to process

    Returns:
        List of search results for each query with the following structure:
        [
            {
                'query': str,  # The original search query
                'results': [   # List of search results for this query
                    {
                        'title': str,  # Title of the search result
                        'url': str,    # URL of the result
                        'content': str,  # Summary/snippet of the content
                        'score': float,  # Relevance score (0-1)
                        'raw_content': str,  # Optional full content if include_raw_content=True
                    },
                    ...
                ]
            },
            ...
        ]

        If an error occurs during the search, the result will have this structure:
        {
            'query': str,  # The original search query
            'error': str   # Error message
        }
    """
    search_tool = create_search_tool()
    results = []

    for query in queries:
        try:
            result = search_tool.invoke(query)
            results.append({"query": query, "results": result})
        except Exception as e:
            results.append({"query": query, "error": str(e)})

    return results


if __name__ == "__main__":
    # Check if Tavily API key is set
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable not set")
        exit(1)
    # Example queries
    example_queries = [
        "diet and longevity",
        "brain and metabolism",
    ]

    print(f"Searching for {len(example_queries)} queries...")
    search_results = search_multiple_queries(example_queries)

    # Display results
    for result in search_results:
        print(f"\nQuery: {result['query']}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Results: {result['results']}")
