"""Module for loading few-shot examples from YAML configuration."""

# %%
from pathlib import Path
from typing import Any

import yaml


def load_examples() -> list[dict[str, Any]]:
    """Load examples from the YAML configuration file.

    Returns:
        List[Dict[str, Any]]: A list of examples with 'human' and 'ai-assistant' keys
    """
    yaml_path = Path(__file__).parent / "examples.yaml"

    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data["examples"]


if __name__ == "__main__":
    # Test the loader
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")
    print(f"First example question: {examples[0]['human']}")
