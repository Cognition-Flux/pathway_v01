"""Module for loading few-shot examples from YAML configuration."""

# %%
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

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


def load_examples() -> list[dict[str, Any]]:
    """Load examples from the YAML configuration file.

    Returns:
        List[Dict[str, Any]]: A list of examples with 'human' and 'ai-assistant' keys
    """
    yaml_path = package_root / "fewshot/examples.yaml"

    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data["examples"]


if __name__ == "__main__":
    # Test the loader
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")
    print(f"First example question: {examples[0]['human']}")
