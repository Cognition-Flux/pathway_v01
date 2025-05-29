#!/bin/bash
# This script runs Ruff to check and format the Python code in the current project.

# Ensure the script exits on any error
set -e

# Run Ruff to check for linting errors and apply fixes
echo "Running Ruff linter and applying fixes..."
./.venv/bin/ruff check --fix .

# Run Ruff to format the code
echo "Running Ruff formatter..."
./.venv/bin/ruff format .

echo "Ruff check and formatting complete."
