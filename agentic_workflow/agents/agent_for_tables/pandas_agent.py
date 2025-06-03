# %%
import os

import pandas as pd
import yaml  # Added import
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

from agentic_workflow.utils import get_llm


def load_csv_dataframes(directory_path="agentic_workflow/dataframes_QA/tables"):
    """Load all CSV files from a directory into pandas DataFrames.

    Args:
        directory_path: Path to the directory containing CSV files

    Returns:
        list: List of tuples containing (filename, dataframe)

    """
    list_of_dataframes = []

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return list_of_dataframes

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {directory_path}.")
        return list_of_dataframes

    # Load each CSV file into a DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            # Remove .csv extension for the name
            name = os.path.splitext(csv_file)[0]
            # Add name as attribute to the dataframe for easy identification
            df.name = name
            list_of_dataframes.append((name, df))
            print(
                f"Loaded {name} with {len(df)} rows and {len(df.columns)} columns",
            )
        except Exception as e:
            print(f"Error loading {csv_file}: {e!s}")

    return list_of_dataframes


# Load all dataframes from the tables directory
list_of_dataframes = load_csv_dataframes()

# Create the pandas dataframe agent
if list_of_dataframes:
    # Extract just the dataframes for the agent (without names)
    dataframes = [df for _, df in list_of_dataframes]

    agent = create_pandas_dataframe_agent(
        get_llm(provider="azure", model="gpt-4.1"),
        dataframes,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        max_iterations=100,
        max_execution_time=600.0,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    # Example of how to use the agent
    if __name__ == "__main__":
        print("\nAvailable dataframes:")
        loaded_tables_info = []  # Initialize list to store table info
        for i, (name, df) in enumerate(list_of_dataframes):
            print(f"- {i}: {name}: {df.shape[0]} rows x {df.shape[1]} columns")
            # Store info for YAML file
            loaded_tables_info.append(
                {
                    "name": name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                },
            )

        # Save the loaded table information to a YAML file
        yaml_file_path = os.path.join(
            os.path.dirname(__file__),
            "loaded_tables.yaml",
        )
        try:
            with open(yaml_file_path, "w") as f:
                yaml.dump({"loaded_tables": loaded_tables_info}, f, indent=2)
            print(f"\nSuccessfully saved loaded table info to {yaml_file_path}")
        except Exception as e:
            print(f"\nError saving table info to YAML: {e!s}")

        # Example query - uncomment to run
        # query = "What's the average BMI across all people? And what's the average daily calorie intake?"
        # print("\nQuery:", query)
        # result = agent.run(query)
        # print("\nResult:", result)
else:
    print("No dataframes loaded. Cannot create agent.")
