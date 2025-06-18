# %%
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# pylint: disable=import-error,wrong-import-position

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))

load_dotenv(override=True)
os.chdir(package_root)
from tpp_agentic_graph_v4.llm_chains import StructuredPandasAgent  # noqa: E402


def load_csv_dataframes(directory_path="dataframes_QA/tables"):
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


# Create the structured pandas agent
if list_of_dataframes:
    agent = StructuredPandasAgent(list_of_dataframes)

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

else:
    print("No dataframes loaded. Cannot create agent.")

if __name__ == "__main__":
    # Solo con input (básico)
    print(agent.invoke("¿Cuál es el promedio de todas las columnas?"))

    # Con algunos parámetros opcionales
    print(agent.invoke("dame los datos", nombre_de_la_tabla="df2"))

    # Con todos los parámetros (como antes)
    print(agent.invoke("dame los ultimos 9 registros.", "df2", "biomarcador", 9))

    # Usando parámetros nombrados
    print(agent.invoke("dame los datos", total_points=5, nombre_de_la_tabla="df1"))
