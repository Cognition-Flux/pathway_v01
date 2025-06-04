# %%
import os
from pathlib import Path

from optuna_dashboard import run_server


# Define Project Root
PROJECT_ROOT = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)

# Construct the absolute path for the SQLite database
db_relative_path = "agentic_workflow/agents/agent_for_forecast/forecast_model/forecast/optuna_dbs/Transformer_hyperparams_opt.sqlite3"
db_abs_path = PROJECT_ROOT / db_relative_path
storage = f"sqlite:///{db_abs_path}"

run_server(storage, host="localhost", port=8081)
# %%
