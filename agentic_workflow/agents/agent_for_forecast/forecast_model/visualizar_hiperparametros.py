# %%
from optuna_dashboard import run_server


storage = "sqlite:///agentic_workflow/agents/agent_for_forecast/forecast_model/forecast/optuna_dbs/Transformer_hyperparams_opt.sqlite3"
run_server(storage, host="localhost", port=8081)
# %%
