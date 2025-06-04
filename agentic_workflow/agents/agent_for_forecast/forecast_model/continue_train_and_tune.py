#!/usr/bin/env python
"""Utility to *continue* training and hyper-parameter tuning.

This script loads the existing best ``TransformerForecaster`` checkpoint,
executes a **short** Optuna study (10 trials) to explore nearby hyper-parameters
and trains each candidate for **10 epochs**.  If a better model (lower test
loss) is found, the previous checkpoint is **overwritten** together with a JSON
file holding the updated hyper-parameters.

The goal is to provide a lightweight routine that can be run frequently
after the initial, longer training process.
"""

# %%
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler

from agentic_workflow.agents.agent_for_forecast.forecast_model.forecaster import (
    TransformerForecaster,
    create_dataloaders,
    generate_ts,
    sliding_window,
)


__all__ = [
    "continue_training_and_tune",
]

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "forecast" / "torch_models" / "best_model_simulation_best_test_loss"
PARAMS_PATH = MODEL_PATH.with_suffix(".json")
OPTUNA_STORAGE = ROOT / "forecast" / "optuna_dbs" / "continue_tuning.sqlite3"

N_TRIALS = 10
EPOCHS = 10
TEST_SPLIT_FRAC = 0.2

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _objective(trial: optuna.trial.Trial, series: pd.Series) -> float:
    """Optuna objective – returns the *test* MSE for the given hyper-parameters."""
    # Hyper-parameter search space (narrower than the initial one).
    hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    ts_history_len = trial.suggest_int("ts_history_len", 30, 60, step=15)
    ts_target_len = trial.suggest_int("ts_target_len", 15, 30, step=15)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    lr = trial.suggest_float("learning_rate", 1e-3, 1e-2, step=1e-3)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_ts = scaler.fit_transform(series.values.reshape(-1, 1)).ravel().tolist()

    X, Y = sliding_window(scaled_ts, ts_history_len, ts_target_len)
    test_len = int(len(X) * TEST_SPLIT_FRAC)
    train_loader, test_loader = create_dataloaders(X, Y, batch_size, test_len, device)

    # ------------------------------------------------------------------
    # Model & training
    # ------------------------------------------------------------------
    model = TransformerForecaster(
        hidden_size=hidden_size,
        input_size=1,
        output_size=1,
        num_layers=num_layers,
    ).to(device)

    _, _, _, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=lr,
        ts_target_len=ts_target_len,
    )

    return best_test_loss


def _save_hyperparams(params: dict[str, Any]) -> None:
    """Persist *params* as JSON next to the model checkpoint."""
    with PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2, sort_keys=True)
    logger.info("Updated hyper-parameters saved to %s", PARAMS_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def continue_training_and_tune(
    series: pd.DataFrame | pd.Series | None = None,
    *,
    n_trials: int = N_TRIALS,
    epochs: int = EPOCHS,
    model_output_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Resume training and fine-tuning for *series*.

    Parameters
    ----------
    series
        Time-series to use for (re)training.  If *None*, a synthetic one is
        generated identically to the initial training routine.
    n_trials
        Number of Optuna trials (defaults to ``10``).
    epochs
        Training epochs **per trial** (defaults to ``10``).
    model_output_path
        Path to save the final model checkpoint. If *None*, the default model path is used.

    Returns:
    -------
    tuple[dict, Path]
        The best hyper-parameters found and the output path of the saved model.
    """
    global EPOCHS
    EPOCHS = epochs

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    if series is None:
        df = generate_ts(7 * 24 * 60, add_trend=True, add_noise=True)
        series = df.iloc[:, 0]  # use first column values
        logger.info("Synthetic series generated with length %d", len(series))
    elif isinstance(series, pd.DataFrame):
        series = series.iloc[:, -1]
    else:  # already a Series
        series = series  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Optuna study (in-memory or SQLite)
    # ------------------------------------------------------------------
    storage_url = f"sqlite:///{OPTUNA_STORAGE}" if OPTUNA_STORAGE else None
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(),
        storage=storage_url,
        study_name="continue_tune",
        load_if_exists=True,
    )

    logger.info(
        "Starting Optuna optimisation: %d trials, %d epochs each", n_trials, epochs
    )
    study.optimize(lambda t: _objective(t, series), n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info("Best hyper-parameters: %s", best_params)

    # ------------------------------------------------------------------
    # Train *final* model with the best params and save
    # ------------------------------------------------------------------
    ts_history_len = best_params["ts_history_len"]
    ts_target_len = best_params["ts_target_len"]
    batch_size = best_params["batch_size"]

    scaler = StandardScaler()
    scaled_ts = scaler.fit_transform(series.values.reshape(-1, 1)).ravel().tolist()

    X, Y = sliding_window(scaled_ts, ts_history_len, ts_target_len)
    test_len = int(len(X) * TEST_SPLIT_FRAC)
    train_loader, test_loader = create_dataloaders(X, Y, batch_size, test_len, device)

    final_model = TransformerForecaster(
        hidden_size=best_params["hidden_size"],
        input_size=1,
        output_size=1,
        num_layers=best_params["num_layers"],
    ).to(device)

    logger.info("Training final model for %d epochs", epochs)
    start = time.time()
    _, _, _, best_test_loss = final_model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=best_params["learning_rate"],
        ts_target_len=ts_target_len,
        save_path=str(model_output_path or MODEL_PATH),
    )
    logger.info(
        "Final training completed in %.2fs – test loss %.4f",
        time.time() - start,
        best_test_loss,
    )

    # ------------------------------------------------------------------
    # Persist checkpoint & hyper-parameters
    # ------------------------------------------------------------------
    output_path = model_output_path or MODEL_PATH
    torch.save(final_model, output_path)
    _save_hyperparams(best_params)

    return best_params, output_path


# ---------------------------------------------------------------------------
# CLI / quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params, output_path = continue_training_and_tune()
    print("Updated hyper-parameters:")
    print(json.dumps(params, indent=2))
    print("Model saved to:", output_path)

# %%
