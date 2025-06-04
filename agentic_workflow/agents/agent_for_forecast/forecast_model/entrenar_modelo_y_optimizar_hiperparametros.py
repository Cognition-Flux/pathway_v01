# %%
import logging
import os
import time
import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from pyentrp import entropy as ent
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from agentic_workflow.agents.agent_for_forecast.forecast_model.forecaster import (
    TransformerForecaster,
    create_dataloaders,
    device,
    generate_ts,
    log_elapsed_time,
    objective,
    sliding_window,
    visualize_predictions,
)


# Configuración de logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Define Project Root
# Assuming this script is in agentic_workflow/agents/agent_for_forecast/forecast_model/
# Adjust the number of ".." based on the actual location relative to the project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

# Parámetros clave para la optimización y entrenamiento

# n_trials: Define la cantidad de intentos en la búsqueda de hiperparámetros óptimos usando Optuna.
# Un número mayor de trials permite explorar un espacio más amplio, incrementando la posibilidad
# de encontrar una configuración óptima, aunque a costa de mayor tiempo computacional.
# recomendado: 50
n_trials: int = 50

# epochs: Especifica cuántas épocas (pasadas completas del conjunto de entrenamiento)
# se realizarán durante el entrenamiento del modelo. Más épocas pueden permitir un aprendizaje
# más profundo, pero también aumentan el riesgo de sobreajuste.
# recomendado: 200
epochs: int = 200


def prepare_time_series(
    series_df: pd.DataFrame, best_params: dict[str, Any]
) -> list[float]:
    """Prepara y escala la serie temporal.

    Parameters:
    - series_df: DataFrame que contiene la serie temporal.
    - best_params: Diccionario con los parámetros óptimos, se utiliza para truncar la serie.

    Returns:
    - ts_scaled: Lista de valores de la serie temporal escalados.
    """
    # Se asume que la columna 1 contiene los datos de la serie.
    series_df.columns[1]
    ts = series_df.iloc[:, 1].tolist()
    # Seleccionar los últimos 'largo_dataset * 60' valores.
    ts = ts[-best_params["largo_dataset"] * 60 :]
    # Escalar la serie.
    data = np.array(ts).reshape(-1, 1)
    scaler = StandardScaler()
    ts_scaled: list[float] = scaler.fit_transform(data).reshape(-1).tolist()
    return ts_scaled


def optimize_hyperparameters(
    n_trials: int, series_list: list[pd.DataFrame], series_index: int, storage: str
) -> dict[str, Any]:
    """Optimiza los hiperparámetros usando Optuna para una serie temporal.

    Parameters:
    - n_trials: Número de intentos de optimización.
    - series_list: Lista de DataFrames con series temporales.
    - series_index: Índice de la serie a procesar.
    - storage: Cadena de conexión para la base de datos de Optuna.

    Returns:
    - best_params: Diccionario con los mejores parámetros encontrados.
    """
    series_df: pd.DataFrame = series_list[series_index]
    study_name: str = series_df.columns[1]
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        logger.info(f"Estudio '{study_name}' no estaba guardado previamente.")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        sampler=TPESampler(),
    )

    study.optimize(
        lambda trial: objective(trial, series_index=series_index, series=series_list),
        n_trials=n_trials,
        n_jobs=1,
    )

    logger.info(f"Best hyperparameters for '{study_name}': {study.best_params}")
    logger.info(f"Best trial MSE: {study.best_trial.values[0]}")
    return study.best_params


def process_data(
    ts: list[float], best_params: dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    """Procesa la serie temporal para crear ventanas de datos y generar los dataloaders.

    Parameters:
    - ts: Lista con la serie temporal escalada.
    - best_params: Diccionario con parámetros como 'ts_history_len', 'batch_size' y 'ts_target_len'.

    Returns:
    - Tuple[DataLoader, DataLoader]: Dataloader para entrenamiento y para prueba.
    """
    test_ds_len: int = int(len(ts) * 0.2)
    ts_target_len: int = best_params["ts_target_len"]
    X, Y = sliding_window(ts, best_params["ts_history_len"], ts_target_len)
    train_loader, test_loader = create_dataloaders(
        X, Y, best_params["batch_size"], test_ds_len, device
    )
    return train_loader, test_loader


def train_and_evaluate_model(
    epochs: int,
    series_name: str,
    best_params: dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    ts: list[float],
    ts_target_len: int,
) -> None:
    """Inicializa, entrena y evalúa el modelo Transformer, además de visualizar las predicciones.

    Parameters:
    - epochs: Número de épocas de entrenamiento.
    - series_name: Nombre de la serie temporal (para títulos y rutas de guardado).
    - best_params: Diccionario con hiperparámetros óptimos.
    - train_loader: Dataloader de entrenamiento.
    - test_loader: Dataloader de prueba.
    - ts: Lista con la serie temporal escalada.
    - ts_target_len: Longitud de la ventana objetivo para las predicciones.
    """
    # Inicializar el modelo.
    model_init_start: float = time.time()
    logger.info(f"Using device: {device}")
    model = TransformerForecaster(
        hidden_size=best_params["hidden_size"],
        input_size=1,
        output_size=1,
        num_layers=best_params["num_layers"],
    ).to(device)
    log_elapsed_time(model_init_start, "Model initialization")

    # Entrenar el modelo.
    training_start: float = time.time()
    model.train()

    # Construct save path and ensure directory exists
    model_save_path = os.path.join(
        PROJECT_ROOT,
        "agentic_workflow",
        "agents",
        "agent_for_forecast",
        "forecast",
        "torch_models",
        f"best_model_{series_name}",
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    train_losses, test_losses, best_model_path, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=best_params["learning_rate"],
        ts_target_len=ts_target_len,
        save_path=model_save_path,
        trial=None,
    )
    log_elapsed_time(training_start, "Model training")

    # Visualizar las predicciones.
    # Construct figure save path and ensure directory exists
    figure_save_path = os.path.join(
        PROJECT_ROOT,
        "agentic_workflow",
        "agents",
        "agent_for_forecast",
        "forecast",
        "figures",
        f"{series_name}.png",
    )
    os.makedirs(os.path.dirname(figure_save_path), exist_ok=True)

    visualize_predictions(
        best_model_path,
        test_loader,
        ts_target_len,
        main_title=series_name,
        MSE=test_losses[-1],
        entropy=(
            round(ent.shannon_entropy(ts), 3),
            round(ent.permutation_entropy(ts), 3),
        ),
        save_path=figure_save_path,
        num_samples=5,
        full_length_ts=ts,
    )


def main() -> None:
    """Función principal para ejecutar el entrenamiento y la optimización del modelo."""
    sim: pd.DataFrame = generate_ts(
        7 * 24 * 60, add_trend=True, add_noise=True
    ).reset_index(drop=False, names=["time"])
    series_list: list[pd.DataFrame] = [sim]
    # Construct the absolute path for the SQLite database
    db_path = os.path.join(
        PROJECT_ROOT,
        "agentic_workflow",
        "agents",
        "agent_for_forecast",
        "forecast",
        "optuna_dbs",
        "Transformer_hyperparams_opt.sqlite3",
    )

    # Ensure the directory exists before creating the database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    storage: str = f"sqlite:///{db_path}"
    for series_index, series_df in enumerate(series_list):
        series_name: str = series_df.columns[1]
        logger.info(f"Processing series: {series_name}")

        # Optimización de hiperparámetros.
        hyperparams_start: float = time.time()
        best_params: dict[str, Any] = optimize_hyperparameters(
            n_trials=n_trials,
            series_list=series_list,
            series_index=series_index,
            storage=storage,
        )
        log_elapsed_time(hyperparams_start, "Hyperparameter optimization")

        # Preparación de la serie temporal.
        data_prep_start: float = time.time()
        ts: list[float] = prepare_time_series(series_df, best_params)
        log_elapsed_time(data_prep_start, "Data preparation")

        # Procesamiento de los datos para entrenar el modelo.
        data_process_start: float = time.time()
        train_loader, test_loader = process_data(ts, best_params)
        log_elapsed_time(data_process_start, "Data processing")

        # Entrenamiento y evaluación del modelo.
        train_and_evaluate_model(
            epochs=epochs,
            series_name=series_name,
            best_params=best_params,
            train_loader=train_loader,
            test_loader=test_loader,
            ts=ts,
            ts_target_len=best_params["ts_target_len"],
        )


if __name__ == "__main__":
    main()

# %%
