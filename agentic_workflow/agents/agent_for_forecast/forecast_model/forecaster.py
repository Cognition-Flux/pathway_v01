#!/usr/bin/env python
"""Forecasting module refactorizado.

Este módulo implementa el entrenamiento y evaluación de un modelo Transformer para series temporales,
incluyendo generación de datos sintéticos, procesamiento, visualización de resultados y tuning de
hiperparámetros mediante Optuna.

Las secciones principales son:
  - Configuración y utilidades
  - Funciones de generación y procesamiento de series temporales
  - Dataset y funciones para DataLoader
  - Definición del modelo Transformer (incluyendo codificación posicional, transformer y forecaster)
  - Funciones de visualización
  - Función objetivo para Optuna
  - Función main (ejecución principal)
"""

# %%
import logging
import math
import random
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from pyentrp import entropy as ent
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuración del logger y del dispositivo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Current device: {device}")


# ---------------------------------------------------------------------------
# UTILS: Funciones auxiliares
# ---------------------------------------------------------------------------
def log_elapsed_time(start: float, description: str) -> None:
    """Registra el tiempo transcurrido desde 'start' con el mensaje 'description'."""
    elapsed_time = time.time() - start
    logger.info(f"{description} completed in {elapsed_time:.2f} seconds")


# ---------------------------------------------------------------------------
# DATA GENERATION & PROCESSING
# ---------------------------------------------------------------------------
def generate_ts(
    length: int,
    freq_cycles_factor: int = 20,
    amplitude_variation: float = 0.5,
    add_trend: bool = False,
    trend_strength: float = 320,
    trend_cycle_length: int = 20,
    add_noise: bool = False,
) -> pd.DataFrame:
    """Genera una serie temporal sintética.

    Parámetros:
      length: Número de puntos en la serie.
      freq_cycles_factor: Factor que aumenta la frecuencia de oscilación.
      amplitude_variation: Variación de la amplitud de la señal.
      add_trend: Si se añade una tendencia global.
      trend_strength: Fuerza de la tendencia global.
      trend_cycle_length: Longitud cíclica de la tendencia.
      add_noise: Si se añade ruido a la señal.
    """
    # Creación de un vector de tiempo
    t = np.linspace(0.0, 80 * np.pi, length)

    # Amplitud variable
    amplitude = 1 + amplitude_variation * np.sin(2 * t)

    # Dirección cíclica para la tendencia
    trend_direction = np.sin(2 * np.pi * t / trend_cycle_length)

    # Composición de la señal
    y = amplitude * (
        np.sin(5 * freq_cycles_factor * t) ** 2
        + np.cos(1 * freq_cycles_factor * t) ** 3
        + (np.sin(5 * freq_cycles_factor * t) * np.cos(1 * freq_cycles_factor * t))
        + 2
    )
    if add_noise:
        y += np.random.normal(0, np.std(y) / 3, length)

    if add_trend:
        global_trend = trend_strength * np.cumsum(trend_direction) / length
        y += global_trend

    start_time = datetime.now().replace(microsecond=0)
    time_index = [start_time + timedelta(minutes=i) for i in range(length)]
    return pd.DataFrame(y, index=time_index, columns=["simulation"])


def rewrite_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Reescribe la primera columna de un DataFrame para generar nuevos timestamps."""
    if df.empty or df.columns.empty:
        return df

    first_column = df.columns[0]
    start_time = pd.to_datetime(df.iloc[0, 0])
    new_timestamps = [
        (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(len(df))
    ]
    df[first_column] = new_timestamps
    return df


def sliding_window(ts: list, features: int, target_len: int = 1):
    """Aplica una técnica de sliding window a la serie de tiempo.

    Parámetros:
      ts: Serie temporal como lista de valores.
      features: Número de puntos para la secuencia de entrada.
      target_len: Número de puntos a predecir.

    Retorna:
      (X, Y): Tupla de listas para input y target.
    """
    X, Y = [], []
    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len) : i - target_len])
        Y.append(ts[i - target_len : i])
    return X, Y


# ---------------------------------------------------------------------------
# DATASET & DATALOADERS
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Dataset para series temporales. Conversión a tensores y adición de dimensión para features."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X).float().unsqueeze(2)
        self.Y = torch.tensor(Y).float().unsqueeze(2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_dataloaders(X, Y, batch_size: int, test_ds_len: int, device):
    """Crea DataLoaders de entrenamiento y prueba."""
    ds_len = len(X)
    train_dataset = TimeSeriesDataset(
        X[: ds_len - test_ds_len], Y[: ds_len - test_ds_len]
    )
    test_dataset = TimeSeriesDataset(
        X[ds_len - test_ds_len :], Y[ds_len - test_ds_len :]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# MODEL DEFINITION: Transformer para Forecasting
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Codificación posicional para información secuencial."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Suma la codificación posicional al tensor de entrada."""
        return x + self.pe[: x.size(0), :]


class TransformerModel(nn.Module):
    """Modelo Transformer básico que procesa la entrada y genera una predicción."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, src, tgt):
        """Proceso de propagación para el Transformer."""
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        tgt = self.input_proj(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        return self.output_proj(output)


class TransformerForecaster(nn.Module):
    """Forecaster basado en Transformer para series temporales."""

    def __init__(
        self,
        hidden_size: int,
        input_size: int = 1,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.15,
        weight_decay: float = 0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.transformer = TransformerModel(
            input_size=input_size,
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
        )
        self.weight_decay = weight_decay

    def train_model(
        self,
        train_loader,
        test_loader,
        epochs: int,
        lr: float = 0.01,
        ts_target_len: int = 1,
        save_path: str | None = None,
        trial=None,
        last_best_model_path: str | None = None,
    ):
        """Entrena el modelo, evaluando en cada época y guardando el mejor modelo basado en test loss.

        Retorna:
          train_losses, test_losses, last_best_model_path, best_test_loss
        """
        train_losses = torch.full((epochs,), float("nan"))
        test_losses = torch.full((epochs,), float("nan"))
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        best_test_loss = float("inf")

        logger.info(f"Starting training for {epochs} epochs")
        for e in range(epochs):
            self.train()
            epoch_loss = 0
            for _batch_idx, (train_data, target) in enumerate(train_loader):
                train_data, target = train_data.to(device), target.to(device)
                optimizer.zero_grad()
                src = train_data.transpose(0, 1)
                tgt = torch.zeros_like(target).transpose(0, 1)
                predicted = self.transformer(src, tgt)
                predicted = predicted.transpose(0, 1)
                loss = criterion(predicted, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses[e] = avg_train_loss

            avg_test_loss = self.evaluate(test_loader, ts_target_len)
            test_losses[e] = avg_test_loss

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                if save_path:
                    last_best_model_path = f"{save_path}_best_test_loss"
                    torch.save(self, last_best_model_path)
                    logger.info(
                        f"#---------------------------New best model saved at epoch {e} with test loss: {best_test_loss:.4f} ----------------------------------#"
                    )

            if e % 2 == 0:
                logger.info(
                    f"Iterando: Epoch {e}/{epochs}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
                )

            if trial is not None:
                trial.report(avg_test_loss, e)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return train_losses, test_losses, last_best_model_path, best_test_loss

    def predict(self, x, target_len: int):
        """Realiza la predicción para una secuencia de entrada x."""
        self.eval()
        x = x.to(device)
        batch_size, seq_len, _ = x.size()
        src = x.transpose(0, 1)
        tgt = torch.zeros(target_len, batch_size, self.input_size, device=device)
        with torch.no_grad():
            output = self.transformer(src, tgt)
        return output.transpose(0, 1)

    def evaluate(self, test_loader, ts_target_len: int) -> float:
        """Evalúa el modelo en el conjunto de test utilizando MSE."""
        self.eval()
        test_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for _batch_idx, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_pred = self.predict(x_test, ts_target_len)
                loss = criterion(y_pred, y_test)
                test_loss += loss.item()
        return test_loss / len(test_loader)


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------
def visualize_predictions(
    path_best_model: str,
    test_loader,
    ts_target_len: int,
    num_samples: int = 5,
    main_title: str | None = None,
    MSE: float | None = None,
    full_length_ts: list | None = None,
    entropy: str | None = None,
    save_path: str | None = None,
):
    """Carga el mejor modelo, predice y visualiza la comparación entre la secuencia de entrada,
    la secuencia objetivo y la predicción.
    """
    # -------------------------------------------------------------------
    # NOTE:
    #   Starting from PyTorch 2.6 the default value of the `weights_only`
    #   parameter in `torch.load` changed from ``False`` to ``True``.
    #   When a full model object (and not just the state-dict) has been
    #   serialized with ``torch.save(model, path)``, trying to load it
    #   with the new default raises an ``UnpicklingError`` because custom
    #   classes (such as ``TransformerForecaster``) are filtered out of
    #   the safe unpickler's allow-list.
    #
    #   The quickest and safest workaround (given that we created the
    #   checkpoint ourselves and therefore trust its contents) is to set
    #   ``weights_only=False`` when loading.  Additionally, we specify
    #   ``map_location=device`` to ensure tensors are restored on the
    #   correct device (CPU/CUDA) irrespective of where they were saved.
    # -------------------------------------------------------------------
    logger.info(f"Loading best model from {path_best_model}")
    model = torch.load(path_best_model, map_location=device, weights_only=False)
    model.eval()
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.to(device), y_test.to(device)

    with torch.no_grad():
        predicted = model.predict(x_test, ts_target_len)

    fig, ax = plt.subplots(
        nrows=num_samples + 1, ncols=1, figsize=(11, 4 * (num_samples + 1) / 2)
    )
    # Si no se provee una serie completa, se genera una aleatoria para la primera fila
    if not full_length_ts:
        full_length_ts_length = random.randint(50, 200)  # Longitud entre 50 y 200
        full_length_ts = np.random.randn(full_length_ts_length).tolist()

    # Graficar la serie completa
    ax[0].plot(
        range(len(full_length_ts)),
        full_length_ts,
        color="purple",
        linewidth=1,
        label="Train/Test Dataset",
    )
    ax[0].legend()

    # Seleccionar índices aleatorios para las muestras de test
    total_samples = x_test.size(0)
    unique_indices = random.sample(
        range(total_samples), min(num_samples, total_samples)
    )

    for i, col in enumerate(ax[1:], start=1):
        r = unique_indices[i - 1]
        in_seq = x_test[r, :, 0].cpu().numpy()
        target_seq = y_test[r, :, 0].cpu().numpy()
        pred_seq = predicted[r, :, 0].cpu().numpy()
        x_axis = range(len(in_seq) + len(target_seq))
        col.set_title(f"Test Sample: {r}", loc="left")
        col.plot(x_axis[: len(in_seq)], in_seq, color="blue", label="Input")
        col.plot(x_axis[len(in_seq) :], target_seq, color="green", label="Target")
        col.plot(
            x_axis[len(in_seq) :],
            pred_seq,
            color="red",
            linestyle="--",
            label="Predicted",
        )
        col.axvline(x=len(in_seq), color="k", linestyle="--")
        col.legend()

    if main_title:
        fig.suptitle(
            f"{main_title} - Avg Test Loss: {MSE:.4f} - Entropy: {entropy}", fontsize=11
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# OPTUNA OBJECTIVE FUNCTION
# ---------------------------------------------------------------------------
def objective(trial, series_index: int, series: list):
    """Función objetivo para la optimización de hiperparámetros con Optuna."""
    # Sugerencia de hiperparámetros
    largo_dataset = trial.suggest_int("largo_dataset", 8, 48, step=8)
    hidden_size = trial.suggest_int("hidden_size", 8, 64, step=8)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    ts_history_len = trial.suggest_int("ts_history_len", 30, 60, step=30)
    ts_target_len = trial.suggest_int("ts_target_len", 15, 30, step=15)
    batch_size = trial.suggest_int("batch_size", 16, 32, step=8)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, step=0.01)
    test_set_fraction = 0.2

    # Preparación y normalización de la serie de tiempo
    data_prep_start = time.time()
    ts = series[series_index].iloc[:, 1].tolist()
    ts = ts[-largo_dataset * 60 :]
    data = np.array(ts).reshape(-1, 1)
    standard_scaler = StandardScaler()
    ts = standard_scaler.fit_transform(data).reshape(-1).tolist()
    log_elapsed_time(data_prep_start, "Data preparation in objective")

    test_ds_len = int(len(ts) * test_set_fraction)
    # Epochs para optimizar hiperparámetros
    epochs = 5

    # Procesamiento de datos mediante sliding window y creación de DataLoaders
    data_process_start = time.time()
    X, Y = sliding_window(ts, ts_history_len, ts_target_len)
    train_loader, test_loader = create_dataloaders(
        X, Y, batch_size, test_ds_len, device
    )
    log_elapsed_time(data_process_start, "Data processing in objective")

    # Inicialización del modelo
    model_init_start = time.time()
    model = TransformerForecaster(
        hidden_size=hidden_size, input_size=1, output_size=1, num_layers=num_layers
    ).to(device)
    log_elapsed_time(model_init_start, "Model initialization in objective")

    # Entrenamiento del modelo
    training_start = time.time()
    model.train()
    train_losses, test_losses, best_model_path, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=learning_rate,
        save_path=None,
        trial=trial,
    )
    log_elapsed_time(training_start, "Model training in objective")
    return test_losses[-1]


# ---------------------------------------------------------------------------
# MAIN: Función principal
# ---------------------------------------------------------------------------
def main():
    # Generación de la serie sintética
    series = []
    sim = generate_ts(7 * 24 * 60, add_trend=True, add_noise=True).reset_index(
        drop=False, names=["time"]
    )
    series.append(sim)
    series_index = -1

    # Configuración de Optuna (almacenamiento en memoria o SQLite)
    storage = (
        optuna.storages.InMemoryStorage()
    )  # Cambia a SQLite si se requiere persistencia
    study_name = series[series_index].columns[1]
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        logger.info(
            f"Study '{study_name}' was not previously saved. Creating a new study."
        )

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        sampler=TPESampler(),
    )
    study.optimize(
        lambda trial: objective(trial, series_index=series_index, series=series),
        n_trials=50,
        n_jobs=1,
    )
    logger.info(
        f"----------------------------------------------Best hyperparameters: {study.best_params}"
    )

    best_params = study.best_params

    # Preparación de datos para el entrenamiento final
    data_prep_start = time.time()
    ts = series[series_index].iloc[:, 1].tolist()
    ts = ts[-best_params["largo_dataset"] * 60 :]
    data = np.array(ts).reshape(-1, 1)
    standard_scaler = StandardScaler()
    ts = standard_scaler.fit_transform(data).reshape(-1).tolist()
    log_elapsed_time(data_prep_start, "Data preparation in main")

    test_ds_len = int(len(ts) * 0.2)
    epochs = 200
    ts_target_len = best_params["ts_target_len"]

    # Procesamiento de datos final y creación de DataLoaders
    data_process_start = time.time()
    X, Y = sliding_window(ts, best_params["ts_history_len"], ts_target_len)
    train_loader, test_loader = create_dataloaders(
        X, Y, best_params["batch_size"], test_ds_len, device
    )
    log_elapsed_time(data_process_start, "Data processing in main")

    # Inicialización y entrenamiento del modelo final
    model_init_start = time.time()
    logger.info(f"Current device: {device}")
    model = TransformerForecaster(
        hidden_size=best_params["hidden_size"],
        input_size=1,
        output_size=1,
        num_layers=best_params["num_layers"],
    ).to(device)
    log_elapsed_time(model_init_start, "Model initialization in main")

    training_start = time.time()
    model.train()
    train_losses, test_losses, best_model_path, best_test_loss = model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=best_params["learning_rate"],
        ts_target_len=ts_target_len,
        save_path=f"agentic_workflow/agents/agent_for_forecast/forecast_model/forecast/torch_models/best_model_{study_name}",
        trial=None,
    )
    log_elapsed_time(training_start, "Model training in main")

    # Visualización de las predicciones
    visualize_predictions(
        best_model_path,
        test_loader,
        ts_target_len,
        main_title=study_name,
        MSE=test_losses[-1],
        entropy=(
            round(ent.shannon_entropy(ts), 3),
            round(ent.permutation_entropy(ts), 3),
        ),
        save_path=f"agentic_workflow/agents/agent_for_forecast/forecast_model/forecast/figures/{study_name}.png",
        num_samples=5,
        full_length_ts=ts,
    )


if __name__ == "__main__":
    main()
