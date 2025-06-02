#!/usr/bin/env python
"""Simple inference script for the time-series forecaster.

This module generates a synthetic time-series, loads a pre-trained and fine-tuned
`TransformerForecaster` model and produces a forecast for the next *n* points.

The code is deliberately **simple** and **modular** so it can serve as a reference
for future, more sophisticated inference pipelines.

Usage (from the project root):

```bash
uv run python agentic_workflow/agents/agent_for_forecast/forecast_model/inference.py
```

The script will:
1. Build a synthetic time-series of configurable length.
2. Perform the same standard scaling used during training.
3. Feed the last `TS_HISTORY_LEN` points to the model and predict the next
   `TS_TARGET_LEN` steps.
4. Print the numerical forecast and display a quick matplotlib figure that
   contrasts the input window with the forecast horizon.
"""

# %%
# %%
# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------------
# Make classes available in global namespace for pickle deserialization
# This is required when loading models that were saved as full objects
# This needs to be done at module level so it works regardless of execution context
# ---------------------------------------------------------------------------
import builtins
import logging

# ---------------------------------------------------------------------------
# Make classes available in global namespace for pickle deserialization
# This is required when loading models that were saved as full objects
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Internal imports – we reuse the utilities already implemented in
# `forecaster.py` so we do **not** duplicate logic.
# ---------------------------------------------------------------------------
from agentic_workflow.agents.agent_for_forecast.forecast_model.forecaster import (  # type: ignore
    PositionalEncoding,
    TransformerForecaster,
    TransformerModel,
    generate_ts,
)


# ---------------------------------------------------------------------------
# Make classes available in global namespace for pickle deserialization
# This is required when loading models that were saved as full objects
# ---------------------------------------------------------------------------

# Make classes available in the current module
current_module = sys.modules[__name__]
current_module.TransformerForecaster = TransformerForecaster
current_module.TransformerModel = TransformerModel
current_module.PositionalEncoding = PositionalEncoding

# Also make them available globally so pickle can find them from any context
# This is essential when this module is imported by other scripts like workflow.py
globals()["TransformerForecaster"] = TransformerForecaster
globals()["TransformerModel"] = TransformerModel
globals()["PositionalEncoding"] = PositionalEncoding

# Ensure they're available in builtins as a last resort
builtins.TransformerForecaster = TransformerForecaster
builtins.TransformerModel = TransformerModel
builtins.PositionalEncoding = PositionalEncoding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TS_HISTORY_LEN: int = 80  # Number of past points fed to the model
TS_TARGET_LEN: int = 30  # How many points we want to forecast

DEFAULT_MODEL_PATH = (
    Path(__file__).parent
    / "forecast"
    / "torch_models"
    / "best_model_simulation_best_test_loss"
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_model(model_path: Path, device: torch.device) -> TransformerForecaster:
    """Load a fine-tuned `TransformerForecaster` from disk."""
    logger.info("Loading model from %s", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Ensure classes are available in __main__ module for pickle deserialization
    # This is necessary when this function is called from workflow.py or other scripts
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_module.TransformerForecaster = TransformerForecaster
        main_module.TransformerModel = TransformerModel
        main_module.PositionalEncoding = PositionalEncoding

    # See note in `forecaster.visualize_predictions` regarding `weights_only`.
    model: TransformerForecaster = torch.load(
        model_path,
        map_location=device,
        weights_only=False,  # type: ignore[arg-type]
    )
    model.eval()
    return model


def prepare_input(
    series: np.ndarray,
    history_len: int,
) -> tuple[torch.Tensor, StandardScaler]:
    """Standardise the full series and extract the last *history_len* points.

    We *fit* the scaler on the **entire** synthetic series so the distribution is
    comparable to what the model saw during training (which applied the same
    transformation).
    """
    scaler = StandardScaler()
    scaled_series = (
        scaler.fit_transform(series.reshape(-1, 1)).astype(np.float32).ravel()
    )

    input_window = scaled_series[-history_len:]
    # Shape expected by the model: (batch, seq_len, features)
    input_tensor = torch.from_numpy(input_window).unsqueeze(0).unsqueeze(2)
    return input_tensor, scaler


def forecast(
    model: TransformerForecaster,
    input_tensor: torch.Tensor,
    target_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the forward pass and return the raw (scaled) forecast."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        predictions = model.predict(input_tensor, target_len=target_len)
    # Remove batch & feature dimensions → (target_len,)
    return predictions.squeeze(0).squeeze(1).cpu()


def inverse_transform(predictions: torch.Tensor, scaler: StandardScaler) -> np.ndarray:
    """Convert the *scaled* predictions back to the original value range."""
    preds_2d = predictions.unsqueeze(1).numpy()
    return scaler.inverse_transform(preds_2d).ravel()


def plot_results(
    original_series: np.ndarray,
    forecasted_values: np.ndarray,
    history_len: int,
    target_len: int,
) -> None:
    """Quick matplotlib helper that shows the input window and the forecast."""
    past = original_series[-(history_len + target_len) : -target_len]
    future_target = original_series[-target_len:]

    x_axis = np.arange(history_len + target_len)
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_axis[:history_len],
        past,
        label="Input – past window",
        color="#70B7FF",  # light sky blue
    )
    plt.plot(
        x_axis[history_len:],
        future_target,
        label="Ground truth",
        color="purple",  # purple hue
        linewidth=1.2,
    )
    plt.plot(
        x_axis[history_len:],
        forecasted_values,
        label="Forecast",
        color="tomato",  # light tomato
        linestyle="-",  # solid line
        linewidth=2.5,
    )
    plt.axvline(x=history_len, color="black", linestyle="--")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.05)
    plt.show()


def build_plotly_forecast_figure(
    original_series: np.ndarray,
    forecasted_values: np.ndarray,
    history_len: int,
    target_len: int,
) -> go.Figure:
    """Build a Plotly figure visualising past window, ground truth and forecast.

    Parameters
    ----------
    original_series:
        Full (unscaled) series used for inference.
    forecasted_values:
        Array with the predicted future points (same scale as *original_series*).
    history_len, target_len:
        Sizes of the past window and forecast horizon.

    Returns:
    -------
    go.Figure
        The constructed figure so it can be further modified or embedded.
    """
    past = original_series[-(history_len + target_len) : -target_len]
    future_target = original_series[-target_len:]

    # Compute error metrics
    mse = float(np.mean((forecasted_values - future_target) ** 2))
    mae = float(np.mean(np.abs(forecasted_values - future_target)))
    rmse = float(np.sqrt(mse))
    metrics_text = f"MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}"

    fig = px.line(
        x=np.arange(history_len + target_len),
        y=np.concatenate([past, future_target]),
        labels={"x": "Time", "y": "Value"},
        title="Time Series Forecast",
    )
    fig.add_scatter(
        x=np.arange(history_len),
        y=past,
        mode="lines",
        line={"color": "#70B7FF"},
        name="Input – past window",
    )
    fig.add_scatter(
        x=np.arange(history_len, history_len + target_len),
        y=future_target,
        mode="lines",
        line={"color": "purple", "width": 1.2},
        name="Ground truth",
    )
    fig.add_scatter(
        x=np.arange(history_len, history_len + target_len),
        y=forecasted_values,
        mode="lines",
        line={"color": "tomato", "width": 3},
        name="Forecast",
    )

    # Vertical dashed line marking the forecast start
    fig.add_vline(
        x=history_len,
        line_dash="dash",
        line_color="#AAAAAA",
        line_width=2,
        yref="paper",
        y0=0.1,
        y1=0.85,
        annotation_text="Forecast start",
        annotation_position="top",
        annotation_font_color="#DDDDDD",
        annotation_yshift=-30,
    )

    # Apply corporate dark theme styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#DDDDDD"},
        title={"text": f"Time Series Forecast<br><sup>{metrics_text}</sup>", "x": 0.5},
        xaxis_title="Time",
        yaxis_title="Value",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0.5)",  # Semi-transparent background
            "bordercolor": "#666666",
            "borderwidth": 1,
        },
        margin={
            "l": 60,
            "r": 20,
            "t": 80,
            "b": 60,
            "pad": 5,
        },  # More responsive margins
        autosize=True,  # Enable auto-sizing to container
        hovermode="x unified",  # Better hover behavior
        # Remove fixed width and height to allow responsiveness
    )

    # Axis styling – graphite grey lines & grid
    axis_style = {
        "showgrid": True,
        "gridcolor": "#444444",
        "gridwidth": 0.5,
        "showline": True,
        "linecolor": "#666666",
        "linewidth": 1,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Save the plot as PNG image and HTML file to the absolute path
    try:
        # Use the absolute path /home/alejandro/Pictures/pathway_plots/
        plots_abs_dir = Path("/home/alejandro/Pictures/pathway_plots")
        plots_abs_dir.mkdir(parents=True, exist_ok=True)

        # Save the figure as PNG (overwrites existing file)
        output_png_path = plots_abs_dir / "forecast_plot.png"
        fig.write_image(
            str(output_png_path),
            width=1200,
            height=800,
            scale=2,  # Higher resolution
            format="png",
        )
        logger.info(f"Forecast plot PNG saved successfully to: {output_png_path}")

        # Save the figure as interactive HTML (overwrites existing file)
        output_html_path = plots_abs_dir / "forecast_plot.html"
        fig.write_html(
            str(output_html_path),
            include_plotlyjs="cdn",  # Use CDN to keep file size smaller
            config={"displayModeBar": True, "responsive": True},
            auto_open=False,  # Don't automatically open in browser
        )
        logger.info(f"Forecast plot HTML saved successfully to: {output_html_path}")

    except Exception as save_exc:
        # Log the error but don't fail the entire function
        logger.warning(f"Failed to save forecast plot files: {save_exc}")

    return fig


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def forecast_series(
    series: pd.DataFrame | pd.Series | np.ndarray | list[float],
    *,
    history_len: int = TS_HISTORY_LEN,
    target_len: int = TS_TARGET_LEN,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    """Run inference on *series* and return the forecast.

    Parameters
    ----------
    series: pd.DataFrame | pd.Series | np.ndarray | list[float]
        The time-series for which to generate the forecast. If a DataFrame is
        provided, the values are extracted from the **last** column. When a
        Series/array-like object is given, the raw values are used directly.
    history_len: int, default=TS_HISTORY_LEN
        Number of past points from *series* that will be fed into the model.
    target_len: int, default=TS_TARGET_LEN
        Forecast horizon – how many future points to predict.
    model_path: Path, default=DEFAULT_MODEL_PATH
        File path to the fine-tuned :class:`TransformerForecaster` checkpoint.

    Returns:
    -------
    np.ndarray
        Forecast values in the original (un-scaled) range.
    """
    # ------------------------------------------------------------------
    # 1) Normalise *entire* series and extract the most recent `history_len`
    # ------------------------------------------------------------------
    if isinstance(series, pd.DataFrame):
        raw_values = series.iloc[:, -1].values  # type: ignore[index]
    elif isinstance(series, pd.Series):  # type: ignore[misc]
        raw_values = series.values  # type: ignore[attr-defined]
    else:
        raw_values = np.asarray(series, dtype=float)

    if raw_values.ndim != 1:
        raise ValueError("`series` must be one-dimensional after value extraction.")

    input_tensor, scaler = prepare_input(raw_values, history_len)

    # ------------------------------------------------------------------
    # 2) Load model & run prediction
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    scaled_forecast = forecast(model, input_tensor, target_len, device)

    # ------------------------------------------------------------------
    # 3) Bring the forecast back to the original scale
    # ------------------------------------------------------------------
    forecast_values = inverse_transform(scaled_forecast, scaler)

    # ------------------------------------------------------------------
    # 4) Optional plot for visual sanity-check
    # ------------------------------------------------------------------
    numeric_series = raw_values
    build_plotly_forecast_figure(
        numeric_series, forecast_values, history_len, target_len
    )

    return forecast_values


# ---------------------------------------------------------------------------
# Main routine (quick self-test)
# ---------------------------------------------------------------------------
def forecast_series_no_plot(
    series: pd.DataFrame | pd.Series | np.ndarray | list[float],
    *,
    history_len: int = TS_HISTORY_LEN,
    target_len: int = TS_TARGET_LEN,
    model_path=Path(
        "agentic_workflow/agents/agent_for_forecast/forecast_model/forecast/torch_models/best_model_simulation_best_test_loss"
    ),
) -> np.ndarray:
    """Run inference on series and return the forecast without plotting."""
    # Normalize input
    if isinstance(series, pd.DataFrame):
        raw_values = series.iloc[:, -1].values
    elif isinstance(series, pd.Series):
        raw_values = series.values
    else:
        raw_values = np.asarray(series, dtype=float)

    if raw_values.ndim != 1:
        raise ValueError("`series` must be one-dimensional after value extraction.")

    input_tensor, scaler = prepare_input(raw_values, history_len)

    # Load model & run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    scaled_forecast = forecast(model, input_tensor, target_len, device)

    # Bring the forecast back to the original scale
    forecast_values = inverse_transform(scaled_forecast, scaler)

    return forecast_values


def main() -> None:
    """Self-contained sanity-check executed when the module is run directly."""
    # Generate the exact synthetic series requested by the user.
    sim: pd.DataFrame = generate_ts(
        7 * 24 * 60, add_trend=True, add_noise=True
    ).reset_index(drop=False, names=["time"])

    # Run inference using *all default* parameters and the fine-tuned model.
    predictions = forecast_series(sim)

    numeric_series = sim.iloc[:, 1].values  # columna 'simulation'
    fig = build_plotly_forecast_figure(
        numeric_series, predictions, TS_HISTORY_LEN, TS_TARGET_LEN
    )
    fig.show()


if __name__ == "__main__":
    main()

# %%
