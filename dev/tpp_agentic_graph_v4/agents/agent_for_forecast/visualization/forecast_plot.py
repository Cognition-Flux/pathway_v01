"""Visualization for time series forecasting: builds and saves interactive Plotly figures for forecast validation."""

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


def build_plotly_forecast_figure(
    training_data: np.ndarray,
    validation_data: np.ndarray,
    forecasted_values: np.ndarray,
    history_len: int,
    target_len: int,
    original_data: np.ndarray,
) -> go.Figure:
    """Build a corrected Plotly figure for forecast validation.

    Parameters
    ----------
    training_data: np.ndarray
        Historical data used for training/context (history_len points).
    validation_data: np.ndarray
        Real data for validation (target_len points).
    forecasted_values: np.ndarray
        Model predictions (target_len points).
    history_len, target_len:
        Sizes of training window and prediction horizon.
    original_data: np.ndarray
        The original data used for the forecast.

    Returns:
    -------
    go.Figure
        The corrected forecast visualization.
    """
    # Comprehensive data validation
    print("PLOT VALIDATION:")
    print(f"Original data length: {len(original_data)}")
    print(f"Training data length: {len(training_data)} (expected: {history_len})")
    print(f"Validation data length: {len(validation_data)} (expected: {target_len})")
    print(f"Predictions length: {len(forecasted_values)} (expected: {target_len})")

    # Verify data consistency
    assert len(training_data) == history_len, (
        f"Training data length mismatch: {len(training_data)} != {history_len}"
    )
    assert len(validation_data) == target_len, (
        f"Validation data length mismatch: {len(validation_data)} != {target_len}"
    )
    assert len(forecasted_values) == target_len, (
        f"Predictions length mismatch: {len(forecasted_values)} != {target_len}"
    )

    # Verify that training + validation = original data segments
    expected_training = original_data[:history_len]
    expected_validation = original_data[history_len : history_len + target_len]

    np.testing.assert_array_almost_equal(
        training_data,
        expected_training,
        decimal=6,
        err_msg="Training data does not match expected segment",
    )
    np.testing.assert_array_almost_equal(
        validation_data,
        expected_validation,
        decimal=6,
        err_msg="Validation data does not match expected segment",
    )

    print("✅ Data validation passed - all segments are correctly aligned")

    # Compute error metrics comparing predictions vs validation data
    mse = float(np.mean((forecasted_values - validation_data) ** 2))
    mae = float(np.mean(np.abs(forecasted_values - validation_data)))
    rmse = float(np.sqrt(mse))
    mape = float(
        np.mean(np.abs((validation_data - forecasted_values) / validation_data)) * 100
    )

    # Enhanced metrics text with more details
    metrics_text = f"MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%"
    data_info = f"Training: {history_len} pts | Prediction: {target_len} pts | Total: {len(original_data)} pts"

    # Create figure
    fig = go.Figure()

    # Add training data (Input – past window) with hover info
    fig.add_scatter(
        x=np.arange(history_len),
        y=training_data,
        mode="lines+markers",
        line={"color": "#70B7FF", "width": 2},
        marker={"size": 4, "color": "#70B7FF"},
        name="Input – past window",
        hovertemplate="<b>Training Data</b><br>Index: %{x}<br>Value: %{y:.4f}<extra></extra>",
    )

    # Add validation data (Ground truth) with continuity from training
    # Include the last training point to create a continuous line
    validation_x = np.arange(
        history_len - 1, history_len + target_len
    )  # [14, 15, 16, 17, 18, 19]
    validation_y = np.concatenate(
        [training_data[-1:], validation_data]
    )  # [last_training, validation_data]

    fig.add_scatter(
        x=validation_x,
        y=validation_y,
        mode="lines+markers",
        line={"color": "purple", "width": 2},
        marker={"size": 6, "color": "purple"},
        name="Ground truth",
        hovertemplate="<b>Ground Truth</b><br>Index: %{x}<br>Value: %{y:.4f}<extra></extra>",
    )

    # Add forecast (Model predictions) with continuity from training
    # Include the last training point to create a continuous line
    forecast_x = np.arange(
        history_len - 1, history_len + target_len
    )  # [14, 15, 16, 17, 18, 19]
    forecast_y = np.concatenate(
        [training_data[-1:], forecasted_values]
    )  # [last_training, predictions]

    fig.add_scatter(
        x=forecast_x,
        y=forecast_y,
        mode="lines+markers",
        line={"color": "tomato", "width": 3},
        marker={"size": 6, "color": "tomato", "symbol": "diamond"},
        name="Forecast",
        hovertemplate="<b>Prediction</b><br>Index: %{x}<br>Value: %{y:.4f}<br>Error: %{customdata:.4f}<extra></extra>",
        customdata=np.concatenate(
            [np.array([0.0]), np.abs(forecasted_values - validation_data)]
        ),  # No error for the connecting point
    )

    # Apply corporate dark theme styling with enhanced layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#DDDDDD"},
        title={
            "text": f"Time Series Forecast Validation<br><sup>{metrics_text}</sup><br><sub>{data_info}</sub>",
            "x": 0.5,
            "font_size": 16,
        },
        xaxis_title="Time Index",
        yaxis_title="Value",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0.7)",
            "bordercolor": "#666666",
            "borderwidth": 1,
        },
        margin={
            "l": 60,
            "r": 120,
            "t": 100,
            "b": 60,
            "pad": 5,
        },  # Adjusted right margin back to normal
        autosize=True,
        hovermode="x unified",
        showlegend=True,
    )

    # Enhanced axis styling
    axis_style = {
        "showgrid": True,
        "gridcolor": "#444444",
        "gridwidth": 0.5,
        "showline": True,
        "linecolor": "#666666",
        "linewidth": 1,
        "zeroline": True,
        "zerolinecolor": "#555555",
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Save the figure as interactive HTML file and PNG to the specified absolute path
    try:
        # Use the absolute path /home/alejandro/Pictures/pathway_plots/

        # Define the absolute directory path
        plots_abs_dir = Path("/home/alejandro/Pictures/pathway_plots")
        plots_abs_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure parents=True for creating intermediate dirs

        # Save the figure as interactive HTML (overwrites existing file)
        output_html_path = plots_abs_dir / "forecast_plot.html"
        fig.write_html(
            str(output_html_path),
            include_plotlyjs="cdn",  # Use CDN to keep file size smaller
            config={"displayModeBar": True, "responsive": True},
            auto_open=False,  # Don't automatically open in browser
        )
        logger.info(f"Forecast plot HTML saved successfully to: {output_html_path}")

        # Also save as PNG image (overwrites existing file)
        output_png_path = plots_abs_dir / "forecast_plot.png"
        fig.write_image(
            str(output_png_path),
            width=1200,
            height=800,
            scale=2,  # Higher resolution
            format="png",
        )
        logger.info(f"Forecast plot PNG saved successfully to: {output_png_path}")

    except Exception as save_exc:
        # Log the error but don't fail the entire function
        logger.warning(f"Failed to save forecast plot files: {save_exc}")

    print("✅ Plot generated successfully with enhanced validation and metrics")
    return fig
