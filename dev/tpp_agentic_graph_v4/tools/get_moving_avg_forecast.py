# %%
"""Moving-average forecast generation and office overview helper.

This module exposes a LangChain tool `get_forecast_moving_avg` that returns a
plain-text summary with two parts:
    1. A high-level overview of the office buckets for the requested tenant /
       legacy ID / date, obtained through a small MongoDB aggregation `pipeline`.
    2. A day-by-day, hour-level forecast for the given date range produced by a
       moving-average model (-30 days / +30 days with scaling).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from pymongo import MongoClient

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


from tpp_agentic_graph_v4.tools.get_simulation import (  # noqa: E402
    run_simulations_for_all_dates,
)

# Set working directory to file location

# ---------------------------------------------------------------------------
# MongoDB & collection helpers
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "agentic-memory"
COLLECTION_NAME = "OfficeBuckets"
MONGO_TIMEZONE = "UTC"

DEFAULT_START_HOUR = 9  # 09:00-10:00 by default

# Mapping from office names to legacy IDs


OFFICE_NAME_TO_LEGACY_ID: dict[str, int] = {
    doc["_id"]: doc["legacyId"]
    for doc in MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME].aggregate(
        [
            {"$match": {"_id.tenant": "BChile"}},
            {
                "$group": {
                    "_id": "$office",  # office name (e.g. "160 - Ñuñoa")
                    "legacyId": {"$first": "$_id.legacyId"},
                }
            },
        ]
    )
}


def _get_collection():
    """Return a `pymongo.Collection` instance or *None* on failure."""

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ismaster")  # quick connection test
        return client[DB_NAME][COLLECTION_NAME]
    except Exception as exc:  # pragma: no cover – just best-effort logging
        print(f"MongoDB connection error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Basic aggregation pipeline used for the office overview section
# ---------------------------------------------------------------------------

pipeline: list[dict] = [
    {"$project": {"serviceQueue": 1}},
    {"$unwind": {"path": "$serviceQueue", "preserveNullAndEmptyArrays": False}},
    {"$replaceRoot": {"newRoot": "$serviceQueue"}},
    {
        "$group": {
            "_id": "$deskId",
            "executives": {"$addToSet": "$executive"},
            "series": {"$addToSet": "$serie"},
            "firstEvent": {"$min": "$eventDate"},
            "lastEvent": {"$max": "$eventDate"},
            "totalServed": {"$sum": 1},
        }
    },
]

# ---------------------------------------------------------------------------
# Moving-average forecast implementation (port from media_movil.py)
# ---------------------------------------------------------------------------


def _find_date_with_records(
    start_date: datetime.date, collection, tenant: str, legacy_id: int
):
    """Search forward for the first date that contains records (safety-capped at 365 days)."""

    current = start_date
    for _ in range(365):
        date_str = current.strftime("%Y-%m-%d")
        if collection.find_one(
            {"_id.tenant": tenant, "_id.legacyId": legacy_id, "_id.date": date_str}
        ):
            return date_str
        current += timedelta(days=1)
    return None


def _get_hourly_events(
    date_str: str, collection, tenant: str, legacy_id: int
) -> pd.DataFrame:
    """Return a DataFrame with [`serie`, `serieId`, `hour`, `count`] for the specified day."""

    agg = [
        {
            "$match": {
                "_id.tenant": tenant,
                "_id.legacyId": legacy_id,
                "_id.date": date_str,
            }
        },
        {"$project": {"serviceQueue": 1, "_id": 0}},
        {"$unwind": "$serviceQueue"},
        {"$replaceRoot": {"newRoot": "$serviceQueue"}},
        {"$match": {"callDate": {"$exists": True, "$ne": None}}},
        {
            "$project": {
                "serie": 1,
                "serieId": 1,
                "hour": {"$hour": {"date": "$callDate", "timezone": MONGO_TIMEZONE}},
            }
        },
        {
            "$group": {
                "_id": {"serie": "$serie", "serieId": "$serieId", "hour": "$hour"},
                "count": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "serie": "$_id.serie",
                "serieId": "$_id.serieId",
                "hour": "$_id.hour",
                "count": "$count",
            }
        },
        {"$sort": {"serie": 1, "hour": 1}},
    ]

    df = pd.DataFrame(list(collection.aggregate(agg)))
    return (
        df
        if not df.empty
        else pd.DataFrame(columns=["serie", "serieId", "hour", "count"])
    )


def _generate_forecast(
    start_date_str: str,
    num_days: int,
    tenant: str,
    legacy_id: int,
    today_str: str,
) -> pd.DataFrame:
    """Return a DataFrame with [`forecast_date`, `serie`, `serieId`, `hour`, `forecast_count`]."""

    collection = _get_collection()
    if collection is None:
        return pd.DataFrame()

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    forecasts: list[pd.DataFrame] = []

    def _worker(offset: int):
        forecast_day = start_date + timedelta(days=offset)
        local_col = _get_collection()
        if local_col is None:
            return None

        # day_0 is the same calendar day a year before *with data*.
        day_0_start = forecast_day.replace(year=forecast_day.year - 1)
        day_0_str = _find_date_with_records(day_0_start, local_col, tenant, legacy_id)
        if not day_0_str:
            return None

        day_0_date = datetime.strptime(day_0_str, "%Y-%m-%d").date()
        d_plus_30 = _find_date_with_records(
            day_0_date + timedelta(days=30), local_col, tenant, legacy_id
        )
        d_minus_30 = _find_date_with_records(
            day_0_date - timedelta(days=30), local_col, tenant, legacy_id
        )
        if not (d_plus_30 and d_minus_30):
            return None

        df_plus = _get_hourly_events(d_plus_30, local_col, tenant, legacy_id)
        df_minus = _get_hourly_events(d_minus_30, local_col, tenant, legacy_id)

        # Determine earliest operating hour
        start_hour = DEFAULT_START_HOUR
        hours = []
        if not df_plus.empty:
            hours.append(int(df_plus["hour"].min()))
        if not df_minus.empty:
            hours.append(int(df_minus["hour"].min()))
        if hours:
            start_hour = min(hours + [start_hour])

        df_plus = df_plus[df_plus["hour"] >= start_hour]
        df_minus = df_minus[df_minus["hour"] >= start_hour]

        merge = pd.merge(
            df_plus,
            df_minus,
            on=["serie", "serieId", "hour"],
            how="outer",
            suffixes=("_plus", "_minus"),
        ).fillna(0)
        merge["forecast_count"] = (merge["count_plus"] + merge["count_minus"]) / 2
        merge["forecast_date"] = forecast_day
        return merge[["forecast_date", "serie", "serieId", "hour", "forecast_count"]]

    with ThreadPoolExecutor(max_workers=min(8, num_days)) as ex:
        futs = {ex.submit(_worker, i): i for i in range(num_days)}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and not df.empty:
                forecasts.append(df)

    if not forecasts:
        return pd.DataFrame()

    df_forecast = pd.concat(forecasts, ignore_index=True)

    # Scaling factor (today vs. first forecast day)
    df_today = _get_hourly_events(today_str, collection, tenant, legacy_id)
    total_today = df_today["count"].sum()
    total_day1 = df_forecast[df_forecast["forecast_date"] == start_date][
        "forecast_count"
    ].sum()
    scale = (total_today / total_day1) if total_day1 else 1.0

    df_forecast["scaled_forecast"] = df_forecast["forecast_count"] * scale
    df_forecast["scaled_forecast_ceiling"] = np.ceil(
        df_forecast["scaled_forecast"]
    ).astype(int)

    return df_forecast


def prepare_forecast_df_display(
    start_date: str,
    num_days: int,
    tenant: str,
    legacy_id: int,
    today_date_str: str,
) -> pd.DataFrame:
    """Public wrapper that makes the forecast DataFrame display-ready (rounded & ordered)."""

    df = _generate_forecast(start_date, num_days, tenant, legacy_id, today_date_str)
    if df.empty:
        return df

    df_disp = df.copy()
    df_disp["scaled_forecast"] = df_disp["scaled_forecast"].round(2)

    # Build tramo (hour slot) and possibly shift so it starts at 09:00
    df_disp["tramo"] = df_disp["hour"].apply(
        lambda h: f"{h:02d}:00-{((h + 1) % 24):02d}:00"
    )

    min_hour = int(df_disp["hour"].min()) if not df_disp.empty else DEFAULT_START_HOUR
    shift = (9 - min_hour) % 24
    if shift:
        df_disp["hour"] = (df_disp["hour"] + shift) % 24
        df_disp["tramo"] = df_disp["hour"].apply(
            lambda h: f"{h:02d}:00-{((h + 1) % 24):02d}:00"
        )

    df_disp = df_disp.sort_values(["forecast_date", "hour", "serie"]).reset_index(
        drop=True
    )
    return df_disp


# ---------------------------------------------------------------------------
# handler & tool wrapper
# ---------------------------------------------------------------------------


def handler(
    office_name: str = "160 - Ñuñoa",
    today_date_str: str = "2025-05-08",
    start_date: str = "2025-06-01",
    num_days: int = 30,
) -> str:
    """Return office overview + forecast as a plain-text block."""

    # Tenant está fijo para esta herramienta
    tenant = "BChile"

    # Convert office name to legacy ID
    legacy_id = OFFICE_NAME_TO_LEGACY_ID.get(office_name)
    if legacy_id is None:
        return f"Error: oficina desconocida '{office_name}'."

    collection = _get_collection()
    if collection is None:
        return "Error: No se pudo conectar a MongoDB."

    # overview_cursor = collection.aggregate(
    #     [
    #         {
    #             "$match": {
    #                 "_id.tenant": tenant,
    #                 "_id.legacyId": legacy_id,
    #                 "_id.date": today_date_str,
    #             }
    #         }
    #     ]
    #     + pipeline
    # )
    # # overview = list(overview_cursor)

    # Forecast (moving-average)
    df_forecast = prepare_forecast_df_display(
        start_date=start_date,
        num_days=num_days,
        tenant=tenant,
        legacy_id=legacy_id,
        today_date_str=today_date_str,
    )

    simulation_report: str = "Sin datos para la simulación."

    if df_forecast.empty:
        forecast_txt = "Sin datos para el pronóstico."
    else:
        # Keep only the relevant columns for the final display
        df_final = df_forecast.drop(
            columns=["hour", "forecast_count", "scaled_forecast"], errors="ignore"
        ).rename(columns={"scaled_forecast_ceiling": "calls"})

        # Optional: order columns for readability if they exist
        col_order = [
            c
            for c in ["forecast_date", "serieId", "serie", "tramo", "calls"]
            if c in df_final.columns
        ]
        df_final = df_final[col_order]

        from pathlib import Path

        script_dir = Path(__file__).parent
        script_dir.joinpath("temp").mkdir(exist_ok=True)
        df_final.to_parquet(script_dir / "temp" / "df_final.parquet")

        # Obtener reportes de simulación para **todas** las fechas
        _, simulation_report = run_simulations_for_all_dates(
            office_name, today_date_str, script_dir / "temp" / "df_final.parquet"
        )

        forecast_txt = df_final.to_string(index=False)

    return (
        f"Pronóstico para los próximos {num_days} días (a partir de {start_date}):\n"
        + forecast_txt
        + "\n"
        + "Resultados de la simulación a partir del forecast:\n"
        + "\n"
        + simulation_report
    )


@tool(
    description=(
        "Obtiene un pronóstico de tráfico por hora basado en media móvil de ±30 días "
        "y devuelve también un resumen agregado de la oficina. "
        "También ejecuta una simulación a partir del forecast y devuelve los resultados."
        "La simulación se hace en base a la configuración de la oficina registrada el día de hoy."
        "Argumentos esperados: office_name (str), today_date_str (YYYY-MM-DD), "
        "start_date (YYYY-MM-DD), num_days (int)."
    ),
)
def get_forecast_moving_avg(
    office_name: str,
    today_date_str: str,
    start_date: str,
    num_days: int,
) -> str:
    """LangChain-tool wrapper that just forwards the call to `handler`."""

    return handler(office_name, today_date_str, start_date, num_days)


# ---------------------------------------------------------------------------
# CLI / quick local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from time import perf_counter

    t0 = perf_counter()
    result = get_forecast_moving_avg.invoke(
        {
            "office_name": "160 - Ñuñoa",
            # "office_name": "159 - Providencia",
            "today_date_str": "2025-05-08",
            "start_date": "2025-06-01",
            "num_days": 1,
        }
    )
    elapsed_min = (perf_counter() - t0) / 60.0
    print(result)

    print(
        f"Tiempo de ejecución de get_forecast_moving_avg.invoke: {elapsed_min:.2f} minutos"
    )

# %%
