"""Simulador de atención de clientes.

Este simulador es una tool que permite simular el comportamiento de un sistema de atención de clientes.

"""

# %%

import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
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


from tpp_agentic_graph_v4.tools.get_office_configurations import (  # noqa: E402  # noqa: E402
    _get_collection,
    get_office_configurations,
)
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    Desk as DeskRequest,
)
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    DeskSerie as DeskSerieRequest,
)
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    NewSimulation as NewSimulationRequest,
)
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    Serie as SerieRequest,
)
from tpp_agentic_graph_v4.workforce.service_logger import logger  # noqa: E402
from tpp_agentic_graph_v4.workforce.simulator import Simulation  # noqa: E402

# ---------------------------------------------------------------------------
# MongoDB config
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "agentic-memory"
COLLECTION_NAME = "OfficeBuckets"
# %%


def _get_collection():
    """Return the OfficeBuckets collection or None on failure."""

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ismaster")
        return client[DB_NAME][COLLECTION_NAME]
    except Exception as exc:  # pragma: no cover
        print(f"MongoDB connection error: {exc}")
        return None


collection = _get_collection()

OFFICE_NAME_TO_LEGACY_ID: dict[str, int] = {}
if collection is not None:
    OFFICE_NAME_TO_LEGACY_ID = {
        doc["_id"]: doc["legacyId"]
        for doc in collection.aggregate(
            [
                {"$match": {"_id.tenant": "BChile"}},
                {
                    "$group": {
                        "_id": "$office",
                        "legacyId": {"$first": "$_id.legacyId"},
                    }
                },
            ]
        )
    }

# -------------------------------------------------------------------------
# Constantes y utilidades globales
# -------------------------------------------------------------------------

# Número de réplicas paralelas por fecha
num_rounds = 5

# Umbrales de evaluación de tiempos de espera (segundos, etiqueta)
THRESHOLDS: list[tuple[int, str]] = [
    (1800, "< 30 min"),
    (1200, "< 20 min"),
    (900, "< 15 min"),
    (600, "< 10 min"),
    (300, "< 5 min"),
]

# Orden auxiliar solo con etiquetas
THRESHOLD_ORDER = [label for _, label in THRESHOLDS]


def _time_from_str(t_str: str) -> time:
    """Convierte "HH:MM" a ``datetime.time``."""

    return datetime.strptime(t_str, "%H:%M").time()


def _mean_std(values: list[float]) -> str:
    """Devuelve cadena "media ± std" con 2 decimales."""

    return f"{np.mean(values):.2f} ± {np.std(values, ddof=0):.2f}%"


def get_series_mapping(legacy_id: int, date_str: str, tenant: str = "BChile") -> dict:
    """Devuelve un dict {serie_name: serieId} filtrando por oficina (legacy_id) y fecha.

    Args:
        legacy_id (int): ID numérico de la oficina en Legacy.
        date_str (str): Fecha en formato YYYY-MM-DD.
        tenant (str, optional): Tenant a usar en la query. Por defecto "BChile".

    Returns:
        dict: Mapeo {nombre_serie: serieId}. Si la colección no está disponible o no hay
              resultados, se devuelve un dict vacío.
    """

    try:
        # Reutilizamos el helper de conexión ya existente en get_office_configurations

        collection = _get_collection()
    except Exception:
        collection = None

    if collection is None:
        logger.warning(
            "No se pudo obtener la colección de MongoDB para series mapping."
        )
        return {}

    pipeline = [
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
        {
            "$group": {
                "_id": "$serie",  # nombre de la serie
                "serieId": {"$first": "$serieId"},
            }
        },
    ]

    try:
        mapping_cursor = collection.aggregate(pipeline)
        mapping = {
            doc["_id"]: str(doc["serieId"])
            for doc in mapping_cursor
            if doc.get("_id") is not None
        }
        return mapping
    except Exception as exc:  # pragma: no cover
        logger.error(f"Error al ejecutar pipeline de series: {exc}")
        return {}


def process_forecast(df_forecast: pd.DataFrame, office_series_map) -> pd.DataFrame:
    # Asegurar que la columna forecast_date sea de tipo datetime para operaciones .dt
    if df_forecast["forecast_date"].dtype != "datetime64[ns]":
        df_forecast["forecast_date"] = pd.to_datetime(df_forecast["forecast_date"])

    # Si no tenemos mapping (p.e. sin conexión a MongoDB) no filtramos para evitar
    # quedarnos sin datos; devolvemos el *forecast* tal cual.
    if not office_series_map:
        logger.warning(
            "office_series_map vacío – se omite filtrado de series y corrección de llamadas."
        )
        return df_forecast.copy()

    # Guardamos métricas originales antes del filtrado
    original_calls_sum = df_forecast["calls"].sum()

    # Aplicar filtrado y calcular porcentaje de filas conservadas
    df_forecast = df_forecast[
        df_forecast["serie"].isin(set(office_series_map.keys()))
    ].copy()

    filtered_calls_sum = df_forecast["calls"].sum()

    if filtered_calls_sum == 0:
        logger.warning(
            "Tras el filtrado por series no queda ninguna llamada. Se devolverá el forecast sin corrección."
        )
        return df_forecast

    extra_calls = int(original_calls_sum - filtered_calls_sum)

    if extra_calls > 0:
        # Factor de escala global para mantener el total de llamadas
        scale_factor = original_calls_sum / filtered_calls_sum

        scaled_float = df_forecast["calls"] * scale_factor
        calls_floor = np.floor(scaled_float).astype(int)

        df_forecast_corrected = df_forecast.copy()
        df_forecast_corrected["calls"] = calls_floor

        # Ajustar la diferencia debida al redondeo
        remainder = int(original_calls_sum - df_forecast_corrected["calls"].sum())
        if remainder > 0:
            fractional_part = scaled_float - calls_floor
            indices_desc = fractional_part.nlargest(remainder).index
            df_forecast_corrected.loc[indices_desc, "calls"] += 1
    else:
        df_forecast_corrected = df_forecast.copy()

    print(
        f"\nTotal de llamadas tras corrección: {df_forecast_corrected['calls'].sum()} (debería coincidir con {original_calls_sum})"
    )

    return df_forecast_corrected


def _collect_metrics(simulation) -> dict:
    """Extrae las métricas requeridas de una instancia de Simulation."""

    events_df = simulation.queue.events

    metrics: dict[str, float | dict] = {}

    total_events = len(events_df)
    if total_events == 0:
        return {}

    metrics["avg_service_time_min"] = events_df["serviceTime"].mean() / 60
    metrics["avg_waiting_time_min"] = events_df["waitingTime"].mean() / 60

    thresholds = THRESHOLDS  # Reutiliza la constante global

    # Porcentaje global bajo cada umbral
    metrics["thresholds"] = {}
    for th_seconds, lbl in thresholds:
        pct = (events_df["waitingTime"] < th_seconds).sum() / total_events * 100
        metrics["thresholds"][lbl] = pct

    # Desglose por serie bajo cada umbral
    metrics["breakdown"] = {}
    simulated_series = events_df["serieId"].unique().tolist()
    for th_seconds, lbl in thresholds:
        per_serie: dict[str, float] = {}
        for serie in simulated_series:
            serie_events = events_df[events_df["serieId"] == serie]
            if len(serie_events) == 0:
                continue
            pct_serie = (
                (serie_events["waitingTime"] < th_seconds).sum()
                / len(serie_events)
                * 100
            )
            per_serie[serie] = pct_serie
        metrics["breakdown"][lbl] = per_serie

    return metrics


def _generate_events(forecast_day: pd.DataFrame) -> pl.DataFrame:
    """Genera un DataFrame de eventos basado en el pronóstico filtrado (forecast_day), con aleatoriedad."""

    events_data_local: list[dict] = []

    for _, row in forecast_day.iterrows():
        start_str, end_str = row["tramo"].split("-")

        start_dt = datetime.combine(
            row["forecast_date"].date(), _time_from_str(start_str)
        )
        end_dt = datetime.combine(row["forecast_date"].date(), _time_from_str(end_str))

        interval_seconds = int((end_dt - start_dt).total_seconds())
        n_calls = int(row["calls"]) if not pd.isna(row["calls"]) else 0

        # Misma lógica de offsets en segundos pero con nueva semilla implícita
        if n_calls <= interval_seconds and n_calls > 0:
            seconds_offsets = sorted(random.sample(range(interval_seconds), n_calls))
        else:
            seconds_offsets = sorted(
                random.randint(0, interval_seconds) for _ in range(n_calls)
            )

        for offset in seconds_offsets:
            events_data_local.append(
                {
                    "eventId": str(uuid4()),
                    "serieId": str(row["serie"]),
                    "serviceTime": random.randint(5 * 60, 20 * 60)
                    + random.randint(2, 6 * 60),
                    "emission": start_dt + timedelta(seconds=offset),
                }
            )

    events_local = (
        pd.DataFrame(events_data_local).sort_values("emission").reset_index(drop=True)
    )
    return pl.from_pandas(events_local, include_index=False)


def _run_single_simulation(
    request_local: NewSimulationRequest, forecast_day: pd.DataFrame
) -> tuple[Simulation, dict] | None:
    """Construye, ejecuta la simulación y devuelve las métricas."""

    try:
        fresh_events = _generate_events(forecast_day)

        sim = Simulation.from_simulation_request(
            request=request_local, events=fresh_events
        )
        sim.initialize()
        for desk in sim.desks:
            try:
                desk.skills_priority.rotate(
                    random.randint(0, len(desk.skills_priority))
                )
            except AttributeError:
                # Para modos que no usan skills_priority
                pass

        sim.run_simulation(debug=False)
        sim.run_overtime(overtime_limit=timedelta(hours=5), debug=False)

        return sim, _collect_metrics(sim)
    except Exception as exc:
        logger.error(f"Simulation round failed: {exc}")
        return None


def simulate_day(
    target_date: date,
    df_forecast: pd.DataFrame,
    office_series_map,
    office_configurations,
) -> tuple[Simulation, str] | None:
    """Ejecuta las simulaciones y reportes para un día dado.

    Args:
        target_date: Fecha objetivo a simular (datetime.date).
        df_forecast: DataFrame completo de pronóstico, que incluye la columna
            ``forecast_date`` en formato datetime.
    """

    # --- Filtrado por fecha específica ---

    series_from_mapping_from_office_configurations: set[str] = set(
        office_series_map.keys()
    )

    # series_from_forecast: set[str] = set(str(s) for s in df_forecast["serie"].unique())

    desks_config_list = office_configurations.get("configurations", [])

    _series = list(series_from_mapping_from_office_configurations)

    _desks = int(len(desks_config_list))

    forecast_day = df_forecast[
        df_forecast["forecast_date"].dt.date == target_date
    ].copy()
    if forecast_day.empty:
        logger.warning(f"No hay datos de forecast para la fecha {target_date}")
        return None

    # Si por alguna razón _series quedó vacío (p.e. mapping generado on-the-fly),
    # recurrimos a las series presentes en el propio forecast para evitar errores.
    if not _series:
        _series = list(df_forecast["serie"].unique())

    # --- Construir el request específico para la fecha ---
    request_local = NewSimulationRequest(
        officeId=uuid4(),
        targetDate=target_date,
        startTime=time(hour=8, minute=0),
        endTime=time(hour=17, minute=0),
        series=[SerieRequest(serieId=s) for s in _series],
        desks=[
            DeskRequest(
                deskId=uuid4(),
                serviceConfiguration="FIFO",
                deskSeries=[
                    DeskSerieRequest(
                        serieId=ds, priority=p + 1, step=random.randint(1, 4)
                    )
                    for p, ds in enumerate(_series)
                ],
            ).model_dump()
            for _ in range(_desks)
        ],
    )

    # --- Paralelizar ejecuciones ---
    with ThreadPoolExecutor(max_workers=num_rounds) as executor:
        futures = [
            executor.submit(_run_single_simulation, request_local, forecast_day)
            for _ in range(num_rounds)
        ]
        results = [f.result() for f in as_completed(futures) if f.result()]

    if not results:
        logger.error("Todas las simulaciones fallaron para la fecha %s", target_date)
        return None

    sims, metrics_runs = zip(*results)

    # --- Agregar resultados ---
    aggregate_numeric: defaultdict[str, list[float]] = defaultdict(list)
    aggregate_breakdown: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for m in metrics_runs:
        aggregate_numeric["avg_service_time_min"].append(m["avg_service_time_min"])
        aggregate_numeric["avg_waiting_time_min"].append(m["avg_waiting_time_min"])

        for lbl, pct in m["thresholds"].items():
            aggregate_numeric[lbl].append(pct)

        for lbl, serie_dict in m["breakdown"].items():
            for serie, pct in serie_dict.items():
                aggregate_breakdown[lbl][serie].append(pct)

    report_lines: list[str] = []
    report_lines.append(
        f"\n================= Fecha simulada {target_date} - ({num_rounds} réplicas)================="
    )

    # Tabla global
    global_rows = [
        [
            "Tiempo de servicio promedio (min)",
            _mean_std(aggregate_numeric["avg_service_time_min"]).replace("%", ""),
        ],
        [
            "Tiempo de espera promedio (min)",
            _mean_std(aggregate_numeric["avg_waiting_time_min"]).replace("%", ""),
        ],
    ] + [
        [f"% eventos con espera {label}", _mean_std(aggregate_numeric[label])]
        for label in THRESHOLD_ORDER
    ]

    report_lines.append("\n| Métrica | Promedio ± Desv. Estándar |")
    report_lines.append("|---|---|")
    for name, value in global_rows:
        report_lines.append(f"| {name} | {value} |")

    # Tabla por serie para cada umbral
    for lbl in THRESHOLD_ORDER:
        report_lines.append(f"\n**Umbral {lbl}**")
        report_lines.append("\n| Serie | % eventos (Prom ± Desv.EE) |")
        report_lines.append("|---|---|")
        serie_means = {
            serie: (np.mean(pcts), np.std(pcts))
            for serie, pcts in aggregate_breakdown[lbl].items()
        }
        for serie, (mean_val, std_val) in sorted(
            serie_means.items(), key=lambda x: x[1][0], reverse=True
        ):
            report_lines.append(f"| {serie} | {mean_val:.2f} ± {std_val:.2f}% |")
    report_lines.append("\n---\n")

    report_str = "\n".join(report_lines)
    print(report_str)

    # Devolver primer simulation (representativa) y el reporte
    return sims[0], report_str


def _build_placeholder_forecast(
    office_series_map: dict[str, str],
    date_str: str,
) -> pd.DataFrame:
    """Crea un *forecast* mínimo viable cuando falta el archivo `.parquet`.

    El objetivo es **no** detener el flujo de simulación en entornos de desarrollo
    donde todavía no existe un pronóstico real. Se genera una distribución
    uniforme de *calls* por serie y por tramo de 30 min entre las 08:00 y 17:00.

    Parameters
    ----------
    office_series_map:
        Mapeo *{serie_name: serieId}* para la oficina / fecha indicada. Sólo se
        utilizan las *keys* (nombres de serie) para construir el *DataFrame*.
    date_str:
        Fecha del pronóstico en formato ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas requeridas por el simulador:
        ``forecast_date``, ``serie``, ``tramo`` y ``calls``.
    """

    forecast_rows: list[dict[str, object]] = []
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Definimos tramos de 30 min entre las 08:00 y las 17:00 (excluido 17:00)
    for serie_name in office_series_map.keys() or ["DEFAULT"]:
        start_hour = 8
        end_hour = 17
        for hour in range(start_hour, end_hour):
            # 08:00-08:30 y 08:30-09:00, ..., 16:30-17:00
            tramo_1 = f"{hour:02d}:00-{hour:02d}:30"
            tramo_2 = f"{hour:02d}:30-{hour + 1:02d}:00"
            for tramo in (tramo_1, tramo_2):
                forecast_rows.append(
                    {
                        "forecast_date": target_date,
                        "serie": serie_name,
                        "tramo": tramo,
                        "calls": 10,  # número fijo de llamadas placeholder
                    }
                )

    df_placeholder = pd.DataFrame(forecast_rows)
    return df_placeholder


def run_simulations_for_all_dates(
    office_user_input: str, date_user_input: str, forecast_path
) -> tuple[Simulation, str]:
    """Ejecuta simulaciones para **todas** las fechas presentes en el pronóstico.

    Por cada día contenido en ``forecast_path`` se corre ``simulate_day`` y se
    construye un reporte en texto plano. Todos los reportes se concatenan (en
    el mismo orden cronológico) y se devuelven junto con una instancia de
    ``Simulation`` representativa (la correspondiente al primer día).

    Notes
    -----
    Este cambio asegura que las herramientas consumidoras –por ejemplo
    ``get_moving_avg_forecast``– muestren los resultados de **todas** las fechas
    solicitadas por el usuario, en vez de sólo el primer día.
    """

    # ------------------------------------------------------------------
    # 1) Obtener el mapping de series de la oficina / fecha
    # ------------------------------------------------------------------
    office_series_map = get_series_mapping(
        legacy_id=OFFICE_NAME_TO_LEGACY_ID[office_user_input],
        date_str=date_user_input,
    )

    # ------------------------------------------------------------------
    # 2) Cargar o construir el pronóstico
    # ------------------------------------------------------------------
    forecast_path = Path(forecast_path)
    if not forecast_path.exists():
        logger.warning(
            "Forecast file %s no encontrado. Generando placeholder de prueba...",
            forecast_path,
        )
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        df_forecast = _build_placeholder_forecast(office_series_map, date_user_input)
        df_forecast.to_parquet(forecast_path)
        logger.info("Placeholder guardado en %s", forecast_path)
    else:
        df_forecast = pd.read_parquet(forecast_path)

    # Si no pudimos obtener el mapping desde MongoDB, lo derivamos del propio forecast
    if not office_series_map:
        derived_map = {str(s): str(s) for s in df_forecast["serie"].unique()}
        office_series_map.update(derived_map)
        logger.info(
            "Se derivó un office_series_map con %d series desde el forecast",
            len(derived_map),
        )

    # ------------------------------------------------------------------
    # 3) Procesar y filtrar el pronóstico
    # ------------------------------------------------------------------
    df_forecast = process_forecast(df_forecast, office_series_map)

    # ------------------------------------------------------------------
    # 4) Continuar con la lógica original
    # ------------------------------------------------------------------
    office_configurations = get_office_configurations.invoke(
        {"office_name": office_user_input, "date": date_user_input}
    )
    _unique_dates = sorted(df_forecast["forecast_date"].dt.date.unique())

    results: list[tuple[Simulation, str]] = []
    for _date in _unique_dates:
        res = simulate_day(_date, df_forecast, office_series_map, office_configurations)
        if res:
            results.append(res)

    if not results:
        raise ValueError("No se encontraron resultados")

    # ------------------------------------------------------------------
    # 5) Concatenar reportes y devolver resultado
    # ------------------------------------------------------------------
    representative_sim, _ = results[0]
    combined_reports = "\n".join([r[1] for r in results])

    combined_reports += (
        f"\nconfiguración de oficina {office_user_input} registrada el {date_user_input} que se usó para la simulación: \n"
        + str(office_configurations)
    )

    return representative_sim, combined_reports


# %%
if __name__ == "__main__":
    forecast_path = package_root / "tools" / "temp" / "df_final.parquet"
    r = run_simulations_for_all_dates("160 - Ñuñoa", "2025-05-08", forecast_path)
    print(r[1])
