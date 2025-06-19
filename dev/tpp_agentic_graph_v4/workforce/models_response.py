"""
This module contains the models for the workforce response.
"""

# %%
import os
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Generic, Self
from uuid import UUID, uuid4

import pandas as pd
import polars as pl
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, TypeAdapter, model_validator
from typing_extensions import TypeVar

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

# region Local Imports
from tpp_agentic_graph_v4.workforce.constants import (  # noqa: E402
    UUID_ZERO,
    ServiceConfiguration,
)
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    NewWorkforceSerie,
    Serie,
)

# endregion Local Imports


T = TypeVar = TypeVar("T")


# region Global Response
# noinspection PyTypeHints
class GlobalResponse(BaseModel, Generic[T]):
    data: T
    status: bool
    code: int
    message: str


# endregion Global Response


# region Forecast and Simulation Response
class SlaValue(BaseModel):
    serieId: UUID | int | str = Field(
        validation_alias=AliasChoices("serieId", "IdSerie")
    )
    sla: float = Field(validation_alias=AliasChoices("sla", "slaValue"))
    eventsCount: int = Field(validation_alias=AliasChoices("eventsCount", "n_ate"))
    desksCount: int = Field(validation_alias=AliasChoices("desksCount", "n_esc"))
    avgServiceTime: float = Field(
        validation_alias=AliasChoices("avgServiceTime", "t_ate")
    )
    avgWaitingTime: float = Field(
        validation_alias=AliasChoices("avgWaitingTime", "t_esp")
    )
    eventsNotServicedCount: int = Field(
        0, validation_alias=AliasChoices("eventsNotServicedCount", "no_ate")
    )
    timestamp: datetime | None = Field(
        None, validation_alias=AliasChoices("timestamp", "FH")
    )


class DeskUsage(BaseModel):
    timestamp: datetime = Field(validation_alias=AliasChoices("timestamp", "FH"))
    deskId: UUID | int = Field(validation_alias=AliasChoices("deskId", "IdEsc"))
    deskLegacyId: int | None = None

    # Result of the given solution
    activeTime: float = Field(validation_alias=AliasChoices("activeTime", "t_ate"))
    inactiveTime: float = Field(
        0, validation_alias=AliasChoices("inactiveTime", "t_inac")
    )
    availableTime: float = Field(
        0, validation_alias=AliasChoices("availableTime", "t_disp")
    )
    percentActive: float = Field(
        validation_alias=AliasChoices("percentActive", "porcentaje_ocupacion")
    )

    eventsCount: int = Field(validation_alias=AliasChoices("eventsCount", "n_ate"))


class ResponseSimulation(BaseModel):
    # Response itself
    simulationId: UUID | None = uuid4()

    # Parameters for the plots
    slaHour: list[SlaValue] = Field(
        validation_alias=AliasChoices("slaHour", "sla_instantaneo")
    )
    slaDay: list[SlaValue] = Field(
        validation_alias=AliasChoices("slaDay", "sla_cumulativo")
    )
    desksUsage: list[DeskUsage] = Field(
        validation_alias=AliasChoices("desksUsage", "uso_escritorios")
    )

    @classmethod
    def from_event_queue(
        cls,
        events: pl.DataFrame,
        series: list[Serie] | list[NewWorkforceSerie] | None = None,
    ) -> Self:
        """Takes a table of events and returns a ResponseSimulation"""
        # TODO: Check if the filled values are appropied for when the waiting time and whatever are null

        # Convert pandas dataframe to polars if needed
        if isinstance(events, pd.DataFrame):
            events = pl.DataFrame(events)

        # Select the columns needed only
        events = events.select(
            pl.col("eventId"),
            pl.col("serieId").cast(pl.Categorical),
            pl.col("deskId"),
            pl.col("emission"),
            pl.col("start"),
            pl.col("end"),
            pl.col("serviceTime").cast(pl.UInt64),
            pl.col("waitingTime").cast(pl.UInt64),
        )

        # Make a table to compute the SLAs
        sla_tresholds = pl.DataFrame(
            {
                "serieId": [str(s.serieId).upper() for s in series],
                "maximumWaitingTime": [s.maximumWaitingTime for s in series],
            }
        ).select(
            pl.col("serieId").cast(pl.Categorical),
            pl.col("maximumWaitingTime").cast(pl.UInt64),
        )

        events = events.join(sla_tresholds, on="serieId").with_columns(
            (pl.col("waitingTime") < pl.col("maximumWaitingTime")).alias("slaValue")
        )

        # region Compute the daily SLAs

        sla_day = (
            events.group_by(pl.col("serieId"))
            .agg(
                pl.col("slaValue").fill_null(True).mean(),
                pl.col("eventId").len().alias("eventsCount"),
                pl.col("deskId").n_unique().alias("desksCount"),
                pl.col("serviceTime").fill_null(1.0).mean().alias("avgServiceTime"),
                pl.col("waitingTime").fill_null(1.0).mean().alias("avgWaitingTime"),
                pl.col("deskId").null_count().alias("eventsNotServicedCount"),
            )
            .to_dicts()
        )

        # Append as a dict is easier than making a new DataFrame
        sla_day.append(
            {
                "serieId": UUID_ZERO,
                "slaValue": events["slaValue"].fill_null(True).mean(),
                "eventsCount": events["slaValue"].len(),
                "desksCount": events["deskId"].n_unique(),
                "avgServiceTime": events["serviceTime"].fill_null(0.0).mean(),
                "avgWaitingTime": events["waitingTime"].fill_null(0.0).mean(),
                "eventsNotServicedCount": events["deskId"].null_count(),
            }
        )

        # endregion Compute the SLAs
        slaDay: list[SlaValue] = [SlaValue(**s) for s in sla_day]  # noqa: N806

        # region Compute the SLA per hour
        sla_hour_serie = events.group_by(
            pl.col("emission").dt.truncate("1h"), pl.col("serieId")
        ).agg(
            pl.col("slaValue").fill_null(True).mean(),
            pl.col("eventId").len().alias("eventsCount"),
            pl.col("deskId").n_unique().alias("desksCount"),
            pl.col("serviceTime").fill_null(0.0).mean().alias("avgServiceTime"),
            pl.col("waitingTime").fill_null(0.0).mean().alias("avgWaitingTime"),
            pl.col("deskId").null_count().alias("eventsNotServicedCount"),
        )

        sla_hour_global = events.group_by(pl.col("emission").dt.truncate("1h")).agg(
            pl.lit(UUID_ZERO).cast(pl.Categorical).alias("serieId"),
            pl.col("slaValue").fill_null(True).mean(),
            pl.col("eventId").len().alias("eventsCount"),
            pl.col("deskId").n_unique().alias("desksCount"),
            pl.col("serviceTime").fill_null(0.0).mean().alias("avgServiceTime"),
            pl.col("waitingTime").fill_null(0.0).mean().alias("avgWaitingTime"),
            pl.col("deskId").null_count().alias("eventsNotServicedCount"),
        )

        # Stacks the sets and rename the Emission column to timestamp
        # Ignore this warning. Is annoying
        sla_hour: pl.DataFrame = (
            sla_hour_serie.vstack(sla_hour_global)
            .with_columns(pl.col("emission").alias("timestamp"))
            .sort(pl.col("serieId"), pl.col("timestamp"))
        )

        # TODO: annoingly, this requieres an index to sort as the dictionary if the hours have gaps

        # endregion Compute the SLA per hour
        slaHour: list[SlaValue] = [SlaValue(**s) for s in sla_hour.to_dicts()]  # noqa: N806

        # region Compute the desks ussage
        desk_usage = (
            events.group_by(pl.col("start").dt.truncate("1h"), pl.col("deskId"))
            .agg(
                pl.col("serviceTime").sum().clip_max(3600).alias("activeTime"),
                pl.lit(0.0).alias("inactiveTime"),
                pl.lit(0.0).alias("availableTime"),
                (pl.col("serviceTime").sum() / 3600)
                .clip_max(1.0)
                .alias("percentActive"),
                pl.col("eventId").len().alias("eventsCount"),
            )
            .drop_nulls(subset="deskId")
            .with_columns(pl.col("start").alias("timestamp"))
            .sort(pl.col("deskId"), pl.col("start"))
        )

        # endregion Compute the desks ussage
        deskUsage: list[DeskUsage] = [DeskUsage(**d) for d in desk_usage.to_dicts()]  # noqa: N806

        # DEBUG: Print the tables
        # with pl.Config(tbl_cols=-1, tbl_width_chars=550, tbl_rows=-1):
        #     print(events)
        #     print(desk_usage)
        #     print(sla_hour)

        return cls(
            slaHour=slaHour,
            slaDay=slaDay,
            desksUsage=deskUsage,  # Nothing here
        )


# endregion Forecast and Simulation Response


# region Workforce Response


# FIXME: Deprecated!
class EscritorioSerieAttributos(BaseModel):
    serie: int
    sla_porcen: float  # Convertir a %
    sla_corte: int
    prioridad: int
    pasos: int | None

    @model_validator(mode="after")
    def convert_pasos(self):
        if self.pasos is None:
            self.pasos = 1

        return self

    @model_validator(mode="after")
    def convertir_sla_a_percent(self):
        if self.sla_porcen > 1:
            self.sla_porcen /= 100

        return self


class EscritorioPropiedades(BaseModel):
    skills: list[int]
    configuracion_atencion: str
    porcentaje_actividad: float
    atributos_series: list[EscritorioSerieAttributos]

    @model_validator(mode="after")
    def elimina_series_no_skill(self) -> "EscritorioPropiedades":
        """Si la serie no está en Skills, se remueve"""
        self.atributos_series = [
            serie for serie in self.atributos_series if serie.serie in self.skills
        ]

        return self


class EscritorioPlanningInterval(BaseModel):
    inicio: str
    termino: str | None
    propiedades: EscritorioPropiedades


DeskPlanningAsDict = dict[str, list[EscritorioPlanningInterval]]


# endregion


class WorkforceResponseDesk(BaseModel):
    """Contains a Desk and the Series (by name) it services"""

    deskId: UUID = Field(validation_alias=AliasChoices("deskId", "DeskId"))
    deskLegacyId: int = Field(validation_alias=AliasChoices("deskLegacyId", "LegacyId"))
    series: list[str | UUID] = Field(
        validation_alias=AliasChoices("series", "AvailableSeries")
    )


class DeskPlanningInterval(BaseModel):
    startTime: time
    endTime: time | None

    # Response itself
    seriesIds: list[UUID]
    seriesSteps: list[int]
    serviceConfiguration: ServiceConfiguration

    eventsCount: int

    @model_validator(mode="after")
    def escritorios_tienen_skills(self) -> "DeskPlanningInterval":
        """Si el escritorio solo atiende una serie, la configuracion es FIFO"""
        if len(self.seriesIds) == 1:
            self.serviceConfiguration = "FIFO"

        return self

    @model_validator(mode="after")
    def escritorios_tienen_pesos(self) -> "DeskPlanningInterval":
        """Codifica la configuracion de Atencion como una Alternancia"""
        if self.serviceConfiguration != "Alternancia":
            self.seriesSteps = [1]

        return self


class WorkforceDeskPlanning(BaseModel):
    """Planificación óptima"""

    deskId: UUID
    deskLegacyId: int
    deskPlanningIntervals: list[DeskPlanningInterval]

    @staticmethod
    def from_desk_planning(
        input_model: DeskPlanningAsDict,
        anti_mapper_series: dict[int, UUID],
        anti_mapper_desks: dict[int, UUID],
    ) -> list["WorkforceDeskPlanning"]:
        """
        Convierte el objeto de planificacion de Alejandro en una salida para Omar
        """

        input_model = TypeAdapter(DeskPlanningAsDict).validate_python(input_model)

        desk_plannings: list[WorkforceDeskPlanning] = []

        for desk_legacy_id, planning_intervals in input_model.items():
            desk_planning = WorkforceDeskPlanning(
                deskId=str(anti_mapper_desks[int(desk_legacy_id)]),
                deskLegacyId=int(desk_legacy_id),
                deskPlanningIntervals=[],
            )
            desk_planning.deskPlanningIntervals = []

            for planning in planning_intervals:
                planning_interval = DeskPlanningInterval(
                    startTime=planning.inicio,
                    endTime=planning.termino,
                    eventsCount=0,  # TODO: Reemplazar por los eventos que hizo el escritorio
                    seriesIds=[
                        anti_mapper_series[attr_serie.serie]  # MAPEA A GUIDS
                        for attr_serie in planning.propiedades.atributos_series
                    ],
                    seriesSteps=(
                        [
                            attr_serie.pasos
                            for attr_serie in planning.propiedades.atributos_series
                        ]
                        if planning.propiedades.atributos_series
                        else [1]
                    ),
                    serviceConfiguration=planning.propiedades.configuracion_atencion,
                )

                desk_planning.deskPlanningIntervals.append(planning_interval)

            desk_plannings.append(desk_planning)

        return desk_plannings


class WorkforceSerie(BaseModel):
    serieId: UUID
    serieName: str = ""
    serieLegacyId: int = 99
    priority: int
    maximumWaitingTime: int
    percentTargetSLA: float
    avgServiceTime: float

    @model_validator(mode="after")
    def fix_percent_target_sla(self) -> "WorkforceSerie":
        """Fix the percentTargetSLA to be in the range [0, 1]"""
        if self.percentTargetSLA > 1:
            self.percentTargetSLA /= 100

        return self


class WorkforceDeskUsage(BaseModel):
    # FIXME: its the same than DeskUsage
    timestamp: datetime | None  # Por el día
    deskId: UUID | str
    deskLegacyId: int = 0

    # Result of the given solution
    activeTime: float
    inactiveTime: float
    availableTime: float
    percentActive: float

    eventsCount: int

    def __lt__(self, other):
        return (self.deskLegacyId, self.timestamp) < (
            other.deskLegacyId,
            self.timestamp,
        )


class WorkforceResponse(BaseModel):
    officeName: str
    targetDate: date

    # Echo parameters
    series: list[WorkforceSerie]
    desks: list[WorkforceResponseDesk]
    percentActive: float

    # Response itself
    deskPlannings: list[WorkforceDeskPlanning]

    # Parameters for the plots
    slaHour: list[SlaValue]
    slaDay: list[SlaValue]
    desksUsage: list[WorkforceDeskUsage]

    slaDayHistoric: list[SlaValue]

    # region Chain hell
    def replace_series_by_names(self) -> Self:
        mapper_series: dict[UUID, str] = {
            serie.serieId: serie.serieName for serie in self.series
        }
        for desk in self.desks:
            desk.series = [mapper_series.get(UUID(str(s)), s) for s in desk.series]
        return self

    def fix_desk_legacy_id(self) -> Self:
        mapper_desks: dict[UUID, int] = {
            UUID(str(desk.deskId)): desk.deskLegacyId for desk in self.desks
        }
        for d in self.desksUsage:
            d.deskLegacyId = mapper_desks.get(UUID(str(d.deskId)), 0)

        return self

    # endregion Chain hell


# endregion Workforce Response
