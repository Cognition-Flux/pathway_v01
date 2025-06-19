"""
This module contains the models for the workforce request.
"""

# %%
import os
import sys
from datetime import date, time
from pathlib import Path
from uuid import UUID, uuid5

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal, Self

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

# region Local imports
from tpp_agentic_graph_v4.workforce.constants import ServiceConfiguration  # noqa: E402

# endregion Local imports


# region Forecast Request
class Serie(BaseModel):
    """
    Represents a series with its properties.

    Attributes
    ----------
    serieId : UUID
        The ID of the series.
    maximumWaitingTime : int
        The maximum waiting time for the series.
    percentTargetSLA : float
        The target SLA for the series, as a fraction from 0-1.
    internal_IdSerie : int
        The ID of the series in the database, in int format.
    """

    serieId: UUID | str
    maximumWaitingTime: int = 600
    percentTargetSLA: float = 0.8
    internal_IdSerie: int = 0

    @model_validator(mode="after")
    def uppercasing(self):
        self.serieId = str(self.serieId).upper()

        return self


class NewForecast(BaseModel):
    officeId: UUID
    targetDate: date
    startTime: time
    endTime: time
    series: list[Serie]
    maximumWaitingTime: int

    def get_forecast_uuid(self) -> UUID:
        # Define a namespace and a name
        namespace = self.officeId
        string_to_hash = f"{self.targetDate} {self.startTime} {self.endTime} {[(s.serieId, s.maximumWaitingTime) for s in self.series]}"  # noqa: E501

        print("\n\nHASHING FORECAST:", namespace, string_to_hash)

        # Generate a deterministic UUID
        deterministic_uuid = uuid5(namespace, string_to_hash)
        return deterministic_uuid


# endregion Forecast Request


# region Simulation Request
class DeskSerie(BaseModel):
    serieId: UUID | str
    priority: int
    step: int = 1

    @model_validator(mode="after")
    def uppercasing(self):
        self.serieId = str(self.serieId).upper()
        return self

    def __lt__(self, other: Self) -> bool:
        return self.priority < other.priority


class Desk(BaseModel):
    """Contiene el Identificador de un escritorio y las series que trabaja."""

    deskId: UUID
    deskSeries: list[DeskSerie]
    serviceConfiguration: ServiceConfiguration
    percentActive: float = 0.8

    @model_validator(mode="after")
    def cambia_el_horrible_fifo(self):
        if self.serviceConfiguration.lower() == "fifo":
            self.serviceConfiguration = "FIFO"

        return self


class NewSimulation(BaseModel):
    """Modelo de Entrada"""

    officeId: UUID
    targetDate: date
    startTime: time
    endTime: time
    desks: list[Desk]
    series: list[Serie]
    maximumWaitingTime: int = 600

    def get_forecast_uuid(self) -> UUID:
        # Define a namespace and a name
        namespace = self.officeId
        string_to_hash = f"{self.targetDate} {self.startTime} {self.endTime} {[(s.serieId, s.maximumWaitingTime) for s in self.series]}"  # noqa: E501

        print("\n\nHASHING SIMULACION:", namespace, string_to_hash)

        # Generate a deterministic UUID
        deterministic_uuid = uuid5(namespace, string_to_hash)
        return deterministic_uuid


# endregion Simulation Request


# region Workforce Request
class NewWorkforceDesk(BaseModel):
    """@Workforce"""

    # TODO: reuse this to the model response

    DeskId: UUID
    AvailableSeries: list[UUID]
    LegacyId: int = 0


class NewWorkforceOffice(BaseModel):
    """@Workforce"""

    OpeningTime: time  # Opening time for the simulation
    ClosingTime: time  # Closigg time for the simulation
    ServiceSpans: int  # Service spans for the simulation, aka, blocks
    PercentActivity: float = 0.7  # Percent activity for the simulation.


class NewWorkforceOptimization(BaseModel):
    """
    @param MinDesks: Should reduce the iddle desk time, and number of desks?
    @param MinSkills: Should reduce the skills in each desk
    """

    MinDesks: bool
    MinSkills: bool


class NewWorkforceSerie(BaseModel):
    """@Workforce"""

    priority: int = Field(alias="Priority")
    maximumWaitingTime: int = Field(alias="MaximumWaitingTime")
    percentTargetSLA: float = Field(alias="PercentTargetSLA")
    serieId: UUID | str = Field(alias="SerieId")
    internal_IdSerie: int = 0

    def __lt__(self, other):
        return self.priority < other.priority


class WorkforceRequest(BaseModel):
    """Available Workforce space for suggesting a Simulation"""

    desks: list[NewWorkforceDesk]
    office: NewWorkforceOffice
    optimization: NewWorkforceOptimization
    series: list[NewWorkforceSerie]


class NewWorkforce(BaseModel):
    """Represents the model as how it's saved to the database"""

    WorkforceId: UUID
    DateRequest: date
    TargetDate: date | None
    ScheduleDate: date | None
    Optimization: str
    Status: Literal["agendado", "completo", "cancelado", "en ejecucion"]
    Request: WorkforceRequest
    Response: str | None
    IsTemplate: bool
    Active: bool
    Start: date | None
    End: date | None
    ScheduleId: UUID | None
    OfficeId: UUID


# endregion Workforce Request
