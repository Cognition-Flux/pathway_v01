# %%
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Self
from uuid import UUID, uuid4

import pandas as pd
import polars as pl
from dotenv import load_dotenv
from typing_extensions import Literal  # noqa: UP035

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
from tpp_agentic_graph_v4.workforce.constants import ServiceConfiguration  # noqa: E402
from tpp_agentic_graph_v4.workforce.models_request import (  # noqa: E402
    NewSimulation as NewSimulationRequest,
)
from tpp_agentic_graph_v4.workforce.service_logger import logger  # noqa: E402

# endregion Local Imports


# region Classes
# TODO: Maybe move this to its own file
class QueueAttention(NamedTuple):
    eventId: UUID | str
    serieId: UUID | str

    emission: datetime
    start: datetime | None
    end: datetime | None

    serviceTime: int
    waitingTime: int | None = None
    deskId: UUID | None = None


class DeskEventLog(NamedTuple):
    """This is just to type correctly the events in the desk log"""

    eventId: UUID | str
    timestamp: datetime
    duration: int
    type: Literal["event", "break"]


class SimulationStepLog(NamedTuple):
    timestamp: datetime
    desks_active: int
    desks_on_break: int
    queue_waiting: int
    cumulative_queue_completed: int


class Desk:
    def __init__(
        self,
        deskId: UUID | str,
        skills: list[UUID | str],
        service_configuration: ServiceConfiguration,
    ):
        self.deskId = deskId
        self.skills = skills
        self.service_configuration = service_configuration

        self.status: Literal["idle", "busy", "break", "unavailable"] = "idle"
        self.unavailable_until: None | datetime = None

        self.service_counter: int = 0
        self.break_counter: int = 0

        # A deque is list-like collection that has O(1) appends and pops from both ends,
        # and supports a .rotate() method to rotate the deque n steps to the right or left.
        # For the purpose of a log and alternate, this eases the implementation --Manu

        self.event_log: deque[DeskEventLog] = deque()
        self.skills_priority: deque[UUID] = deque(skills)  # Do not init empty

    def take_event(self, event: QueueAttention, timestamp: datetime) -> None:
        self.status = "busy"
        self.unavailable_until = timestamp + timedelta(seconds=int(event.serviceTime))
        self.service_counter += 1

        self.event_log.append(
            DeskEventLog(
                eventId=event.eventId,
                timestamp=timestamp,
                duration=event.serviceTime,
                type="event",
            )
        )

        # Note that the implementation to step the alternate is part of the queue system

    def take_break(self, timestamp: datetime, break_duration: int = 60):
        self.status = "break"
        self.unavailable_until = timestamp + timedelta(seconds=break_duration)
        self.break_counter += 1

        self.event_log.append(
            DeskEventLog(
                eventId=str(uuid4()),
                timestamp=timestamp,
                duration=break_duration,
                type="break",
            )
        )

    # region Properties and desk info
    def attentions_since_last_break(self) -> int:
        return sum([1 for event in self.event_log if event.type == "event"])

    def total_active_time(self, start_time: datetime = None) -> int:
        """Returns the time in seconds for the sum of service events"""
        if start_time is None:
            return sum(
                [event.duration for event in self.event_log if event.type == "event"]
            )
        else:
            return sum(
                [
                    event.duration
                    for event in self.event_log
                    if (event.type == "event") and (event.timestamp >= start_time)
                ]
            )

    # endregion Properties and desk info

    # Load balancer. Used within the list[Desk] sort method
    # TODO: Apparently, the total usage is also a component of this load balancer

    def __lt__(self, other: Self) -> bool:
        # NOTE: using active time does a better load balancing than the old method
        #   of just using the desk with more time iddle.
        # return self.total_active_time() < other.total_active_time()
        return self.unavailable_until < other.unavailable_until

    # region Representations
    def __str__(self) -> str:
        match self.status:
            case "break" | "busy":
                status = f"{self.status} until {self.unavailable_until.time()}"
            case _:
                status = self.status

        match self.service_configuration:
            case "FIFO" | "Fifo":
                service_mode = f"FIFO, skills: {self.skills}"
            case "Rebalse":
                service_mode = f"priority, order: {self.skills}"
            case "Alternancia":
                service_mode = (
                    f"alternate ({len(self.skills_priority)} iterations), current order: {list(self.skills_priority)}"
                    # noqa: E501
                )
            case _:
                service_mode = self.service_configuration

        return f"Desk: status: {status}, service mode: {service_mode}, requests: {self.service_counter}, breaks: {self.break_counter}"  # noqa: E501

    # endregion Representations

    # region Class methods
    @staticmethod
    def break_generator() -> int:
        """Returns a break length in seconds, or None if it should not take a break"""
        # TODO: Complete this method or abstract to its own class

        from numpy.random import gamma

        break_duration: int = int(gamma(shape=2.2, scale=2.0, size=None) * 60 + 120)
        return random.choices(population=(break_duration, 0), weights=(0.3, 0.7), k=1)[
            0
        ]

    # endregion Class methods


class Queue:
    """Stores an attention table and provides methods to select attentions (FIFO, Priority, alternate)"""

    def __init__(self, events: pl.DataFrame, desk_ids: list[UUID]) -> None:
        # Assign enum columns for status and desks
        attention_status = pl.Enum(
            ["not_arrived", "on_queue", "in_progress", "completed", "unserviceable"]
        )
        desks_ids = pl.Enum(desk_ids)

        # region Schema validation
        try:
            # NOTE: use select instead of with_columns to keep only the relevant columns
            events_df = (
                events.select(
                    pl.col("eventId").cast(pl.Categorical),
                    pl.col("serieId").cast(pl.Categorical),
                    pl.lit(None, dtype=desks_ids).alias("deskId"),
                    pl.col("emission").cast(pl.Datetime),
                    pl.lit(None).cast(pl.Datetime).alias("start"),
                    pl.lit(None).cast(pl.Datetime).alias("end"),
                    pl.col("serviceTime").cast(pl.UInt64),
                    pl.lit(None).cast(pl.UInt64).alias("waitingTime"),
                    pl.lit("not_arrived", dtype=attention_status).alias("status"),
                )
                .sort(by="emission")
                .rechunk()
            )

        except Exception as e:
            raise ValueError("The DataFrame does not have the correct schema") from e

        # endregion Schema validation

        # Set it as pandas DF, for ease of access to the data
        self.events: pd.DataFrame = events_df.to_pandas().set_index("eventId")

    # region Simulation runtime methods
    def step(self, timestamp: datetime) -> None:
        """Updates the arrivals until the given timestamp"""
        # Set the values of the status column to on_queue
        # if the start time is less than the current time and the status is "not_arrived"
        self.events.loc[
            (self.events["status"] == "not_arrived")
            & (self.events["emission"] < timestamp),
            "status",
        ] = "on_queue"

        # NOTE: The setting of the start and end times is managed by the methods below, not this one

        # Set the values of the status column to completed
        # if the end time is less than the current time and the status is "in_progress"
        self.events.loc[
            (self.events["status"] == "in_progress") & (self.events["end"] < timestamp),
            "status",
        ] = "completed"

        logger.debug(self)

    def commit_event(
        self, eventId: UUID, deskId: UUID, timestamp: datetime
    ) -> QueueAttention:  # noqa: N803
        """Commits the event to the queue"""
        # TODO: can this be done more cleanly?
        # This can be done in a single line with multiple assignments. Its the same, but less readable
        self.events.at[eventId, "status"] = "in_progress"
        self.events.at[eventId, "deskId"] = deskId
        self.events.at[eventId, "start"] = timestamp
        self.events.at[eventId, "end"] = timestamp + timedelta(
            seconds=int(self.events.at[eventId, "serviceTime"])
        )
        self.events.at[eventId, "waitingTime"] = int(
            (timestamp - self.events.at[eventId, "emission"]).total_seconds()
        )

        return QueueAttention(
            eventId=eventId,
            serieId=self.events.at[eventId, "serieId"],
            emission=self.events.at[eventId, "emission"],
            start=self.events.at[eventId, "start"],
            end=self.events.at[eventId, "end"],
            serviceTime=int(self.events.at[eventId, "serviceTime"]),
            deskId=deskId,
        )

    # endregion Simulation runtime methods

    # region Selection methods
    # NOTE: The selection methods are agnostic to the Desk object, and do not commit the changes.
    #    All these return the eventId of the selected attention or None if there are no matches

    def fifo(self, skills: list[UUID]) -> UUID | None:
        current_queue = self.events.loc[
            (self.events["status"] == "on_queue")
            & (self.events["serieId"].isin(skills))
        ].sort_values(by="emission")

        try:
            return current_queue.iloc[0].name
        except IndexError:
            return None

    def priority(self, order: list[UUID]) -> UUID | None:
        for serie in order:
            current_queue = self.events.loc[
                (self.events["status"] == "on_queue")
                & (self.events["serieId"] == serie)
            ]

            if current_queue.shape[0] > 0:
                return current_queue.iloc[0].name
            else:
                continue
        return None

    def alternate(self, skills_deque: deque[UUID]) -> UUID | None:
        """
        Rotates over a deque of series to select the first one that has an event in the queue

        Note it modifies the deque in place.

        Params
        ------
        deque: A deque of series to rotate over, from a desk. Modified in place.
        """
        for _ in range(len(skills_deque)):  # can be while true,
            series = skills_deque[0]
            current_queue = self.events.loc[
                (self.events["status"] == "on_queue")
                & (self.events["serieId"] == series)
            ]
            # If dataframe has a result, return it and step the queue
            if current_queue.shape[0] > 0:
                skills_deque.rotate(1)
                return current_queue.iloc[0].name
            # Else, step the deque and continue until next series is reached
            else:
                # If all series are the same, return None (this is a failed state that Optuna can suggest)
                if len(set(skills_deque)) == 1:
                    return None
                while skills_deque[0] == series:
                    skills_deque.rotate(1)
                continue
        return None  # being the case above the loop failed

    # endregion Selection methods

    # region Class methods
    def __repr__(self) -> str:
        return self.events.__repr__()

    def __str__(self) -> str:
        counts = self.events["status"].value_counts()
        return f"Queue: {counts.in_progress} in progress, {counts.on_queue} waiting, {counts.completed} completed, {counts.not_arrived} to arrive ({counts.sum()} total)"

    # IDEA: Implement a __getattribute__ method to select "on_queue", "in_progress", etc. attentions

    # endregion Class methods


class Scheduler:
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        simulation_step: timedelta,
        current_time: datetime | None = None,
    ):
        # Set the component variables
        self.start_time = start_time
        self.end_time = end_time
        self.simulation_step = simulation_step

        # Set the scheduler to the start time if the current time is not given
        if current_time is None:
            self.current_time = start_time

        assert start_time <= self.current_time <= end_time, (
            "The start time should be before the end time"
        )

        # Set the total steps and the remaining steps
        self.total_steps: int = int((end_time - start_time) / simulation_step)
        self.remaining_steps: int = self.total_steps
        self.current_step: int = 0

    # This function should be evaluated at every step of the simulation
    def step_simulation(self) -> bool:
        """Returns True if the simulation should continue"""

        self.current_time = self.current_time + self.simulation_step
        self.remaining_steps -= 1
        self.current_step += 1
        return self.current_time < self.end_time

    def __str__(self) -> str:
        return f"Scheduler: {self.current_time.time()} ({self.start_time.time()} to {self.end_time.time()}) ({self.remaining_steps} steps remaining)"  # noqa: E501


class Simulation:
    """
    Encapsulates and provides methods to run the Simulations

    Parameters
    ----------
    scheduler: a Scheduler object that tracks the current time
    desks: the agents that can service the events served
    queue: a Queue object that holds the events to serve and the methods to serve them

    Methods
    -------

    """

    def __init__(self, scheduler: Scheduler, desks: list[Desk], queue: Queue):
        self.scheduler = scheduler
        self.desks = desks
        self.queue = queue

        self.steps_log = deque()  # O(1) list for loggin steps

        # logger.info(f"{self}")

    # region Simulation runtime methods
    def initialize(self) -> None:
        """Makes all the desks available at the simulation start time, and sets the scheduler to it"""
        # Doesn't run remove_unserviceable_events as the configuration may change during the simulation
        # self.remove_unserviceable_events()

        self.scheduler.current_time = self.scheduler.start_time
        for desk in self.desks:
            desk.unavailable_until = self.scheduler.start_time

    def remove_unserviceable_events(self) -> None:
        """Remove from the queue the events that cannot be served"""
        simulation_available_series = set(
            [ds for desk in self.desks for ds in desk.skills]
        )
        unserviceable_events: pd.Series = ~self.queue.events.serieId.isin(
            simulation_available_series
        )
        if unserviceable_events.any():
            logger.warning(
                f"Removing {unserviceable_events.sum()} unserviceable events"
            )
            self.queue.events.loc[unserviceable_events, "status"] = "unserviceable"

    def step(self):
        """
        Runs the events of a step of the simulation

        Note it DOES NOT increment the scheduler status, this is done in the main system loop
        """
        # If the queue is empty or all desks are busy, return
        if not (self.queue.events.status == "on_queue").any():
            return None

        # Sort the desks by time idle, as by
        self.desks.sort()

        # region SYSTEM: Use the queue to assign events to the desks
        for desk in self.desks:
            if self.scheduler.current_time <= desk.unavailable_until:
                continue  # If the desk is unavailable
            else:
                # Set the desk to idle and maybe take a break
                desk.status = "idle"
                if break_duration := Desk.break_generator():
                    desk.take_break(
                        timestamp=self.scheduler.current_time,
                        break_duration=break_duration,
                    )

                # Take an event from the main queue
                match desk.service_configuration:
                    case "FIFO" | "Fifo":
                        eventId = self.queue.fifo(skills=desk.skills)  # noqa: N806
                    case "Rebalse":
                        eventId = self.queue.priority(order=desk.skills)  # noqa: N806
                    case "Alternancia":
                        eventId = self.queue.alternate(
                            skills_deque=desk.skills_priority
                        )  # noqa: N806
                    case _:
                        raise ValueError("Invalid service mode")

                # Commit and assign the event to the desk
                if eventId is None:
                    logger.debug(f"No event for desk {desk.deskId}")
                    continue
                else:
                    event = self.queue.commit_event(
                        eventId=eventId,
                        deskId=desk.deskId,
                        timestamp=self.scheduler.current_time,
                    )
                    desk.take_event(event, timestamp=self.scheduler.current_time)
                    logger.debug(f"Desk {desk.deskId} took {event}")
        # endregion SYSTEM: Use the queue to assign events to the desks

    def run_simulation(
        self, stop_time: None | datetime = None, debug: bool = False
    ) -> None:
        """Runs the simulation until the end time or the stop time

        :param stop_time: The time to stop (early) the simulation
        """
        while self.scheduler.step_simulation() and (
            stop_time is None or self.scheduler.current_time < stop_time
        ):
            self.queue.step(self.scheduler.current_time)
            self.step()

            self.steps_log.append(self.step_log())  # Add this step to the loggin

            if debug and (self.scheduler.remaining_steps % 30 == 0):
                logger.info(self)

    def run_overtime(
        self,
        overtime_limit: timedelta | None,
        remove_unserviceable: bool = True,
        debug: bool = False,
    ) -> timedelta:
        """Runs the simulation for a given overtime limit or until the queue is empty. Returns the total overtime"""

        if remove_unserviceable:
            self.remove_unserviceable_events()

        # Run the overtime
        while (overtime_limit is None) or (
            self.scheduler.current_time < self.scheduler.current_time + overtime_limit
        ):
            # Are we waiting for someone?
            if (
                self.queue.events.status.value_counts()[
                    ["not_arrived", "in_progress"]
                ].sum()
                == 0
            ):
                break

            self.scheduler.step_simulation()  # Will return false
            self.queue.step(self.scheduler.current_time)
            self.step()

            self.steps_log.append(self.step_log())  # Add this step to the loggin

            if debug and (self.scheduler.remaining_steps % 30 == 0):
                logger.info(self)

        return self.scheduler.current_time - self.scheduler.end_time  # Total overtime

    # endregion Simulation runtime methods

    # region Representations
    def step_log(self) -> SimulationStepLog:
        """Returns a step logger object"""

        queue_stats = self.queue.events["status"].value_counts()

        return SimulationStepLog(
            timestamp=self.scheduler.current_time,
            desks_active=sum(d.status == "busy" for d in self.desks),
            desks_on_break=sum(d.status == "break" for d in self.desks),
            queue_waiting=queue_stats.on_queue,
            cumulative_queue_completed=queue_stats.completed,
        )

    def get_steps_log_df(self) -> pd.DataFrame:
        # TODO: include logic for more cumulative stats, such as desk usage
        return pd.DataFrame(self.steps_log).set_index("timestamp").sort_index()

    def __str__(self) -> str:
        # TODO: do implement a representation of the simulation (use colors)
        return f"Simulation: step {self.scheduler.current_step}\n\t{self.scheduler}\n\t{self.queue}\n\t{[str(desk) for desk in self.desks]}"  # noqa: E501

    # endregion Representations

    # region Constructors
    @classmethod
    def from_simulation_request(
        cls, request: NewSimulationRequest, events: pl.DataFrame
    ) -> Self:
        """
        Constructs a simulation from a request and a set of events

        NOTE: This may be slow but only runs once, thus is not a bottleneck.
        """

        # region Make the desks
        # TODO: move the builder to the Desk class?
        desks = []
        for req_desk in request.desks:
            match req_desk.serviceConfiguration:
                case "FIFO" | "Fifo":
                    skills = [str(ds.serieId) for ds in req_desk.deskSeries]
                case "Rebalse":
                    req_desk.deskSeries.sort()
                    skills = [str(ds.serieId) for ds in req_desk.deskSeries]
                case "Alternancia":
                    req_desk.deskSeries.sort()
                    skills = [str(ds.serieId) for ds in req_desk.deskSeries]
                    skills_deque: deque[UUID | str] = deque()
                    for ds in req_desk.deskSeries:
                        for _ in range(ds.step):
                            skills_deque.append(str(ds.serieId))

            desk = Desk(
                deskId=str(req_desk.deskId),
                skills=skills,
                service_configuration=req_desk.serviceConfiguration,
            )

            if req_desk.serviceConfiguration == "Alternancia":
                desk.skills_priority = skills_deque

            desks.append(desk)
        # endregion Make the desks

        queue = Queue(events, desk_ids=[d.deskId for d in desks])

        scheduler = Scheduler(
            start_time=datetime.combine(request.targetDate, request.startTime),
            end_time=datetime.combine(request.targetDate, request.endTime),
            simulation_step=timedelta(minutes=1),
        )

        return cls(
            desks=desks,
            queue=queue,
            scheduler=scheduler,
        )

    # endregion Constructors


# endregion Classes


def main(_n_events=1_800, _desks=15):
    from time import time as timer

    logger.setLevel(logging.INFO)

    st = timer()
    logger.info("Running the simulation as test!")
    from datetime import date, datetime, time

    _series = [str(uuid4()) for _ in range(5)]

    # region Helper: generate events
    events = pd.DataFrame(
        {
            "eventId": [str(uuid4()) for _ in range(_n_events)],
            "serieId": random.choices(
                _series, weights=[0.19, 0.15, 0.06, 0.22, 0.37], k=_n_events
            ),
            "serviceTime": [random.randint(100, 600) for _ in range(_n_events)],
            "emission": [
                datetime(year=2024, month=4, day=12, hour=9, minute=0)
                + timedelta(seconds=random.randint(1, 3600 * 7))
                for _ in range(_n_events)
            ],
        }
    )

    events = pl.from_pandas(
        events,
        include_index=False,
    )
    # endregion Helper: generate events

    # region Helper: NewSimulationRequest

    from .models_request import (
        Desk as DeskRequest,
    )
    from .models_request import (
        DeskSerie as DeskSerieRequest,
    )
    from .models_request import (
        Serie as SerieRequest,
    )

    # noinspection PyTypeChecker
    request = NewSimulationRequest(
        officeId=uuid4(),
        targetDate=date(year=2024, month=4, day=12),
        startTime=time(hour=8, minute=0),
        endTime=time(hour=17, minute=0),
        series=[SerieRequest(serieId=s) for s in _series],
        desks=[
            DeskRequest(
                deskId=uuid4(),
                serviceConfiguration="FIFO",
                # random.choice(["Alternancia", "FIFO", "Rebalse"]),  # Alternancia | FIFO | Rebalse
                deskSeries=[
                    DeskSerieRequest(
                        serieId=ds, priority=p + 1, step=random.randint(1, 4)
                    )
                    for p, ds in enumerate(_series)  # randomly sorted
                ],
            ).model_dump()
            for _ in range(_desks)
        ],
    )

    # endregion Helper: NewSimulationRequest

    # region SYSTEM: Run the simulation
    simulation = Simulation.from_simulation_request(request=request, events=events)
    simulation.initialize()

    # simulation.scheduler.end_time = simulation.scheduler.start_time + timedelta(days=5)

    # add some noise to the desk configuration
    for desk in simulation.desks:
        desk.skills_priority.rotate(random.randint(0, len(desk.skills_priority)))

    logger.info(simulation)
    simulation.run_simulation(debug=True)
    logger.info(simulation)

    # endregion SYSTEM: Run the simulation
    logger.info(f"Simulation completed in {(timer() - st):.3f} seconds.")
    logger.info(
        f"Average waiting time: {simulation.queue.events.waitingTime.mean():.3f} seconds."
    )

    logger.info("\n\n\n\n\n")

    total_overtime = simulation.run_overtime(
        overtime_limit=timedelta(hours=5), debug=True
    )
    logger.info(simulation)
    logger.info(f"Total overtime: {total_overtime}")

    # with pd.option_context("display.max_rows", 10, "display.max_columns", 10, "expand_frame_repr", False):
    #     print(simulation.queue.events)
    #     print(simulation.desks[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the simulation")
    parser.add_argument(
        "-e", "--n-events", type=int, help="Number of events to simulate", default=1400
    )
    parser.add_argument(
        "-d", "--n-desks", type=int, help="Number of desks to simulate", default=20
    )
    parser.add_argument(
        "--debug", action="store_true", help="Increase output verbosity", default=False
    )

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args.n_events, args.n_desks)
