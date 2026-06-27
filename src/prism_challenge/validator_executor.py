"""Decentralized validator execution of an assigned prism work unit.

The assigned (online, gpu) validator pulls its single prism work unit from the master coordination
plane and runs the WHOLE re-execution by dispatching to its OWN broker-backed ``DockerExecutor``
(monkeypatched to the CPU re-exec mock in tests). The master coordinator never invokes the executor
for the prism unit - it only assigns the work (VAL-PRISM-037).

Execution reuses :class:`~prism_challenge.queue.PrismWorker`, whose container path builds the
validator's broker executor and preserves forced-random-init + prequential bits-per-byte scoring.
Driving it through :meth:`PrismWorker.process_submission` (a CAS claim on the specific submission)
makes the loop idempotent and concurrency-safe: re-running a completed/in-flight unit is a no-op
that neither re-dispatches the broker nor mutates the recorded result (VAL-PRISM-002 / 004).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .coordination import (
    PRISM_DEFAULT_CONCURRENCY,
    PRISM_WORK_UNIT_CAPABILITY,
    PrismWorkUnit,
    list_pending_prism_work_units,
    pull_assigned_work_units,
)
from .queue import PrismWorker

#: Submission statuses at which a prism work unit is terminal (no re-execution, no re-dispatch).
TERMINAL_SUBMISSION_STATUSES = frozenset({"completed", "failed", "rejected"})


@dataclass(frozen=True)
class PrismWorkUnitExecution:
    """Outcome of executing (or idempotently skipping) one assigned prism work unit."""

    work_unit_id: str
    submission_id: str
    status: str
    #: True when the validator's broker was actually dispatched (False = idempotent no-op).
    executed: bool
    #: True when a fresh result was persisted by this run (False = already terminal).
    posted: bool


@dataclass(frozen=True)
class PrismValidatorCycleSummary:
    """Aggregate of one validator pull/execute/post cycle."""

    pulled: int
    executed: int
    skipped: int
    completed_submissions: tuple[str, ...]


async def execute_work_unit(worker: PrismWorker, unit: PrismWorkUnit) -> PrismWorkUnitExecution:
    """Run one assigned prism re-execution on the validator's own broker and report the outcome.

    Idempotent: :meth:`PrismWorker.process_submission` claims the submission only while it is
    pending, so a unit that already reached a terminal state is not re-dispatched and its recorded
    result is left untouched.
    """

    result_id = await worker.process_submission(unit.submission_id)
    executed = result_id is not None
    status = await worker.repository.submission_status(unit.submission_id)
    return PrismWorkUnitExecution(
        work_unit_id=unit.work_unit_id,
        submission_id=unit.submission_id,
        status=status or "",
        executed=executed,
        posted=executed,
    )


async def run_validator_cycle(
    *,
    worker: PrismWorker,
    work_unit_ids: Iterable[str] | None = None,
    capabilities: Iterable[str] = (PRISM_WORK_UNIT_CAPABILITY,),
    in_flight: int = 0,
    max_concurrency: int = PRISM_DEFAULT_CONCURRENCY,
) -> PrismValidatorCycleSummary:
    """Run one decentralized validator cycle: pull -> execute (own broker) -> post.

    Pulls the caller's assigned, capability-matched prism units (at most
    ``max_concurrency - in_flight`` of them, so a busy validator runs one submission at a time),
    executes each on the validator's own broker, and reports which submissions completed. The pull
    and assignment are execution-free; only :func:`execute_work_unit` dispatches the broker.
    """

    units = await list_pending_prism_work_units(worker.repository)
    pulled = pull_assigned_work_units(
        units,
        work_unit_ids=work_unit_ids,
        capabilities=capabilities,
        in_flight=in_flight,
        max_concurrency=max_concurrency,
    )
    executed = 0
    skipped = 0
    completed: list[str] = []
    for unit in pulled:
        outcome = await execute_work_unit(worker, unit)
        if outcome.executed:
            executed += 1
        else:
            skipped += 1
        if outcome.status == "completed":
            completed.append(outcome.submission_id)
    return PrismValidatorCycleSummary(
        pulled=len(pulled),
        executed=executed,
        skipped=skipped,
        completed_submissions=tuple(completed),
    )
