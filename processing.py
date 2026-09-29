"""Durable processing-control primitives for the Phase A migration.

This module is deliberately not imported by ``main.py`` yet.  The current
FastAPI/BackgroundTasks path therefore remains the production path until a
later migration phase enables the durable adapter.

The PostgreSQL repository uses DB-API 2.0 connections and PostgreSQL
placeholders (``%s``).  The driver dependency is declared for the future
coordinator/worker process, but this module is not imported by the production
FastAPI entrypoint and exposes no endpoint.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
from uuid import UUID, uuid4


UTC = timezone.utc
MAX_REFERENCE_LENGTH = 255
MAX_PROCESSING_VERSION_LENGTH = 100
DEFAULT_LEASE_SECONDS = 60
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_WORKER_LEASE_SECONDS = 60
MAX_TRANSACTION_RETRIES = 3
TRANSACTION_RETRY_BASE_SECONDS = 0.01


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.ACCEPTED: frozenset({JobStatus.QUEUED}),
    JobStatus.QUEUED: frozenset({JobStatus.PROCESSING}),
    JobStatus.PROCESSING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.FAILED_RETRYABLE,
            JobStatus.FAILED_TERMINAL,
        }
    ),
    JobStatus.FAILED_RETRYABLE: frozenset({JobStatus.QUEUED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED_TERMINAL: frozenset(),
}


class ProcessingError(RuntimeError):
    """Base error for durable processing-control failures."""


class JobNotFound(ProcessingError):
    pass


class InvalidJobTransition(ProcessingError):
    pass


class LeaseConflict(ProcessingError):
    pass


class OutboxConflict(ProcessingError):
    pass


class DatabaseOperationError(ProcessingError):
    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def utc_now() -> datetime:
    return datetime.now(UTC)


def _normalise_reference(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > MAX_REFERENCE_LENGTH:
        raise ValueError(f"{field_name} exceeds {MAX_REFERENCE_LENGTH} characters")
    return value


def _normalise_processing_version(value: str) -> str:
    value = _normalise_reference(value, "processing_version")
    if len(value) > MAX_PROCESSING_VERSION_LENGTH:
        raise ValueError(
            f"processing_version exceeds {MAX_PROCESSING_VERSION_LENGTH} characters"
        )
    return value


def _normalise_label(value: str, field_name: str) -> str:
    value = _normalise_reference(value, field_name)
    if len(value) > 100 or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return value


def _validate_delay_seconds(value: int, field_name: str = "retry_delay_seconds") -> int:
    if type(value) is not int or value < 0 or value > 86_400:
        raise ValueError(f"{field_name} must be an integer from 0 to 86400")
    return value


def _validate_capacity(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 10_000:
        raise ValueError("capacity must be an integer from 1 to 10000")
    return value


def _validate_active_slots(value: int, capacity: int) -> int:
    if type(value) is not int or not 0 <= value <= capacity:
        raise ValueError("active_slots must be between 0 and capacity")
    return value


SAFE_OUTBOX_PAYLOAD_KEYS = frozenset(
    {
        "event_key",
        "job_id",
        "scan_id",
        "processing_version",
        "trace_id",
        "status",
        "result_ref",
        "failure_code",
        "attempt_count",
    }
)


def _safe_outbox_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("outbox payload must be a mapping")
    unknown = set(payload) - SAFE_OUTBOX_PAYLOAD_KEYS
    if unknown:
        raise ValueError("outbox payload contains unsupported fields")
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError("outbox payload values must be scalar metadata")
        if value is not None and not isinstance(value, (str, int, bool, float)):
            raise ValueError("outbox payload values must be scalar metadata")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("outbox payload values must be finite")
        result[key] = value
    return result


@dataclass(frozen=True)
class JobSpec:
    scan_id: str
    processing_version: str
    requester_user_id: str
    member_id: str
    business_profile_id: str
    trace_id: str | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    created_at: datetime | None = None

    def validated(self) -> "JobSpec":
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be an integer between 1 and 10")
        trace_id = self.trace_id or str(uuid4())
        return replace(
            self,
            scan_id=_normalise_reference(self.scan_id, "scan_id"),
            processing_version=_normalise_processing_version(self.processing_version),
            requester_user_id=_normalise_reference(
                self.requester_user_id, "requester_user_id"
            ),
            member_id=_normalise_reference(self.member_id, "member_id"),
            business_profile_id=_normalise_reference(
                self.business_profile_id, "business_profile_id"
            ),
            trace_id=_normalise_reference(trace_id, "trace_id"),
            created_at=self.created_at or utc_now(),
        )


@dataclass(frozen=True)
class ProcessingJob:
    id: UUID
    scan_id: str
    processing_version: str
    requester_user_id: str
    member_id: str
    business_profile_id: str
    status: JobStatus
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_token: UUID | None
    lease_expires_at: datetime | None
    heartbeat_at: datetime | None
    next_attempt_at: datetime
    result_ref: str | None
    failure_code: str | None
    trace_id: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class AttemptRecord:
    job_id: UUID
    attempt_number: int
    worker_id: str
    lease_token: UUID
    started_at: datetime
    finished_at: datetime | None = None
    outcome: str | None = None
    error_code: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class LeaseClaim:
    job: ProcessingJob
    attempt: AttemptRecord


@dataclass(frozen=True)
class OutboxEvent:
    id: int
    event_key: str
    job_id: UUID
    event_type: str
    payload: Mapping[str, Any]
    available_at: datetime
    attempt_count: int = 0
    locked_by: str | None = None
    locked_until: datetime | None = None
    delivered_at: datetime | None = None
    last_error_code: str | None = None


@dataclass(frozen=True)
class WorkerLease:
    worker_id: str
    lease_token: UUID
    capacity: int
    active_slots: int
    heartbeat_at: datetime
    lease_expires_at: datetime
    draining: bool
    worker_version: str


@dataclass(frozen=True)
class EnqueuedJob:
    """The canonical job and its transactionally-created dispatch event."""

    job: ProcessingJob
    event: OutboxEvent | None
    created: bool = False


def _assert_transition(current: JobStatus, target: JobStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidJobTransition(f"cannot transition {current.value} -> {target.value}")


def _assert_lease(
    job: ProcessingJob,
    worker_id: str,
    lease_token: UUID,
    now: datetime,
) -> None:
    if (
        job.status is not JobStatus.PROCESSING
        or job.lease_owner != worker_id
        or job.lease_token != lease_token
        or job.lease_expires_at is None
        or job.lease_expires_at <= now
    ):
        raise LeaseConflict("worker lease is missing, expired, or owned by another worker")


def _sqlstate_from_exception(exc: BaseException) -> str | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("sqlstate", "pgcode"):
            value = getattr(current, attribute, None)
            if isinstance(value, str) and value:
                return value
        current = current.__cause__ or current.__context__
    return None


def _is_retryable_transaction_error(exc: BaseException) -> bool:
    return _sqlstate_from_exception(exc) in {"40001", "40P01"}


def _retry_transaction_method(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        for retry_number in range(MAX_TRANSACTION_RETRIES + 1):
            try:
                return method(self, *args, **kwargs)
            except DatabaseOperationError as exc:
                if retry_number >= MAX_TRANSACTION_RETRIES or not _is_retryable_transaction_error(exc):
                    raise
                delay = min(
                    TRANSACTION_RETRY_BASE_SECONDS * (2**retry_number)
                    + random.uniform(0, TRANSACTION_RETRY_BASE_SECONDS),
                    0.25,
                )
                time.sleep(delay)
        raise AssertionError("unreachable transaction retry state")

    return wrapped


class ProcessingRepository(Protocol):
    def create_or_get_job(self, spec: JobSpec) -> ProcessingJob: ...

    def enqueue_job(
        self,
        spec: JobSpec,
        *,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EnqueuedJob: ...

    def get_job(self, job_id: UUID) -> ProcessingJob: ...

    def get_job_by_scan(self, scan_id: str, processing_version: str) -> ProcessingJob | None: ...

    def transition_job(
        self,
        job_id: UUID,
        expected_status: JobStatus,
        target_status: JobStatus,
        *,
        now: datetime | None = None,
    ) -> ProcessingJob: ...

    def claim_job(
        self,
        job_id: UUID,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> LeaseClaim | None: ...

    def heartbeat(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> ProcessingJob: ...

    def assert_lease(
        self, job_id: UUID, worker_id: str, lease_token: UUID
    ) -> ProcessingJob: ...

    def complete_job(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        result_ref: str | None = None,
        now: datetime | None = None,
    ) -> ProcessingJob: ...

    def schedule_retry(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        failure_code: str,
        error_type: str,
        retry_delay_seconds: int = 0,
        terminal_failure: bool = False,
        now: datetime | None = None,
    ) -> ProcessingJob: ...

    def recover_expired_leases(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]: ...

    def queue_due_retries(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]: ...

    def reconcile_queued_job_events(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        min_age_seconds: int = 30,
    ) -> Sequence[OutboxEvent]: ...

    def list_attempts(self, job_id: UUID) -> Sequence[AttemptRecord]: ...

    def add_outbox_event(
        self,
        *,
        event_key: str,
        job_id: UUID,
        event_type: str,
        payload: Mapping[str, Any],
        available_at: datetime | None = None,
        available_delay_seconds: int = 0,
    ) -> OutboxEvent: ...

    def claim_outbox_events(
        self,
        dispatcher_id: str,
        *,
        now: datetime | None = None,
        limit: int = 50,
        lock_seconds: int = 60,
        event_type: str | None = None,
    ) -> Sequence[OutboxEvent]: ...

    def mark_outbox_delivered(
        self, event_id: int, dispatcher_id: str, *, now: datetime | None = None
    ) -> OutboxEvent: ...

    def release_outbox_event(
        self,
        event_id: int,
        dispatcher_id: str,
        *,
        failure_code: str,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> OutboxEvent: ...

    def register_worker(
        self,
        worker_id: str,
        worker_version: str,
        *,
        capacity: int = 1,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    def heartbeat_worker(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        active_slots: int,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    def set_worker_draining(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        draining: bool = True,
        now: datetime | None = None,
    ) -> WorkerLease: ...

    def reap_expired_workers(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[WorkerLease]: ...


class InMemoryProcessingRepository:
    """Deterministic repository used by Phase A unit tests.

    The lock models the atomicity of the PostgreSQL transactions.  It is not
    intended as a production implementation or a distributed lock.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: dict[UUID, ProcessingJob] = {}
        self._job_keys: dict[tuple[str, str], UUID] = {}
        self._attempts: dict[tuple[UUID, int], AttemptRecord] = {}
        self._outbox: dict[int, OutboxEvent] = {}
        self._workers: dict[str, WorkerLease] = {}
        self._next_outbox_id = 1

    @contextmanager
    def transaction(self) -> Iterator["InMemoryProcessingRepository"]:
        with self._lock:
            snapshot = (
                copy.deepcopy(self._jobs),
                copy.deepcopy(self._job_keys),
                copy.deepcopy(self._attempts),
                copy.deepcopy(self._outbox),
                copy.deepcopy(self._workers),
                self._next_outbox_id,
            )
            try:
                yield self
            except Exception:
                (
                    self._jobs,
                    self._job_keys,
                    self._attempts,
                    self._outbox,
                    self._workers,
                    self._next_outbox_id,
                ) = snapshot
                raise

    def _job(self, job_id: UUID) -> ProcessingJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFound("processing job was not found") from exc

    def create_or_get_job(self, spec: JobSpec) -> ProcessingJob:
        spec = spec.validated()
        with self._lock:
            key = (spec.scan_id, spec.processing_version)
            existing_id = self._job_keys.get(key)
            if existing_id is not None:
                return self._job(existing_id)
            now = spec.created_at or utc_now()
            job = ProcessingJob(
                id=uuid4(),
                scan_id=spec.scan_id,
                processing_version=spec.processing_version,
                requester_user_id=spec.requester_user_id,
                member_id=spec.member_id,
                business_profile_id=spec.business_profile_id,
                status=JobStatus.ACCEPTED,
                attempt_count=0,
                max_attempts=spec.max_attempts,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                heartbeat_at=None,
                next_attempt_at=now,
                result_ref=None,
                failure_code=None,
                trace_id=spec.trace_id or str(uuid4()),
                created_at=now,
                updated_at=now,
                completed_at=None,
            )
            self._jobs[job.id] = job
            self._job_keys[key] = job.id
            return job

    def enqueue_job(
        self,
        spec: JobSpec,
        *,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EnqueuedJob:
        """Create/return a job and its dispatch event atomically in-memory."""
        spec = spec.validated()
        event_key = _normalise_reference(event_key, "event_key")
        event_type = _normalise_label(event_type, "event_type")
        payload = _safe_outbox_payload(payload)
        with self.transaction():
            existing_id = self._job_keys.get((spec.scan_id, spec.processing_version))
            job = self.create_or_get_job(spec)
            if job.status is JobStatus.ACCEPTED:
                job = self.transition_job(
                    job.id,
                    JobStatus.ACCEPTED,
                    JobStatus.QUEUED,
                )
            event_payload = dict(payload)
            if event_type == "job.process":
                event_payload.setdefault("event_key", event_key)
                event_payload["job_id"] = str(job.id)
            event_payload = _safe_outbox_payload(event_payload)
            event = None
            if job.status is JobStatus.QUEUED:
                existing = next(
                    (item for item in self._outbox.values() if item.event_key == event_key),
                    None,
                )
                if existing is not None:
                    if existing.job_id != job.id or existing.event_type != event_type:
                        raise OutboxConflict("event_key already belongs to different dispatch data")
                    event = existing
                else:
                    event = self.add_outbox_event(
                        event_key=event_key,
                        job_id=job.id,
                        event_type=event_type,
                        payload=event_payload,
                    )
            return EnqueuedJob(job=job, event=event, created=existing_id is None)

    def get_job(self, job_id: UUID) -> ProcessingJob:
        with self._lock:
            return self._job(job_id)

    def get_job_by_scan(self, scan_id: str, processing_version: str) -> ProcessingJob | None:
        scan_id = _normalise_reference(scan_id, "scan_id")
        processing_version = _normalise_processing_version(processing_version)
        with self._lock:
            job_id = self._job_keys.get((scan_id, processing_version))
            return self._jobs.get(job_id) if job_id is not None else None

    def transition_job(
        self,
        job_id: UUID,
        expected_status: JobStatus,
        target_status: JobStatus,
        *,
        now: datetime | None = None,
    ) -> ProcessingJob:
        now = now or utc_now()
        with self._lock:
            job = self._job(job_id)
            if job.status is not expected_status:
                raise InvalidJobTransition(
                    f"expected {expected_status.value}, found {job.status.value}"
                )
            _assert_transition(job.status, target_status)
            completed_at = now if target_status is JobStatus.COMPLETED else job.completed_at
            return self._replace_job(
                job,
                status=target_status,
                updated_at=now,
                completed_at=completed_at,
            )

    def _replace_job(self, job: ProcessingJob, **changes: Any) -> ProcessingJob:
        updated = replace(job, **changes)
        self._jobs[job.id] = updated
        return updated

    def claim_job(
        self,
        job_id: UUID,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> LeaseClaim | None:
        now = now or utc_now()
        worker_id = _normalise_label(worker_id, "worker_id")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        with self._lock:
            job = self._job(job_id)
            if job.status is not JobStatus.QUEUED or job.next_attempt_at > now:
                return None
            attempt_number = job.attempt_count + 1
            if attempt_number > job.max_attempts:
                raise InvalidJobTransition("maximum processing attempts exhausted")
            lease_token = uuid4()
            expires = now + timedelta(seconds=lease_seconds)
            updated = self._replace_job(
                job,
                status=JobStatus.PROCESSING,
                attempt_count=attempt_number,
                lease_owner=worker_id,
                lease_token=lease_token,
                lease_expires_at=expires,
                heartbeat_at=now,
                updated_at=now,
                failure_code=None,
            )
            attempt = AttemptRecord(
                job_id=job.id,
                attempt_number=attempt_number,
                worker_id=worker_id,
                lease_token=lease_token,
                started_at=now,
            )
            self._attempts[(job.id, attempt_number)] = attempt
            return LeaseClaim(job=updated, attempt=attempt)

    def heartbeat(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> ProcessingJob:
        now = now or utc_now()
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        with self._lock:
            job = self._job(job_id)
            _assert_lease(job, worker_id, lease_token, now)
            return self._replace_job(
                job,
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )

    def assert_lease(
        self, job_id: UUID, worker_id: str, lease_token: UUID
    ) -> ProcessingJob:
        with self._lock:
            job = self._job(job_id)
            _assert_lease(job, worker_id, lease_token, utc_now())
            return job

    def _finish_attempt(
        self,
        job: ProcessingJob,
        *,
        now: datetime,
        outcome: str,
        error_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        key = (job.id, job.attempt_count)
        attempt = self._attempts.get(key)
        if attempt is not None:
            self._attempts[key] = replace(
                attempt,
                finished_at=now,
                outcome=outcome,
                error_code=error_code,
                error_type=error_type,
            )

    def complete_job(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        result_ref: str | None = None,
        now: datetime | None = None,
    ) -> ProcessingJob:
        now = now or utc_now()
        with self._lock:
            job = self._job(job_id)
            if job.status is JobStatus.COMPLETED:
                return job
            _assert_lease(job, worker_id, lease_token, now)
            _assert_transition(job.status, JobStatus.COMPLETED)
            self._finish_attempt(job, now=now, outcome="completed")
            return self._replace_job(
                job,
                status=JobStatus.COMPLETED,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                heartbeat_at=None,
                result_ref=result_ref,
                updated_at=now,
                completed_at=now,
            )

    def schedule_retry(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        failure_code: str,
        error_type: str,
        retry_delay_seconds: int = 0,
        terminal_failure: bool = False,
        now: datetime | None = None,
    ) -> ProcessingJob:
        now = now or utc_now()
        failure_code = _normalise_label(failure_code, "failure_code")
        error_type = _normalise_label(error_type, "error_type")
        retry_delay_seconds = _validate_delay_seconds(retry_delay_seconds)
        with self._lock:
            job = self._job(job_id)
            _assert_lease(job, worker_id, lease_token, now)
            terminal = terminal_failure or job.attempt_count >= job.max_attempts
            target = JobStatus.FAILED_TERMINAL if terminal else JobStatus.FAILED_RETRYABLE
            _assert_transition(job.status, target)
            self._finish_attempt(
                job,
                now=now,
                outcome="failed_terminal" if terminal else "failed_retryable",
                error_code=failure_code,
                error_type=error_type,
            )
            updated = self._replace_job(
                job,
                status=target,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                heartbeat_at=None,
                failure_code=failure_code,
                next_attempt_at=now + timedelta(seconds=retry_delay_seconds),
                updated_at=now,
            )
            if target is JobStatus.FAILED_RETRYABLE:
                retry_event_key = f"job:{job.id}:retry:{job.attempt_count}"
                self.add_outbox_event(
                    event_key=retry_event_key,
                    job_id=job.id,
                    event_type="job.process",
                    payload={
                        "event_key": retry_event_key,
                        "processing_version": job.processing_version,
                        "trace_id": job.trace_id,
                    },
                    available_at=now,
                    available_delay_seconds=retry_delay_seconds,
                )
            return updated

    def recover_expired_leases(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]:
        now = now or utc_now()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        recovered: list[ProcessingJob] = []
        with self._lock:
            for job in list(self._jobs.values()):
                if len(recovered) >= limit:
                    break
                if (
                    job.status is not JobStatus.PROCESSING
                    or job.lease_expires_at is None
                    or job.lease_expires_at > now
                ):
                    continue
                terminal = job.attempt_count >= job.max_attempts
                target = JobStatus.FAILED_TERMINAL if terminal else JobStatus.FAILED_RETRYABLE
                self._finish_attempt(
                    job,
                    now=now,
                    outcome="lease_expired",
                    error_code="lease_expired",
                    error_type="LeaseExpired",
                )
                recovered_job = self._replace_job(
                        job,
                        status=target,
                        lease_owner=None,
                        lease_token=None,
                        lease_expires_at=None,
                        heartbeat_at=None,
                        failure_code="lease_expired",
                        next_attempt_at=now,
                        updated_at=now,
                    )
                if target is JobStatus.FAILED_RETRYABLE:
                    retry_event_key = f"job:{job.id}:retry:{job.attempt_count}"
                    self.add_outbox_event(
                        event_key=retry_event_key,
                        job_id=job.id,
                        event_type="job.process",
                        payload={
                            "event_key": retry_event_key,
                            "processing_version": job.processing_version,
                            "trace_id": job.trace_id,
                        },
                        available_at=now,
                    )
                recovered.append(recovered_job)
        return recovered

    def queue_due_retries(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]:
        now = now or utc_now()
        queued: list[ProcessingJob] = []
        with self._lock:
            for job in list(self._jobs.values()):
                if len(queued) >= limit:
                    break
                if job.status is JobStatus.FAILED_RETRYABLE and job.next_attempt_at <= now:
                    _assert_transition(job.status, JobStatus.QUEUED)
                    queued.append(self._replace_job(job, status=JobStatus.QUEUED, updated_at=now))
        return queued

    def reconcile_queued_job_events(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        min_age_seconds: int = 30,
    ) -> Sequence[OutboxEvent]:
        """Reopen delivered process events for jobs still waiting in queued."""
        now = now or utc_now()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        min_age_seconds = _validate_delay_seconds(min_age_seconds, "min_age_seconds")
        cutoff = now - timedelta(seconds=min_age_seconds)
        reopened: list[OutboxEvent] = []
        with self._lock:
            for event_id, event in sorted(self._outbox.items(), key=lambda item: item[0]):
                if len(reopened) >= limit or event.event_type != "job.process" or event.delivered_at is None:
                    continue
                if event.delivered_at > cutoff:
                    continue
                job = self._jobs.get(event.job_id)
                if job is None or job.status is not JobStatus.QUEUED:
                    continue
                updated = replace(
                    event,
                    delivered_at=None,
                    locked_by=None,
                    locked_until=None,
                    available_at=now,
                )
                self._outbox[event_id] = updated
                reopened.append(updated)
        return tuple(reopened)

    def list_attempts(self, job_id: UUID) -> Sequence[AttemptRecord]:
        with self._lock:
            self._job(job_id)
            return tuple(
                self._attempts[key]
                for key in sorted(self._attempts)
                if key[0] == job_id
            )

    def add_outbox_event(
        self,
        *,
        event_key: str,
        job_id: UUID,
        event_type: str,
        payload: Mapping[str, Any],
        available_at: datetime | None = None,
        available_delay_seconds: int = 0,
    ) -> OutboxEvent:
        event_key = _normalise_reference(event_key, "event_key")
        event_type = _normalise_label(event_type, "event_type")
        payload = _safe_outbox_payload(payload)
        available_delay_seconds = _validate_delay_seconds(
            available_delay_seconds, "available_delay_seconds"
        )
        available_at = available_at or utc_now()
        if available_delay_seconds:
            available_at = available_at + timedelta(seconds=available_delay_seconds)
        with self._lock:
            for event in self._outbox.values():
                if event.event_key == event_key:
                    if event.job_id != job_id or event.event_type != event_type or dict(event.payload) != payload:
                        raise OutboxConflict("event_key already belongs to different event data")
                    return event
            event = OutboxEvent(
                id=self._next_outbox_id,
                event_key=event_key,
                job_id=job_id,
                event_type=event_type,
                payload=payload,
                available_at=available_at,
            )
            self._outbox[event.id] = event
            self._next_outbox_id += 1
            return event

    def claim_outbox_events(
        self,
        dispatcher_id: str,
        *,
        now: datetime | None = None,
        limit: int = 50,
        lock_seconds: int = 60,
        event_type: str | None = None,
    ) -> Sequence[OutboxEvent]:
        now = now or utc_now()
        dispatcher_id = _normalise_label(dispatcher_id, "dispatcher_id")
        if event_type is not None:
            event_type = _normalise_label(event_type, "event_type")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if type(lock_seconds) is not int or lock_seconds <= 0:
            raise ValueError("lock_seconds must be a positive integer")
        claimed: list[OutboxEvent] = []
        with self._lock:
            for event_id in sorted(self._outbox):
                if len(claimed) >= limit:
                    break
                event = self._outbox[event_id]
                lock_available = event.locked_until is None or event.locked_until <= now
                if (
                    event.delivered_at is not None
                    or event.available_at > now
                    or not lock_available
                    or (event_type is not None and event.event_type != event_type)
                ):
                    continue
                if event.event_type == "job.process":
                    job = self._jobs.get(event.job_id)
                    if job is None or job.status is not JobStatus.QUEUED:
                        continue
                updated = replace(
                    event,
                    locked_by=dispatcher_id,
                    locked_until=now + timedelta(seconds=lock_seconds),
                )
                self._outbox[event_id] = updated
                claimed.append(updated)
        return tuple(claimed)

    def mark_outbox_delivered(
        self, event_id: int, dispatcher_id: str, *, now: datetime | None = None
    ) -> OutboxEvent:
        now = now or utc_now()
        with self._lock:
            try:
                event = self._outbox[event_id]
            except KeyError as exc:
                raise OutboxConflict("outbox event was not found") from exc
            if event.delivered_at is not None:
                return event
            if event.locked_by != dispatcher_id or event.locked_until is None or event.locked_until <= now:
                raise OutboxConflict("outbox event is not owned by dispatcher")
            updated = replace(event, delivered_at=now, locked_by=None, locked_until=None)
            self._outbox[event_id] = updated
            return updated

    def register_worker(
        self,
        worker_id: str,
        worker_version: str,
        *,
        capacity: int = 1,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or utc_now()
        worker_id = _normalise_label(worker_id, "worker_id")
        worker_version = _normalise_label(worker_version, "worker_version")
        capacity = _validate_capacity(capacity)
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        worker = WorkerLease(
            worker_id=worker_id,
            lease_token=uuid4(),
            capacity=capacity,
            active_slots=0,
            heartbeat_at=now,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            draining=False,
            worker_version=worker_version,
        )
        with self._lock:
            self._workers[worker_id] = worker
        return worker

    def heartbeat_worker(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        active_slots: int,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or utc_now()
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or worker.lease_token != lease_token or worker.lease_expires_at <= now:
                raise LeaseConflict("worker liveness lease is missing, expired, or owned by another worker")
            active_slots = _validate_active_slots(active_slots, worker.capacity)
            updated = replace(
                worker,
                active_slots=active_slots,
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._workers[worker_id] = updated
            return updated

    def set_worker_draining(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        draining: bool = True,
        now: datetime | None = None,
    ) -> WorkerLease:
        now = now or utc_now()
        if type(draining) is not bool:
            raise ValueError("draining must be a boolean")
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or worker.lease_token != lease_token or worker.lease_expires_at <= now:
                raise LeaseConflict("worker liveness lease is missing, expired, or owned by another worker")
            updated = replace(worker, draining=draining)
            self._workers[worker_id] = updated
            return updated

    def reap_expired_workers(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[WorkerLease]:
        now = now or utc_now()
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        expired: list[WorkerLease] = []
        with self._lock:
            for worker_id, worker in list(self._workers.items()):
                if len(expired) >= limit:
                    break
                if worker.lease_expires_at <= now:
                    expired.append(worker)
                    del self._workers[worker_id]
        return tuple(expired)

    def release_outbox_event(
        self,
        event_id: int,
        dispatcher_id: str,
        *,
        failure_code: str,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> OutboxEvent:
        now = now or utc_now()
        failure_code = _normalise_label(failure_code, "failure_code")
        retry_delay_seconds = _validate_delay_seconds(retry_delay_seconds)
        with self._lock:
            try:
                event = self._outbox[event_id]
            except KeyError as exc:
                raise OutboxConflict("outbox event was not found") from exc
            if (
                event.locked_by != dispatcher_id
                or event.locked_until is None
                or event.locked_until <= now
            ):
                raise OutboxConflict("outbox event is not owned by dispatcher")
            updated = replace(
                event,
                attempt_count=event.attempt_count + 1,
                locked_by=None,
                locked_until=None,
                available_at=now + timedelta(seconds=retry_delay_seconds),
                last_error_code=failure_code,
            )
            self._outbox[event_id] = updated
            return updated


class PostgresProcessingRepository:
    """PostgreSQL implementation of the Phase A control-plane repository.

    ``connection_factory`` must return a DB-API connection.  This repository
    explicitly disables autocommit and starts each transaction itself.  No
    method in this class reads or writes Directus business tables.
    """

    _JOB_COLUMNS = (
        "id, scan_id, processing_version, requester_user_id, member_id, "
        "business_profile_id, status, attempt_count, max_attempts, lease_owner, "
        "lease_token, lease_expires_at, heartbeat_at, next_attempt_at, result_ref, "
        "failure_code, trace_id, created_at, updated_at, completed_at"
    )
    _WORKER_COLUMNS = (
        "worker_id, lease_token, capacity, active_slots, heartbeat_at, "
        "lease_expires_at, draining, worker_version"
    )

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory

    @_retry_transaction_method
    def ping(self) -> None:
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise DatabaseOperationError("processing database ping failed")

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = None
        try:
            connection = self._connection_factory()
            try:
                connection.autocommit = False
            except Exception as exc:
                raise DatabaseOperationError("processing connection cannot disable autocommit") from exc
            if getattr(connection, "autocommit", None) is not False:
                raise DatabaseOperationError("processing connection autocommit must be disabled")
            begin_cursor = connection.cursor()
            try:
                begin_cursor.execute("BEGIN")
            finally:
                close = getattr(begin_cursor, "close", None)
                if close is not None:
                    close()
            yield connection
            connection.commit()
        except ProcessingError:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise DatabaseOperationError(
                "processing database transaction failed",
                sqlstate=_sqlstate_from_exception(exc),
            ) from exc
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    @staticmethod
    def _row_dict(cursor: Any, row: Any) -> dict[str, Any]:
        if isinstance(row, Mapping):
            return dict(row)
        names = [item[0] for item in cursor.description]
        return dict(zip(names, row))

    @classmethod
    def _job_from_row(cls, cursor: Any, row: Any) -> ProcessingJob:
        data = cls._row_dict(cursor, row)
        status = JobStatus(data["status"])
        return ProcessingJob(
            id=data["id"] if isinstance(data["id"], UUID) else UUID(str(data["id"])),
            scan_id=data["scan_id"],
            processing_version=data["processing_version"],
            requester_user_id=data["requester_user_id"],
            member_id=data["member_id"],
            business_profile_id=data["business_profile_id"],
            status=status,
            attempt_count=data["attempt_count"],
            max_attempts=data["max_attempts"],
            lease_owner=data["lease_owner"],
            lease_token=(
                data["lease_token"]
                if data["lease_token"] is None or isinstance(data["lease_token"], UUID)
                else UUID(str(data["lease_token"]))
            ),
            lease_expires_at=data["lease_expires_at"],
            heartbeat_at=data["heartbeat_at"],
            next_attempt_at=data["next_attempt_at"],
            result_ref=data["result_ref"],
            failure_code=data["failure_code"],
            trace_id=data["trace_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            completed_at=data["completed_at"],
        )

    @classmethod
    def _worker_from_row(cls, cursor: Any, row: Any) -> WorkerLease:
        data = cls._row_dict(cursor, row)
        token = data["lease_token"]
        return WorkerLease(
            worker_id=data["worker_id"],
            lease_token=token if isinstance(token, UUID) else UUID(str(token)),
            capacity=data["capacity"],
            active_slots=data["active_slots"],
            heartbeat_at=data["heartbeat_at"],
            lease_expires_at=data["lease_expires_at"],
            draining=data["draining"],
            worker_version=data["worker_version"],
        )

    def _fetch_job(self, cursor: Any, job_id: UUID, *, for_update: bool = False) -> ProcessingJob:
        suffix = " FOR UPDATE" if for_update else ""
        cursor.execute(
            f"SELECT {self._JOB_COLUMNS} FROM ai_processing.ai_processing_jobs WHERE id = %s{suffix}",
            (job_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise JobNotFound("processing job was not found")
        return self._job_from_row(cursor, row)

    def _fetch_job_by_key(
        self,
        cursor: Any,
        scan_id: str,
        processing_version: str,
        *,
        for_update: bool = False,
    ) -> ProcessingJob:
        suffix = " FOR UPDATE" if for_update else ""
        cursor.execute(
            f"""
            SELECT {self._JOB_COLUMNS}
            FROM ai_processing.ai_processing_jobs
            WHERE scan_id = %s AND processing_version = %s{suffix}
            """,
            (scan_id, processing_version),
        )
        row = cursor.fetchone()
        if row is None:
            raise JobNotFound("processing job was not found")
        return self._job_from_row(cursor, row)

    @_retry_transaction_method
    def create_or_get_job(self, spec: JobSpec) -> ProcessingJob:
        spec = spec.validated()
        job_id = uuid4()
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                INSERT INTO ai_processing.ai_processing_jobs
                    (id, scan_id, processing_version, requester_user_id, member_id,
                     business_profile_id, status, attempt_count, max_attempts,
                     trace_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 0, %s, %s)
                ON CONFLICT (scan_id, processing_version) DO NOTHING
                RETURNING {self._JOB_COLUMNS}
                """,
                (
                    job_id,
                    spec.scan_id,
                    spec.processing_version,
                    spec.requester_user_id,
                    spec.member_id,
                    spec.business_profile_id,
                    JobStatus.ACCEPTED.value,
                    spec.max_attempts,
                    spec.trace_id,
                ),
            )
            row = cursor.fetchone()
            if row is not None:
                return self._job_from_row(cursor, row)
            cursor.execute(
                f"SELECT {self._JOB_COLUMNS} FROM ai_processing.ai_processing_jobs "
                "WHERE scan_id = %s AND processing_version = %s",
                (spec.scan_id, spec.processing_version),
            )
            row = cursor.fetchone()
            created = row is not None
            if row is None:
                raise DatabaseOperationError("job disappeared after idempotent insert")
            return self._job_from_row(cursor, row)

    @_retry_transaction_method
    def enqueue_job(
        self,
        spec: JobSpec,
        *,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> EnqueuedJob:
        """Atomically create/queue a job and create its dispatch outbox row."""
        spec = spec.validated()
        event_key = _normalise_reference(event_key, "event_key")
        event_type = _normalise_label(event_type, "event_type")
        payload = _safe_outbox_payload(payload)
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                INSERT INTO ai_processing.ai_processing_jobs
                    (id, scan_id, processing_version, requester_user_id, member_id,
                     business_profile_id, status, attempt_count, max_attempts,
                     trace_id)
                VALUES (%s, %s, %s, %s, %s, %s, 'accepted', 0, %s, %s)
                ON CONFLICT (scan_id, processing_version) DO NOTHING
                RETURNING {self._JOB_COLUMNS}
                """,
                (
                    uuid4(),
                    spec.scan_id,
                    spec.processing_version,
                    spec.requester_user_id,
                    spec.member_id,
                    spec.business_profile_id,
                    spec.max_attempts,
                    spec.trace_id,
                ),
            )
            row = cursor.fetchone()
            created = row is not None
            if row is None:
                job = self._fetch_job_by_key(
                    cursor,
                    spec.scan_id,
                    spec.processing_version,
                    for_update=True,
                )
            else:
                job = self._job_from_row(cursor, row)

            if job.status is JobStatus.ACCEPTED:
                cursor.execute(
                    """
                    UPDATE ai_processing.ai_processing_jobs
                    SET status = 'queued', updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = 'accepted'
                    RETURNING id
                    """,
                    (job.id,),
                )
                if cursor.fetchone() is None:
                    raise InvalidJobTransition("job enqueue transition compare-and-set failed")
                job = self._fetch_job(cursor, job.id)

            event = None
            if job.status is JobStatus.QUEUED:
                event_payload = dict(payload)
                if event_type == "job.process":
                    event_payload.setdefault("event_key", event_key)
                    event_payload["job_id"] = str(job.id)
                event_payload = _safe_outbox_payload(event_payload)
                payload_json = json.dumps(
                    event_payload,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                cursor.execute(
                    """
                    INSERT INTO ai_processing.ai_processing_outbox
                        (event_key, job_id, event_type, payload, available_at)
                    VALUES (%s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                    ON CONFLICT (event_key) DO NOTHING
                    RETURNING id, event_key, job_id, event_type, payload,
                              available_at, attempt_count, locked_by, locked_until,
                              delivered_at, last_error_code
                    """,
                    (event_key, job.id, event_type, payload_json),
                )
                event_row = cursor.fetchone()
                if event_row is not None:
                    event = self._outbox_from_row(cursor, event_row)
                else:
                    cursor.execute(
                        """
                        SELECT id, event_key, job_id, event_type, payload,
                               available_at, attempt_count, locked_by, locked_until,
                               delivered_at, last_error_code
                        FROM ai_processing.ai_processing_outbox
                        WHERE event_key = %s
                        """,
                        (event_key,),
                    )
                    event_row = cursor.fetchone()
                    if event_row is None:
                        raise DatabaseOperationError("dispatch event disappeared after idempotent insert")
                    event = self._outbox_from_row(cursor, event_row)
                    if (
                        event.job_id != job.id
                        or event.event_type != event_type
                        
                    ):
                        raise OutboxConflict("event_key already belongs to different dispatch data")
            return EnqueuedJob(job=job, event=event, created=created)

    @_retry_transaction_method
    def get_job(self, job_id: UUID) -> ProcessingJob:
        with self._transaction() as connection:
            cursor = connection.cursor()
            return self._fetch_job(cursor, job_id)

    @_retry_transaction_method
    def get_job_by_scan(self, scan_id: str, processing_version: str) -> ProcessingJob | None:
        scan_id = _normalise_reference(scan_id, "scan_id")
        processing_version = _normalise_processing_version(processing_version)
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {self._JOB_COLUMNS}
                FROM ai_processing.ai_processing_jobs
                WHERE scan_id = %s AND processing_version = %s
                """,
                (scan_id, processing_version),
            )
            row = cursor.fetchone()
            return self._job_from_row(cursor, row) if row is not None else None

    @_retry_transaction_method
    def transition_job(
        self,
        job_id: UUID,
        expected_status: JobStatus,
        target_status: JobStatus,
        *,
        now: datetime | None = None,
    ) -> ProcessingJob:
        with self._transaction() as connection:
            cursor = connection.cursor()
            job = self._fetch_job(cursor, job_id, for_update=True)
            if job.status is not expected_status:
                raise InvalidJobTransition(
                    f"expected {expected_status.value}, found {job.status.value}"
                )
            _assert_transition(job.status, target_status)
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_jobs
                SET status = %s, updated_at = CURRENT_TIMESTAMP,
                    completed_at = CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE id = %s AND status = %s
                RETURNING id
                """,
                (
                    target_status.value,
                    target_status.value,
                    job_id,
                    expected_status.value,
                ),
            )
            if cursor.fetchone() is None:
                raise InvalidJobTransition("job transition compare-and-set failed")
            return self._fetch_job(cursor, job_id)

    @_retry_transaction_method
    def claim_job(
        self,
        job_id: UUID,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> LeaseClaim | None:
        worker_id = _normalise_label(worker_id, "worker_id")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        lease_token = uuid4()
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {self._JOB_COLUMNS}
                FROM ai_processing.ai_processing_jobs
                WHERE id = %s
                  AND status = %s
                  AND next_attempt_at <= CURRENT_TIMESTAMP
                  AND (lease_expires_at IS NULL OR lease_expires_at <= CURRENT_TIMESTAMP)
                FOR UPDATE
                """,
                (job_id, JobStatus.QUEUED.value),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            job = self._job_from_row(cursor, row)
            attempt_number = job.attempt_count + 1
            if attempt_number > job.max_attempts:
                raise InvalidJobTransition("maximum processing attempts exhausted")
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_jobs
                SET status = 'processing', attempt_count = %s,
                    lease_owner = %s, lease_token = %s,
                    lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    heartbeat_at = CURRENT_TIMESTAMP,
                    failure_code = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'queued'
                RETURNING id
                """,
                (
                    attempt_number,
                    worker_id,
                    lease_token,
                    lease_seconds,
                    job_id,
                ),
            )
            if cursor.fetchone() is None:
                raise LeaseConflict("job claim compare-and-set failed")
            cursor.execute(
                """
                INSERT INTO ai_processing.ai_processing_attempts
                    (id, job_id, attempt_number, worker_id, lease_token, started_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (uuid4(), job_id, attempt_number, worker_id, lease_token),
            )
            updated = self._fetch_job(cursor, job_id)
            attempt = AttemptRecord(
                job_id,
                attempt_number,
                worker_id,
                lease_token,
                updated.heartbeat_at or updated.created_at,
            )
            return LeaseClaim(updated, attempt)

    @_retry_transaction_method
    def heartbeat(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> ProcessingJob:
        worker_id = _normalise_label(worker_id, "worker_id")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                UPDATE ai_processing.ai_processing_jobs
                SET heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'processing'
                  AND lease_owner = %s AND lease_token = %s
                  AND lease_expires_at > CURRENT_TIMESTAMP
                RETURNING {self._JOB_COLUMNS}
                """,
                (
                    lease_seconds,
                    job_id,
                    worker_id,
                    lease_token,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise LeaseConflict("worker heartbeat compare-and-set failed")
            return self._job_from_row(cursor, row)

    @_retry_transaction_method
    def assert_lease(
        self, job_id: UUID, worker_id: str, lease_token: UUID
    ) -> ProcessingJob:
        worker_id = _normalise_label(worker_id, "worker_id")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {self._JOB_COLUMNS}
                FROM ai_processing.ai_processing_jobs
                WHERE id = %s AND status = 'processing'
                  AND lease_owner = %s AND lease_token = %s
                  AND lease_expires_at > CURRENT_TIMESTAMP
                """,
                (job_id, worker_id, lease_token),
            )
            row = cursor.fetchone()
            if row is None:
                raise LeaseConflict("active processing lease was not found")
            return self._job_from_row(cursor, row)

    @_retry_transaction_method
    def complete_job(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        result_ref: str | None = None,
        now: datetime | None = None,
    ) -> ProcessingJob:
        with self._transaction() as connection:
            cursor = connection.cursor()
            job = self._fetch_job(cursor, job_id, for_update=True)
            if job.status is JobStatus.COMPLETED:
                return job
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_jobs
                SET status = 'completed', result_ref = %s,
                    lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'processing'
                  AND lease_owner = %s AND lease_token = %s
                  AND lease_expires_at > CURRENT_TIMESTAMP
                RETURNING id
                """,
                (result_ref, job_id, worker_id, lease_token),
            )
            if cursor.fetchone() is None:
                raise LeaseConflict("completion compare-and-set failed")
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_attempts
                SET finished_at = CURRENT_TIMESTAMP, outcome = 'completed'
                WHERE job_id = %s AND attempt_number = %s AND lease_token = %s
                """,
                (job_id, job.attempt_count, lease_token),
            )
            if cursor.rowcount != 1:
                raise DatabaseOperationError("processing attempt could not be completed")
            return self._fetch_job(cursor, job_id)

    @_retry_transaction_method
    def schedule_retry(
        self,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        *,
        failure_code: str,
        error_type: str,
        retry_delay_seconds: int = 0,
        terminal_failure: bool = False,
        now: datetime | None = None,
    ) -> ProcessingJob:
        failure_code = _normalise_label(failure_code, "failure_code")
        error_type = _normalise_label(error_type, "error_type")
        retry_delay_seconds = _validate_delay_seconds(retry_delay_seconds)
        with self._transaction() as connection:
            cursor = connection.cursor()
            job = self._fetch_job(cursor, job_id, for_update=True)
            target = (
                JobStatus.FAILED_TERMINAL
                if terminal_failure or job.attempt_count >= job.max_attempts
                else JobStatus.FAILED_RETRYABLE
            )
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_jobs
                SET status = %s, failure_code = %s,
                    next_attempt_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    lease_owner = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'processing'
                  AND lease_owner = %s AND lease_token = %s
                  AND lease_expires_at > CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    target.value,
                    failure_code,
                    retry_delay_seconds,
                    job_id,
                    worker_id,
                    lease_token,
                ),
            )
            if cursor.fetchone() is None:
                raise LeaseConflict("retry compare-and-set failed")
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_attempts
                SET finished_at = CURRENT_TIMESTAMP, outcome = %s,
                    error_code = %s, error_type = %s
                WHERE job_id = %s AND attempt_number = %s AND lease_token = %s
                """,
                (
                    target.value,
                    failure_code,
                    error_type,
                    job_id,
                    job.attempt_count,
                    lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise DatabaseOperationError("processing attempt could not be closed for retry")
            if target is JobStatus.FAILED_RETRYABLE:
                retry_event_key = f"job:{job.id}:retry:{job.attempt_count}"
                retry_payload = json.dumps(
                    {
                        "event_key": retry_event_key,
                        "job_id": str(job.id),
                        "processing_version": job.processing_version,
                        "trace_id": job.trace_id,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                cursor.execute(
                    """
                    INSERT INTO ai_processing.ai_processing_outbox
                        (event_key, job_id, event_type, payload, available_at)
                    VALUES (%s, %s, 'job.process', %s::jsonb,
                            CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'))
                    ON CONFLICT (event_key) DO NOTHING
                    """,
                    (retry_event_key, job.id, retry_payload, retry_delay_seconds),
                )
            return self._fetch_job(cursor, job_id)

    @_retry_transaction_method
    def recover_expired_leases(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        recovered: list[ProcessingJob] = []
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT {self._JOB_COLUMNS}
                FROM ai_processing.ai_processing_jobs
                WHERE status = 'processing' AND lease_expires_at <= CURRENT_TIMESTAMP
                ORDER BY lease_expires_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            for row in rows:
                job = self._job_from_row(cursor, row)
                target = (
                    JobStatus.FAILED_TERMINAL
                    if job.attempt_count >= job.max_attempts
                    else JobStatus.FAILED_RETRYABLE
                )
                cursor.execute(
                    """
                    UPDATE ai_processing.ai_processing_jobs
                    SET status = %s, failure_code = 'lease_expired',
                        next_attempt_at = CURRENT_TIMESTAMP, lease_owner = NULL,
                        lease_token = NULL, lease_expires_at = NULL,
                        heartbeat_at = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = 'processing'
                    """,
                    (target.value, job.id),
                )
                cursor.execute(
                    """
                    UPDATE ai_processing.ai_processing_attempts
                    SET finished_at = CURRENT_TIMESTAMP, outcome = 'lease_expired',
                        error_code = 'lease_expired', error_type = 'LeaseExpired'
                    WHERE job_id = %s AND attempt_number = %s AND lease_token = %s
                    """,
                    (job.id, job.attempt_count, job.lease_token),
                )
                if cursor.rowcount != 1:
                    raise DatabaseOperationError("expired processing attempt could not be closed")
                if target is JobStatus.FAILED_RETRYABLE:
                    retry_event_key = f"job:{job.id}:retry:{job.attempt_count}"
                    retry_payload = json.dumps(
                        {
                            "event_key": retry_event_key,
                            "job_id": str(job.id),
                            "processing_version": job.processing_version,
                            "trace_id": job.trace_id,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    cursor.execute(
                        """
                        INSERT INTO ai_processing.ai_processing_outbox
                            (event_key, job_id, event_type, payload, available_at)
                        VALUES (%s, %s, 'job.process', %s::jsonb, CURRENT_TIMESTAMP)
                        ON CONFLICT (event_key) DO NOTHING
                        """,
                        (retry_event_key, job.id, retry_payload),
                    )
                recovered.append(self._fetch_job(cursor, job.id))
        return tuple(recovered)

    @_retry_transaction_method
    def queue_due_retries(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[ProcessingJob]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        qualified_job_columns = ", ".join(
            f"jobs.{column.strip()}" for column in self._JOB_COLUMNS.split(",")
        )
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                WITH candidates AS (
                    SELECT id
                    FROM ai_processing.ai_processing_jobs
                    WHERE status = 'failed_retryable' AND next_attempt_at <= CURRENT_TIMESTAMP
                    ORDER BY next_attempt_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                UPDATE ai_processing.ai_processing_jobs AS jobs
                SET status = 'queued', updated_at = CURRENT_TIMESTAMP
                FROM candidates
                WHERE jobs.id = candidates.id
                RETURNING {qualified_job_columns}
                """,
                (limit,),
            )
            return tuple(self._job_from_row(cursor, row) for row in cursor.fetchall())

    @_retry_transaction_method
    def reconcile_queued_job_events(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        min_age_seconds: int = 30,
    ) -> Sequence[OutboxEvent]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        min_age_seconds = _validate_delay_seconds(min_age_seconds, "min_age_seconds")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                WITH candidates AS (
                    SELECT events.id
                    FROM ai_processing.ai_processing_outbox AS events
                    JOIN ai_processing.ai_processing_jobs AS jobs
                      ON jobs.id = events.job_id
                    WHERE events.event_type = 'job.process'
                      AND events.delivered_at IS NOT NULL
                      AND events.delivered_at <= CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                      AND jobs.status = 'queued'
                    ORDER BY events.delivered_at, events.id
                    FOR UPDATE OF events SKIP LOCKED
                    LIMIT %s
                )
                UPDATE ai_processing.ai_processing_outbox AS events
                SET delivered_at = NULL,
                    locked_by = NULL,
                    locked_until = NULL,
                    available_at = CURRENT_TIMESTAMP
                FROM candidates
                WHERE events.id = candidates.id
                RETURNING events.id, events.event_key, events.job_id,
                          events.event_type, events.payload, events.available_at,
                          events.attempt_count, events.locked_by,
                          events.locked_until, events.delivered_at,
                          events.last_error_code
                """,
                (min_age_seconds, limit),
            )
            return tuple(self._outbox_from_row(cursor, row) for row in cursor.fetchall())

    @_retry_transaction_method
    def list_attempts(self, job_id: UUID) -> Sequence[AttemptRecord]:
        with self._transaction() as connection:
            cursor = connection.cursor()
            self._fetch_job(cursor, job_id)
            cursor.execute(
                """
                SELECT job_id, attempt_number, worker_id, lease_token,
                       started_at, finished_at, outcome, error_code, error_type
                FROM ai_processing.ai_processing_attempts
                WHERE job_id = %s
                ORDER BY attempt_number
                """,
                (job_id,),
            )
            return tuple(
                AttemptRecord(
                    job_id=(row[0] if isinstance(row[0], UUID) else UUID(str(row[0]))),
                    attempt_number=row[1],
                    worker_id=row[2],
                    lease_token=(row[3] if isinstance(row[3], UUID) else UUID(str(row[3]))),
                    started_at=row[4],
                    finished_at=row[5],
                    outcome=row[6],
                    error_code=row[7],
                    error_type=row[8],
                )
                for row in cursor.fetchall()
            )

    @staticmethod
    def _outbox_from_row(cursor: Any, row: Any) -> OutboxEvent:
        data = PostgresProcessingRepository._row_dict(cursor, row)
        payload = data["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        job_id = data["job_id"] if isinstance(data["job_id"], UUID) else UUID(str(data["job_id"]))
        return OutboxEvent(
            id=data["id"],
            event_key=data["event_key"],
            job_id=job_id,
            event_type=data["event_type"],
            payload=payload,
            available_at=data["available_at"],
            attempt_count=data["attempt_count"],
            locked_by=data["locked_by"],
            locked_until=data["locked_until"],
            delivered_at=data["delivered_at"],
            last_error_code=data["last_error_code"],
        )

    @_retry_transaction_method
    def add_outbox_event(
        self,
        *,
        event_key: str,
        job_id: UUID,
        event_type: str,
        payload: Mapping[str, Any],
        available_at: datetime | None = None,
        available_delay_seconds: int = 0,
    ) -> OutboxEvent:
        event_key = _normalise_reference(event_key, "event_key")
        event_type = _normalise_label(event_type, "event_type")
        payload = _safe_outbox_payload(payload)
        available_delay_seconds = _validate_delay_seconds(
            available_delay_seconds, "available_delay_seconds"
        )
        if available_at is not None:
            raise ValueError("PostgreSQL outbox availability must use a delay, not application time")
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO ai_processing.ai_processing_outbox
                    (event_key, job_id, event_type, payload, available_at)
                VALUES (%s, %s, %s, %s::jsonb,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'))
                ON CONFLICT (event_key) DO NOTHING
                RETURNING id, event_key, job_id, event_type, payload,
                          available_at, attempt_count, locked_by, locked_until,
                          delivered_at, last_error_code
                """,
                (event_key, job_id, event_type, payload_json, available_delay_seconds),
            )
            row = cursor.fetchone()
            if row is not None:
                return self._outbox_from_row(cursor, row)
            cursor.execute(
                """
                SELECT id, event_key, job_id, event_type, payload,
                       available_at, attempt_count, locked_by, locked_until,
                       delivered_at, last_error_code
                FROM ai_processing.ai_processing_outbox
                WHERE event_key = %s
                """,
                (event_key,),
            )
            row = cursor.fetchone()
            if row is None:
                raise DatabaseOperationError("outbox event disappeared after idempotent insert")
            event = self._outbox_from_row(cursor, row)
            if event.job_id != job_id or event.event_type != event_type or dict(event.payload) != payload:
                raise OutboxConflict("event_key already belongs to different event data")
            return event

    @_retry_transaction_method
    def claim_outbox_events(
        self,
        dispatcher_id: str,
        *,
        now: datetime | None = None,
        limit: int = 50,
        lock_seconds: int = 60,
        event_type: str | None = None,
    ) -> Sequence[OutboxEvent]:
        dispatcher_id = _normalise_label(dispatcher_id, "dispatcher_id")
        if event_type is not None:
            event_type = _normalise_label(event_type, "event_type")
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if type(lock_seconds) is not int or lock_seconds <= 0:
            raise ValueError("lock_seconds must be a positive integer")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                WITH candidates AS (
                    SELECT source.id
                    FROM ai_processing.ai_processing_outbox AS source
                    WHERE source.delivered_at IS NULL
                      AND source.available_at <= CURRENT_TIMESTAMP
                      AND (source.locked_until IS NULL OR source.locked_until <= CURRENT_TIMESTAMP)
                      AND (%s::text IS NULL OR source.event_type = %s::text)
                      AND (
                          source.event_type <> 'job.process'
                          OR EXISTS (
                              SELECT 1
                              FROM ai_processing.ai_processing_jobs AS jobs
                              WHERE jobs.id = source.job_id
                                AND jobs.status = 'queued'
                          )
                      )
                    ORDER BY source.available_at, source.id
                    FOR UPDATE OF source SKIP LOCKED
                    LIMIT %s
                )
                UPDATE ai_processing.ai_processing_outbox AS events
                SET locked_by = %s,
                    locked_until = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                FROM candidates
                WHERE events.id = candidates.id
                RETURNING events.id, events.event_key, events.job_id,
                          events.event_type, events.payload, events.available_at,
                          events.attempt_count, events.locked_by,
                          events.locked_until, events.delivered_at,
                          events.last_error_code
                """,
                (event_type, event_type, limit, dispatcher_id, lock_seconds),
            )
            return tuple(self._outbox_from_row(cursor, row) for row in cursor.fetchall())

    @_retry_transaction_method
    def mark_outbox_delivered(
        self, event_id: int, dispatcher_id: str, *, now: datetime | None = None
    ) -> OutboxEvent:
        dispatcher_id = _normalise_label(dispatcher_id, "dispatcher_id")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_outbox
                SET delivered_at = CURRENT_TIMESTAMP, locked_by = NULL, locked_until = NULL
                WHERE id = %s AND locked_by = %s
                  AND delivered_at IS NULL AND locked_until > CURRENT_TIMESTAMP
                RETURNING id, event_key, job_id, event_type, payload,
                          available_at, attempt_count, locked_by, locked_until,
                          delivered_at, last_error_code
                """,
                (event_id, dispatcher_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise OutboxConflict("outbox delivery compare-and-set failed")
            return self._outbox_from_row(cursor, row)

    @_retry_transaction_method
    def release_outbox_event(
        self,
        event_id: int,
        dispatcher_id: str,
        *,
        failure_code: str,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> OutboxEvent:
        dispatcher_id = _normalise_label(dispatcher_id, "dispatcher_id")
        failure_code = _normalise_label(failure_code, "failure_code")
        retry_delay_seconds = _validate_delay_seconds(retry_delay_seconds)
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE ai_processing.ai_processing_outbox
                SET attempt_count = attempt_count + 1,
                    locked_by = NULL, locked_until = NULL,
                    available_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    last_error_code = %s
                WHERE id = %s AND locked_by = %s
                  AND delivered_at IS NULL AND locked_until > CURRENT_TIMESTAMP
                RETURNING id, event_key, job_id, event_type, payload,
                          available_at, attempt_count, locked_by, locked_until,
                          delivered_at, last_error_code
                """,
                (retry_delay_seconds, failure_code, event_id, dispatcher_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise OutboxConflict("outbox release compare-and-set failed")
            return self._outbox_from_row(cursor, row)

    @_retry_transaction_method
    def register_worker(
        self,
        worker_id: str,
        worker_version: str,
        *,
        capacity: int = 1,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease:
        worker_id = _normalise_label(worker_id, "worker_id")
        worker_version = _normalise_label(worker_version, "worker_version")
        capacity = _validate_capacity(capacity)
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        token = uuid4()
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                INSERT INTO ai_processing.ai_worker_leases
                    (worker_id, lease_token, capacity, active_slots,
                     heartbeat_at, lease_expires_at, draining, worker_version)
                VALUES (%s, %s, %s, 0, CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'), FALSE, %s)
                ON CONFLICT (worker_id) DO UPDATE
                SET lease_token = EXCLUDED.lease_token,
                    capacity = EXCLUDED.capacity,
                    active_slots = 0,
                    heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    draining = FALSE,
                    worker_version = EXCLUDED.worker_version
                RETURNING {self._WORKER_COLUMNS}
                """,
                (worker_id, token, capacity, lease_seconds, worker_version, lease_seconds),
            )
            row = cursor.fetchone()
            if row is None:
                raise DatabaseOperationError("worker registration did not return a lease")
            return self._worker_from_row(cursor, row)

    @_retry_transaction_method
    def heartbeat_worker(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        active_slots: int,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> WorkerLease:
        worker_id = _normalise_label(worker_id, "worker_id")
        if type(active_slots) is not int or not 0 <= active_slots <= 10_000:
            raise ValueError("active_slots must be an integer from 0 to 10000")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                UPDATE ai_processing.ai_worker_leases
                SET active_slots = %s,
                    heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                WHERE worker_id = %s AND lease_token = %s
                  AND %s <= capacity
                  AND lease_expires_at > CURRENT_TIMESTAMP
                RETURNING {self._WORKER_COLUMNS}
                """,
                (active_slots, lease_seconds, worker_id, lease_token, active_slots),
            )
            row = cursor.fetchone()
            if row is None:
                raise LeaseConflict("worker liveness heartbeat compare-and-set failed")
            return self._worker_from_row(cursor, row)

    @_retry_transaction_method
    def set_worker_draining(
        self,
        worker_id: str,
        lease_token: UUID,
        *,
        draining: bool = True,
        now: datetime | None = None,
    ) -> WorkerLease:
        worker_id = _normalise_label(worker_id, "worker_id")
        if type(draining) is not bool:
            raise ValueError("draining must be a boolean")
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                UPDATE ai_processing.ai_worker_leases
                SET draining = %s
                WHERE worker_id = %s AND lease_token = %s
                  AND lease_expires_at > CURRENT_TIMESTAMP
                RETURNING {self._WORKER_COLUMNS}
                """,
                (draining, worker_id, lease_token),
            )
            row = cursor.fetchone()
            if row is None:
                raise LeaseConflict("worker draining update compare-and-set failed")
            return self._worker_from_row(cursor, row)

    @_retry_transaction_method
    def reap_expired_workers(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> Sequence[WorkerLease]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        qualified_worker_columns = ", ".join(
            f"workers.{column.strip()}" for column in self._WORKER_COLUMNS.split(",")
        )
        with self._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""
                WITH candidates AS (
                    SELECT worker_id
                    FROM ai_processing.ai_worker_leases
                    WHERE lease_expires_at <= CURRENT_TIMESTAMP
                    ORDER BY lease_expires_at, worker_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                DELETE FROM ai_processing.ai_worker_leases AS workers
                USING candidates
                WHERE workers.worker_id = candidates.worker_id
                RETURNING {qualified_worker_columns}
                """,
                (limit,),
            )
            return tuple(self._worker_from_row(cursor, row) for row in cursor.fetchall())
