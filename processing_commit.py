"""Phase A contract for the future atomic result commit authority."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from uuid import UUID

from processing import PostgresProcessingRepository, _retry_transaction_method


ALLOWED_RISK_LEVELS = frozenset(
    {"stable", "low_focus", "elevated_fatigue", "high_risk", "unknown"}
)
REQUIRED_RESULT_FIELDS = frozenset(
    {"readiness_score", "confidence", "risk_level", "explanation", "suggested_action"}
)
NORMALIZED_RESULT_FIELDS = REQUIRED_RESULT_FIELDS


@dataclass(frozen=True)
class ValidatedResultPayload:
    """A result already validated before entering the final commit boundary."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        required = REQUIRED_RESULT_FIELDS
        missing = required - set(self.values)
        if missing:
            raise ValueError("validated result is missing required fields")
        score = self.values["readiness_score"]
        confidence = self.values["confidence"]
        risk_level = self.values["risk_level"]
        if type(score) is not int or not 0 <= score <= 100:
            raise ValueError("readiness_score must be an integer from 0 to 100")
        if type(confidence) not in (int, float) or not math.isfinite(float(confidence)):
            raise ValueError("confidence must be finite")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(risk_level, str) or risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError("risk_level is not an allowed product value")
        for field_name in ("explanation", "suggested_action"):
            value = self.values[field_name]
            if not isinstance(value, str) or not value.strip() or len(value) > 65_535:
                raise ValueError(f"{field_name} must be a non-empty bounded string")

    def normalized(self) -> dict[str, Any]:
        return {field_name: self.values[field_name] for field_name in NORMALIZED_RESULT_FIELDS}

    def result_hash(self) -> str:
        encoded = json.dumps(
            self.normalized(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CommitProcessingResultRequest:
    job_id: UUID
    scan_id: str
    processing_version: str
    lease_token: UUID
    result_id: UUID
    result: ValidatedResultPayload


@dataclass(frozen=True)
class CommitProcessingResultResponse:
    status: str
    result_ref: str | None = None


class ProcessingResultCommitter(Protocol):
    """Future owner of the one-transaction completion operation.

    The implementation must execute one PostgreSQL transaction that validates
    the job lease, writes/reads the canonical result, updates the wellness
    scan, completes the processing job, and writes outbox/idempotency records.
    """

    def commit_processing_result(
        self, request: CommitProcessingResultRequest
    ) -> CommitProcessingResultResponse: ...


class PostgresProcessingResultCommitter:
    """Calls the database-owned, single-transaction completion function."""

    def __init__(self, connection_factory) -> None:
        self._repository = PostgresProcessingRepository(connection_factory)

    @_retry_transaction_method
    def commit_processing_result(
        self, request: CommitProcessingResultRequest
    ) -> CommitProcessingResultResponse:
        if not isinstance(request, CommitProcessingResultRequest):
            raise TypeError("request must be CommitProcessingResultRequest")
        if not isinstance(request.scan_id, str) or not request.scan_id.strip():
            raise ValueError("scan_id must be a non-empty string")
        if not isinstance(request.processing_version, str) or not request.processing_version.strip():
            raise ValueError("processing_version must be a non-empty string")
        result = request.result
        normalized = result.normalized()
        with self._repository._transaction() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT status, result_id, history_id
                FROM ai_processing.commit_processing_result(
                    %s::uuid, %s::uuid, %s::varchar, %s::uuid, %s::uuid,
                    %s::integer, %s::numeric, %s::varchar, %s::text,
                    %s::text, %s::varchar
                )
                """,
                (
                    request.job_id,
                    request.scan_id,
                    request.processing_version,
                    request.lease_token,
                    request.result_id,
                    normalized["readiness_score"],
                    normalized["confidence"],
                    normalized["risk_level"],
                    normalized["explanation"],
                    normalized["suggested_action"],
                    result.result_hash(),
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("processing commit function returned no result")
            return CommitProcessingResultResponse(
                status=row[0],
                result_ref=str(row[1]) if row[1] is not None else None,
            )
