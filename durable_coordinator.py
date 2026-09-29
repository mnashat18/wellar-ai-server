"""Trusted coordinator operations for the Phase B durable path."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import quote
from uuid import UUID, uuid4

import requests

from directus_client import DirectusClient
from processing import (
    EnqueuedJob,
    JobSpec,
    JobStatus,
    LeaseClaim,
    LeaseConflict,
    ProcessingJob,
    ProcessingRepository,
)
from processing_commit import (
    CommitProcessingResultRequest,
    CommitProcessingResultResponse,
    ProcessingResultCommitter,
    ValidatedResultPayload,
)
from utils import MAX_DOWNLOAD_BYTES


ASSET_ROLES = frozenset({"video", "audio", "image", "thumbnail"})


class CoordinatorError(RuntimeError):
    pass


class CoordinatorDependencyError(CoordinatorError):
    pass


class CoordinatorAuthorizationError(CoordinatorError):
    pass


def verify_internal_secret(provided: str | None, expected: str | None) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _relation_id(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("id", value.get("uuid"))
    return value


def _same(left: Any, right: Any) -> bool:
    left_value = _relation_id(left)
    right_value = _relation_id(right)
    return left_value is not None and right_value is not None and str(left_value) == str(right_value)


def _safe_token(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoordinatorAuthorizationError("internal authentication required")
    return value.strip()


@dataclass(frozen=True)
class DurableJobSubmission:
    job: ProcessingJob
    event_key: str
    created: bool


class DurableCoordinator:
    """The only Phase B component allowed to use Directus/processing DB access."""

    def __init__(
        self,
        *,
        repository: ProcessingRepository,
        directus: DirectusClient,
        committer: ProcessingResultCommitter,
        internal_secret: str | None,
        processing_version: str,
        lease_seconds: int = 60,
        heartbeat_seconds: int = 20,
        directus_timeout: float | None = None,
    ) -> None:
        self.repository = repository
        self.directus = directus
        self.committer = committer
        self.internal_secret = internal_secret
        self.processing_version = processing_version
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.directus_timeout = directus_timeout or float(getattr(directus, "timeout", 30))

    def check_ready(self) -> None:
        if not self.directus.is_configured():
            raise CoordinatorDependencyError("directus_not_configured")
        ping = getattr(self.repository, "ping", None)
        if ping is not None:
            try:
                ping()
            except Exception as exc:
                raise CoordinatorDependencyError("processing_database_unavailable") from exc
        try:
            self.directus.check_processing_readiness()
        except Exception as exc:
            raise CoordinatorDependencyError("directus_unavailable") from exc

    def submit_authorized(self, *, scan_context: Mapping[str, Any], requester_user_id: Any) -> DurableJobSubmission:
        scan_id = str(_relation_id(scan_context.get("id")) or "").strip()
        member_id = str(_relation_id(scan_context.get("member")) or "").strip()
        business_profile_id = str(_relation_id(scan_context.get("business_profile")) or "").strip()
        requester = str(_relation_id(requester_user_id) or "").strip()
        if not scan_id or not requester or not member_id or not business_profile_id:
            raise CoordinatorAuthorizationError("scan_authorization_context_invalid")
        if str(scan_context.get("status") or "").strip() not in {"media_ready", "processing", "completed"}:
            raise CoordinatorAuthorizationError("scan_not_ready")
        event_key = f"job:{scan_id}:process:{self.processing_version}"
        trace_id = str(uuid4())
        payload = {
            "event_key": event_key,
            "processing_version": self.processing_version,
            "trace_id": trace_id,
        }
        spec = JobSpec(
            scan_id=scan_id,
            processing_version=self.processing_version,
            requester_user_id=requester,
            member_id=member_id,
            business_profile_id=business_profile_id,
            trace_id=trace_id,
        ).validated()
        result: EnqueuedJob = self.repository.enqueue_job(
            spec,
            event_key=event_key,
            event_type="job.process",
            payload=payload,
        )
        return DurableJobSubmission(
            job=result.job,
            event_key=event_key,
            created=result.created,
        )

    def _job_context(self, job: ProcessingJob) -> dict[str, Any]:
        try:
            context = self.directus.get_scan_context(job.scan_id)
        except Exception as exc:
            raise CoordinatorDependencyError("scan_context_unavailable") from exc
        if not isinstance(context, Mapping):
            raise CoordinatorAuthorizationError("scan_context_invalid")
        if (
            not _same(context.get("id"), job.scan_id)
            or not _same(context.get("user"), job.requester_user_id)
            or not _same(context.get("member"), job.member_id)
            or not _same(context.get("business_profile"), job.business_profile_id)
        ):
            raise CoordinatorAuthorizationError("job_scan_binding_invalid")
        return dict(context)

    def claim(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        processing_version: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[str, LeaseClaim | None, dict[str, Any] | None]:
        canonical_job = self.repository.get_job(job_id)
        if (
            processing_version is not None
            and canonical_job.processing_version != processing_version
        ) or (trace_id is not None and canonical_job.trace_id != trace_id):
            raise CoordinatorAuthorizationError("job_message_binding_invalid")
        claim = self.repository.claim_job(
            job_id,
            worker_id,
            lease_seconds=self.lease_seconds,
        )
        if claim is None:
            job = self.repository.get_job(job_id)
            if job.status is JobStatus.ACCEPTED:
                return "not_ready", None, None
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED_TERMINAL}:
                return "terminal", None, None
            return "not_claimable", None, None
        context = self._job_context(claim.job)
        baseline_rows = self.directus.get_employee_baselines(
            claim.job.member_id,
            claim.job.business_profile_id,
        )
        context["baseline"] = baseline_rows[0] if len(baseline_rows) == 1 else None
        return "claimed", claim, context

    def heartbeat(self, *, job_id: UUID, worker_id: str, lease_token: UUID) -> ProcessingJob:
        return self.repository.heartbeat(
            job_id,
            worker_id,
            lease_token,
            lease_seconds=self.lease_seconds,
        )

    def assert_lease(self, *, job_id: UUID, worker_id: str, lease_token: UUID) -> ProcessingJob:
        return self.repository.assert_lease(job_id, worker_id, lease_token)

    def read_asset(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        asset_role: str,
    ) -> tuple[bytes, str]:
        role = asset_role.strip().lower()
        if role not in ASSET_ROLES:
            raise CoordinatorAuthorizationError("asset_role_not_allowed")
        job = self.assert_lease(job_id=job_id, worker_id=worker_id, lease_token=lease_token)
        context = self._job_context(job)
        media = context.get("scan_media")
        if (
            not isinstance(media, Mapping)
            or not _same(media.get("scan_id"), job.scan_id)
            or not _same(media.get("business_profile"), job.business_profile_id)
            or media.get("is_deleted") is True
        ):
            raise CoordinatorAuthorizationError("scan_media_not_authorized")
        field = "thumbnail" if role in {"image", "thumbnail"} else f"{role}_file"
        file_id = _relation_id(media.get(field))
        if not file_id:
            raise CoordinatorAuthorizationError("asset_not_available")
        return self._download_asset(file_id)

    def _download_asset(self, file_id: Any) -> tuple[bytes, str]:
        if not self.directus.base_url or not self.directus.token:
            raise CoordinatorDependencyError("directus_not_configured")
        path = f"/assets/{quote(str(file_id), safe='')}"
        response = None
        try:
            response = requests.get(
                f"{self.directus.base_url}{path}",
                headers={"Authorization": f"Bearer {self.directus.token}"},
                timeout=self.directus_timeout,
                stream=True,
                allow_redirects=False,
            )
            if response.status_code >= 300:
                raise CoordinatorDependencyError("asset_download_failed")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                raise CoordinatorDependencyError("asset_too_large")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise CoordinatorDependencyError("asset_too_large")
                chunks.append(chunk)
            if total == 0:
                raise CoordinatorDependencyError("asset_empty")
            return b"".join(chunks), response.headers.get("Content-Type", "application/octet-stream")
        except CoordinatorError:
            raise
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise CoordinatorDependencyError("asset_download_failed") from exc
        finally:
            if response is not None:
                response.close()

    def commit(
        self,
        *,
        job_id: UUID,
        scan_id: str,
        processing_version: str,
        lease_token: UUID,
        result_id: UUID,
        result: ValidatedResultPayload,
    ) -> CommitProcessingResultResponse:
        return self.committer.commit_processing_result(
            CommitProcessingResultRequest(
                job_id=job_id,
                scan_id=scan_id,
                processing_version=processing_version,
                lease_token=lease_token,
                result_id=result_id,
                result=result,
            )
        )

    def fail(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        failure_code: str,
        error_type: str,
        retry_delay_seconds: int,
        terminal_failure: bool = False,
    ) -> ProcessingJob:
        return self.repository.schedule_retry(
            job_id,
            worker_id,
            lease_token,
            failure_code=failure_code,
            error_type=error_type,
            retry_delay_seconds=retry_delay_seconds,
            terminal_failure=terminal_failure,
        )

    def register_worker(self, *, worker_id: str, worker_version: str, capacity: int) -> Any:
        return self.repository.register_worker(
            worker_id,
            worker_version,
            capacity=capacity,
        )

    def heartbeat_worker(self, *, worker_id: str, lease_token: UUID, active_slots: int) -> Any:
        return self.repository.heartbeat_worker(
            worker_id,
            lease_token,
            active_slots=active_slots,
        )

    def drain_worker(self, *, worker_id: str, lease_token: UUID) -> Any:
        return self.repository.set_worker_draining(
            worker_id,
            lease_token,
            draining=True,
        )

    def public_status(self, *, scan_id: str, processing_version: str) -> ProcessingJob:
        # The caller performs current Directus tenant authorization before this
        # method is reached; this method never exposes internal lease/broker data.
        job = self.repository.get_job_by_scan(scan_id, processing_version)
        if job is None:
            raise CoordinatorAuthorizationError("processing_status_not_found")
        return job
