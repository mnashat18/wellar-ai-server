"""RabbitMQ AI worker with coordinator-only access."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

import requests

from canonical_analysis import AnalysisInput, AnalysisOutcome, analyze_scan
from analysis_runtime import AnalysisRuntimeStartupError, get_runtime
from durable_config import DurableConfigurationError, DurableSettings
from processing_commit import ValidatedResultPayload
from rabbitmq_adapter import ProcessingMessage, RabbitMQConsumer
from validation import ValidationPolicy


logger = logging.getLogger("ai-worker")


def _convert_audio_to_wav(path: Path, temp_dir: Path) -> Path:
    """Keep the worker's audio input contract aligned with the legacy path."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg_unavailable")
    output = temp_dir / "audio-normalized.wav"
    try:
        timeout = float(os.getenv("AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS", "3.5"))
    except (TypeError, ValueError):
        timeout = 3.5
    if timeout <= 0:
        timeout = 3.5
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-t",
            "3.0",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=timeout,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("audio_conversion_empty")
    return output


class CoordinatorHttpError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LeaseLost(CoordinatorHttpError):
    pass


class CoordinatorHttpClient:
    """Worker-side HTTP client; it never stores Directus/PostgreSQL credentials."""

    def __init__(self, base_url: str, secret: str, *, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._secret = secret
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, *, worker_id: str | None = None, lease_token: UUID | None = None) -> dict[str, str]:
        headers = {"X-AI-Internal-Secret": self._secret, "Accept": "application/json"}
        if worker_id:
            headers["X-AI-Worker-ID"] = worker_id
        if lease_token is not None:
            headers["X-AI-Lease-Token"] = str(lease_token)
        return headers

    def _request(self, method: str, path: str, *, worker_id: str | None = None, lease_token: UUID | None = None, json: Mapping[str, Any] | None = None) -> requests.Response:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(worker_id=worker_id, lease_token=lease_token),
                json=dict(json) if json is not None else None,
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise CoordinatorHttpError("coordinator_unavailable") from exc
        return response

    @staticmethod
    def _json(response: requests.Response) -> dict[str, Any]:
        try:
            value = response.json()
        except ValueError as exc:
            raise CoordinatorHttpError("coordinator_invalid_response", status_code=response.status_code) from exc
        return value if isinstance(value, dict) else {}

    def claim(
        self,
        job_id: UUID,
        worker_id: str,
        processing_version: str,
        trace_id: str,
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"/internal/processing/jobs/{job_id}/claim",
            json={
                "worker_id": worker_id,
                "processing_version": processing_version,
                "trace_id": trace_id,
            },
        )
        if response.status_code >= 500:
            raise CoordinatorHttpError("coordinator_unavailable", status_code=response.status_code)
        return self._json(response)

    def heartbeat(self, job_id: UUID, worker_id: str, lease_token: UUID) -> None:
        response = self._request(
            "POST",
            f"/internal/processing/jobs/{job_id}/heartbeat",
            json={"worker_id": worker_id, "lease_token": str(lease_token)},
        )
        if response.status_code != 200:
            raise LeaseLost("processing_lease_lost", status_code=response.status_code)

    def asset(self, job_id: UUID, worker_id: str, lease_token: UUID, role: str) -> tuple[bytes, str]:
        response = self._request(
            "GET",
            f"/internal/processing/jobs/{job_id}/input/{role}",
            worker_id=worker_id,
            lease_token=lease_token,
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("asset_not_available", status_code=response.status_code)
        return response.content, response.headers.get("Content-Type", "application/octet-stream")

    def commit(self, job_id: UUID, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"/internal/processing/jobs/{job_id}/commit",
            json=payload,
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("processing_commit_rejected", status_code=response.status_code)
        return self._json(response)

    def fail(self, job_id: UUID, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"/internal/processing/jobs/{job_id}/fail",
            json=payload,
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("processing_failure_rejected", status_code=response.status_code)
        return self._json(response)

    def register_worker(self, worker_id: str, worker_version: str, capacity: int) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/internal/processing/workers/register",
            json={"worker_id": worker_id, "worker_version": worker_version, "capacity": capacity},
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("worker_registration_failed", status_code=response.status_code)
        return self._json(response)

    def heartbeat_worker(self, worker_id: str, lease_token: UUID, active_slots: int) -> None:
        response = self._request(
            "POST",
            "/internal/processing/workers/heartbeat",
            json={"worker_id": worker_id, "lease_token": str(lease_token), "active_slots": active_slots},
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("worker_liveness_lost", status_code=response.status_code)

    def drain_worker(self, worker_id: str, lease_token: UUID) -> None:
        response = self._request(
            "POST",
            "/internal/processing/workers/drain",
            json={"worker_id": worker_id, "lease_token": str(lease_token)},
        )
        if response.status_code != 200:
            raise CoordinatorHttpError("worker_drain_failed", status_code=response.status_code)


class DurableWorker:
    def __init__(
        self,
        *,
        coordinator: Any,
        consumer: Any,
        worker_id: str,
        lease_seconds: int = 60,
        heartbeat_seconds: int = 20,
        analyzer: Callable[[AnalysisInput], AnalysisOutcome] = analyze_scan,
        policy: ValidationPolicy | None = None,
        shutdown_grace_seconds: int = 30,
    ) -> None:
        self.coordinator = coordinator
        self.consumer = consumer
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.analyzer = analyzer
        self.policy = policy or ValidationPolicy.from_env()
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self._shutdown = threading.Event()
        self._active_job: UUID | None = None
        self._active_lease: UUID | None = None

    def request_shutdown(self, *_args: Any) -> None:
        self._shutdown.set()
        stop = getattr(self.consumer, "stop", None)
        if stop is not None:
            stop()

    def handle_message(self, message: ProcessingMessage, headers: Mapping[str, Any]) -> str:
        if self._shutdown.is_set():
            return "retry"
        try:
            claim = self.coordinator.claim(
                message.job_id,
                self.worker_id,
                message.processing_version,
                message.trace_id,
            )
        except Exception as exc:
            logger.warning("worker_claim_unavailable error_type=%s", type(exc).__name__)
            return "retry"
        state = str(claim.get("status") or "")
        if state in {"terminal", "completed"}:
            return "ack"
        if state not in {"processing"}:
            # The dispatcher can publish immediately after a commit.  ACKing
            # here would lose the event, so the broker adapter sends bounded
            # retries through the retry queue.
            return "retry"

        lease_token = UUID(str(claim["lease_token"]))
        self._active_job = message.job_id
        self._active_lease = lease_token
        lease_lost = threading.Event()
        heartbeat_stop = threading.Event()

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(self.heartbeat_seconds):
                try:
                    self.coordinator.heartbeat(message.job_id, self.worker_id, lease_token)
                except Exception:
                    lease_lost.set()
                    return

        heartbeat_thread = threading.Thread(target=heartbeat_loop, name="ai-worker-heartbeat", daemon=True)
        heartbeat_thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="ai-worker-") as temp_dir:
                paths: dict[str, str | None] = {"video": None, "audio": None, "image": None}
                for role in ("video", "audio", "image"):
                    try:
                        content, content_type = self.coordinator.asset(
                            message.job_id,
                            self.worker_id,
                            lease_token,
                            role,
                        )
                    except CoordinatorHttpError as exc:
                        if role == "video" and self.policy.require_video:
                            raise
                        if role == "audio" and self.policy.require_audio:
                            raise
                        if role == "image" and self.policy.require_image:
                            raise
                        continue
                    suffix = ".mp4" if role == "video" else ".wav" if role == "audio" else ".jpg"
                    if role == "audio":
                        suffix = ".bin"
                    path = Path(temp_dir) / f"{role}{suffix}"
                    path.write_bytes(content)
                    if role == "audio" and str(content_type).lower() not in {"audio/wav", "audio/x-wav", "audio/wave"}:
                        path = _convert_audio_to_wav(path, Path(temp_dir))
                    paths[role] = str(path)
                if lease_lost.is_set():
                    raise LeaseLost("processing_lease_lost")
                analysis_context = claim.get("analysis_context") or {}
                outcome = self.analyzer(
                    AnalysisInput(
                        scan_id=str(claim.get("scan_id") or ""),
                        video_path=paths["video"],
                        audio_path=paths["audio"],
                        image_path=paths["image"],
                        task_metrics=analysis_context.get("task_metrics"),
                        baseline=analysis_context.get("baseline"),
                        expected_phrase=analysis_context.get("expected_phrase"),
                    )
                )
                if lease_lost.is_set():
                    raise LeaseLost("processing_lease_lost")
                if not outcome.ok:
                    self.coordinator.fail(
                        message.job_id,
                        {
                            "worker_id": self.worker_id,
                            "lease_token": str(lease_token),
                            "failure_code": outcome.failure_code or "analysis_failed",
                            "error_type": outcome.error_type or "AnalysisError",
                            "retry_delay_seconds": 5,
                            "terminal_failure": outcome.error_type == "ValidationError",
                        },
                    )
                    return "ack"
                result = ValidatedResultPayload(outcome.result or {})
                commit_result = self.coordinator.commit(
                    message.job_id,
                    {
                        "scan_id": str(claim.get("scan_id") or ""),
                        "processing_version": message.processing_version,
                        "lease_token": str(lease_token),
                        "result_id": str(uuid4()),
                        **result.normalized(),
                    },
                )
                if commit_result.get("status") != "completed":
                    raise CoordinatorHttpError("processing_commit_rejected")
                logger.info("commit_completed event=job.completed")
                return "ack"
        except LeaseLost:
            logger.info("lease_lost result_discarded")
            return "ack"
        except CoordinatorHttpError as exc:
            if exc.status_code is not None and 400 <= exc.status_code < 500:
                try:
                    self.coordinator.fail(
                        message.job_id,
                        {
                            "worker_id": self.worker_id,
                            "lease_token": str(lease_token),
                            "failure_code": "coordinator_rejected",
                            "error_type": type(exc).__name__,
                            "retry_delay_seconds": 5,
                            "terminal_failure": True,
                        },
                    )
                    return "ack"
                except Exception:
                    return "retry"
            return "retry"
        except Exception as exc:
            logger.warning("worker_processing_failed error_type=%s", type(exc).__name__)
            try:
                self.coordinator.fail(
                    message.job_id,
                    {
                        "worker_id": self.worker_id,
                        "lease_token": str(lease_token),
                        "failure_code": "worker_processing_failed",
                        "error_type": type(exc).__name__,
                        "retry_delay_seconds": 5,
                        "terminal_failure": False,
                    },
                )
                return "ack"
            except Exception:
                return "retry"
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
            self._active_job = None
            self._active_lease = None

    def run(self) -> None:
        runtime = get_runtime()
        if not runtime.is_ready():
            runtime.start_background()
            deadline = time.monotonic() + 120.0
            while not runtime.is_ready() and time.monotonic() < deadline:
                if self._shutdown.wait(0.25):
                    return
            if not runtime.is_ready():
                raise AnalysisRuntimeStartupError("analysis_runtime_not_ready")
        worker_lease = self.coordinator.register_worker(
            self.worker_id,
            os.getenv("AI_WORKER_VERSION", "phase-b"),
            1,
        )
        worker_token = UUID(str(worker_lease["lease_token"]))
        liveness_stop = threading.Event()

        def liveness_loop() -> None:
            while not liveness_stop.wait(self.heartbeat_seconds):
                try:
                    self.coordinator.heartbeat_worker(
                        self.worker_id,
                        worker_token,
                        1 if self._active_job is not None else 0,
                    )
                except Exception:
                    logger.warning("worker_liveness_lost")
                    return

        liveness_thread = threading.Thread(target=liveness_loop, name="ai-worker-liveness", daemon=True)
        liveness_thread.start()
        try:
            self.consumer.run(self.handle_message)
        finally:
            liveness_stop.set()
            liveness_thread.join(timeout=1.0)
            try:
                self.coordinator.drain_worker(self.worker_id, worker_token)
            except Exception:
                logger.info("worker_drain_unavailable")


def build_worker_from_env() -> DurableWorker:
    settings = DurableSettings.from_env(role="worker")
    if not settings.enabled:
        raise DurableConfigurationError("durable processing is disabled")
    coordinator = CoordinatorHttpClient(
        settings.coordinator_url,
        settings.internal_secret,
    )
    consumer = RabbitMQConsumer(
        settings.rabbitmq_url,
        retry_ttl_ms=settings.broker_retry_ttl_ms,
        max_broker_redeliveries=settings.max_broker_redeliveries,
    )
    return DurableWorker(
        coordinator=coordinator,
        consumer=consumer,
        worker_id=settings.worker_id,
        lease_seconds=settings.lease_seconds,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def main() -> None:
    try:
        worker = build_worker_from_env()
    except DurableConfigurationError:
        logger.error("worker_configuration_invalid")
        raise SystemExit(2)
    signal.signal(signal.SIGTERM, worker.request_shutdown)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, worker.request_shutdown)
    worker.run()


if __name__ == "__main__":  # pragma: no cover
    main()
