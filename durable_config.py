"""Fail-closed configuration for the optional durable processing path."""

from __future__ import annotations

from dataclasses import dataclass
import os


class DurableConfigurationError(RuntimeError):
    """Raised when durable mode cannot be safely configured."""


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise DurableConfigurationError(f"{name} must be a boolean")


def _positive_int(name: str, default: int, *, maximum: int = 3600) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw.strip())
    except (TypeError, ValueError):
        raise DurableConfigurationError(f"{name} must be a positive integer") from None
    if value <= 0 or value > maximum:
        raise DurableConfigurationError(f"{name} must be between 1 and {maximum}")
    return value


@dataclass(frozen=True)
class DurableSettings:
    enabled: bool
    processing_version: str
    database_dsn: str | None
    rabbitmq_url: str | None
    internal_secret: str | None
    coordinator_url: str | None
    worker_id: str
    worker_capacity: int
    lease_seconds: int
    heartbeat_seconds: int
    outbox_poll_seconds: int
    broker_retry_ttl_ms: int
    max_broker_redeliveries: int

    @classmethod
    def from_env(cls, *, role: str = "api") -> "DurableSettings":
        enabled = _bool("DURABLE_PROCESSING_ENABLED", False)
        processing_version = os.getenv("AI_PROCESSING_VERSION", "cie_v1_2").strip()
        if not processing_version or len(processing_version) > 100:
            raise DurableConfigurationError("AI_PROCESSING_VERSION must be 1..100 characters")

        settings = cls(
            enabled=enabled,
            processing_version=processing_version,
            database_dsn=(os.getenv("AI_PROCESSING_DATABASE_DSN") or "").strip() or None,
            rabbitmq_url=(os.getenv("RABBITMQ_URL") or "").strip() or None,
            internal_secret=(os.getenv("AI_INTERNAL_SERVICE_SECRET") or "").strip() or None,
            coordinator_url=(os.getenv("AI_COORDINATOR_URL") or "").strip() or None,
            worker_id=(os.getenv("AI_WORKER_ID", "ai-worker-local").strip() or "ai-worker-local"),
            worker_capacity=_positive_int("AI_WORKER_CAPACITY", 1, maximum=64),
            lease_seconds=_positive_int("AI_PROCESSING_LEASE_SECONDS", 60),
            heartbeat_seconds=_positive_int("AI_PROCESSING_HEARTBEAT_SECONDS", 20),
            outbox_poll_seconds=_positive_int("AI_OUTBOX_POLL_SECONDS", 1, maximum=300),
            broker_retry_ttl_ms=_positive_int("AI_BROKER_RETRY_TTL_MS", 5000, maximum=300_000),
            max_broker_redeliveries=_positive_int("AI_BROKER_MAX_REDELIVERIES", 5, maximum=100),
        )
        if settings.heartbeat_seconds >= settings.lease_seconds:
            raise DurableConfigurationError(
                "AI_PROCESSING_HEARTBEAT_SECONDS must be less than AI_PROCESSING_LEASE_SECONDS"
            )
        if role == "api" and enabled:
            settings.require_api_dependencies()
        elif role == "worker" and enabled:
            settings.require_worker_dependencies()
        elif role == "dispatcher" and enabled:
            settings.require_dispatcher_dependencies()
        return settings

    def require_api_dependencies(self) -> None:
        self._require("AI_PROCESSING_DATABASE_DSN", self.database_dsn)
        self._require("RABBITMQ_URL", self.rabbitmq_url)
        self._require_secret()

    def require_worker_dependencies(self) -> None:
        self._require("RABBITMQ_URL", self.rabbitmq_url)
        self._require("AI_COORDINATOR_URL", self.coordinator_url)
        self._require_secret()

    def require_dispatcher_dependencies(self) -> None:
        self._require("AI_PROCESSING_DATABASE_DSN", self.database_dsn)
        self._require("RABBITMQ_URL", self.rabbitmq_url)

    def _require(self, name: str, value: str | None) -> None:
        if not value:
            raise DurableConfigurationError(f"{name} is required when durable processing is enabled")

    def _require_secret(self) -> None:
        self._require("AI_INTERNAL_SERVICE_SECRET", self.internal_secret)
        if self.internal_secret is not None and len(self.internal_secret) < 32:
            raise DurableConfigurationError("AI_INTERNAL_SERVICE_SECRET must be at least 32 characters")


def durable_enabled() -> bool:
    return _bool("DURABLE_PROCESSING_ENABLED", False)
