"""Durable outbox-to-RabbitMQ dispatcher entrypoint."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable
import uuid

from durable_config import DurableConfigurationError, DurableSettings
from processing import OutboxConflict, PostgresProcessingRepository
from rabbitmq_adapter import ProcessingMessage, RabbitMQPublisher


logger = logging.getLogger("ai-outbox-dispatcher")


class OutboxDispatcher:
    def __init__(
        self,
        *,
        repository: Any,
        publisher: Any,
        dispatcher_id: str,
        batch_size: int = 50,
        lock_seconds: int = 60,
        retry_delay_seconds: int = 5,
        reconcile_delay_seconds: int = 30,
    ) -> None:
        self.repository = repository
        self.publisher = publisher
        self.dispatcher_id = dispatcher_id
        self.batch_size = batch_size
        self.lock_seconds = lock_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.reconcile_delay_seconds = reconcile_delay_seconds

    def run_once(self) -> int:
        is_connected = getattr(self.publisher, "is_connected", True)
        if not is_connected:
            self.publisher.connect()
        reconcile = getattr(self.repository, "reconcile_queued_job_events", None)
        if reconcile is not None:
            reconcile(
                limit=self.batch_size,
                min_age_seconds=self.reconcile_delay_seconds,
            )
        recover_expired_leases = getattr(self.repository, "recover_expired_leases", None)
        if recover_expired_leases is not None:
            recover_expired_leases(limit=self.batch_size)
        queue_due_retries = getattr(self.repository, "queue_due_retries", None)
        if queue_due_retries is not None:
            queue_due_retries(limit=self.batch_size)
        events = self.repository.claim_outbox_events(
            self.dispatcher_id,
            limit=self.batch_size,
            lock_seconds=self.lock_seconds,
            event_type="job.process",
        )
        completed = 0
        for event in events:
            try:
                payload = dict(event.payload)
                message = ProcessingMessage.from_mapping(payload)
                self.publisher.publish(message)
                self.repository.mark_outbox_delivered(event.id, self.dispatcher_id)
                completed += 1
                logger.info("dispatch_confirmed event_type=job.process")
            except Exception as exc:
                logger.warning(
                    "dispatch_failed error_type=%s",
                    type(exc).__name__,
                )
                try:
                    self.repository.release_outbox_event(
                        event.id,
                        self.dispatcher_id,
                        failure_code="broker_publish_failed",
                        retry_delay_seconds=self.retry_delay_seconds,
                    )
                except OutboxConflict:
                    logger.info("dispatch_ownership_lost")
        return completed

    def run_forever(self, *, stop_event: Any | None = None, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        stop_event = stop_event or _NeverStop()
        reconnect_delay = 1.0
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                close = getattr(self.publisher, "close", None)
                if close is not None:
                    close()
                logger.warning("dispatcher_cycle_failed error_type=%s", type(exc).__name__)
                sleep_fn(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, 30.0)
            else:
                reconnect_delay = 1.0
                sleep_fn(1.0)


class _NeverStop:
    def is_set(self) -> bool:
        return False


def build_dispatcher_from_env() -> OutboxDispatcher:
    settings = DurableSettings.from_env(role="dispatcher")
    if not settings.enabled:
        raise DurableConfigurationError("durable processing is disabled")
    import psycopg

    factory = lambda: psycopg.connect(settings.database_dsn)
    return OutboxDispatcher(
        repository=PostgresProcessingRepository(factory),
        publisher=RabbitMQPublisher(
            settings.rabbitmq_url,
            retry_ttl_ms=settings.broker_retry_ttl_ms,
        ),
        dispatcher_id=os.getenv("AI_DISPATCHER_ID", f"dispatcher-{uuid.uuid4()}"),
    )


def main() -> None:
    try:
        settings = DurableSettings.from_env(role="dispatcher")
    except DurableConfigurationError:
        logger.error("dispatcher_configuration_invalid")
        raise SystemExit(2)
    dispatcher = build_dispatcher_from_env()
    dispatcher.publisher.connect()
    try:
        dispatcher.run_forever()
    finally:
        dispatcher.publisher.close()


if __name__ == "__main__":  # pragma: no cover
    main()
