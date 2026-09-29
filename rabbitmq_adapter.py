"""RabbitMQ transport for durable processing.

The broker is deliberately only a delivery mechanism.  PostgreSQL job and
outbox state remains authoritative when messages are duplicated or lost during
connection recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable, Mapping
from uuid import UUID


EXCHANGE = "ai.process.v1.exchange"
DEAD_LETTER_EXCHANGE = "ai.process.v1.dlx"
PROCESS_QUEUE = "ai.process.v1"
RETRY_QUEUE = "ai.process.v1.retry"
DLQ = "ai.process.v1.dlq"
ROUTING_KEY = "ai.process.v1"
RETRY_ROUTING_KEY = "retry"
DLQ_ROUTING_KEY = "dead"

logger = logging.getLogger("ai-rabbitmq")


class BrokerError(RuntimeError):
    pass


class BrokerMessageError(BrokerError):
    pass


@dataclass(frozen=True)
class ProcessingMessage:
    event_key: str
    job_id: UUID
    processing_version: str
    trace_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProcessingMessage":
        if not isinstance(value, Mapping):
            raise BrokerMessageError("message must be an object")
        required = {"event_key", "job_id", "processing_version", "trace_id"}
        if set(value) != required:
            raise BrokerMessageError("message fields are invalid")
        try:
            event_key = str(value["event_key"]).strip()
            job_id = UUID(str(value["job_id"]))
            version = str(value["processing_version"]).strip()
            trace_id = str(value["trace_id"]).strip()
        except (TypeError, ValueError, AttributeError) as exc:
            raise BrokerMessageError("message references are invalid") from exc
        if not event_key or len(event_key) > 255 or not version or len(version) > 100:
            raise BrokerMessageError("message references are invalid")
        if not trace_id or len(trace_id) > 255:
            raise BrokerMessageError("message references are invalid")
        return cls(event_key, job_id, version, trace_id)

    @classmethod
    def from_json(cls, body: bytes | str) -> "ProcessingMessage":
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise BrokerMessageError("message JSON is invalid") from exc
        return cls.from_mapping(parsed)

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "event_key": self.event_key,
                "job_id": str(self.job_id),
                "processing_version": self.processing_version,
                "trace_id": self.trace_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def _pika():
    try:
        import pika
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise BrokerError("pika is required for RabbitMQ transport") from exc
    return pika


def declare_topology(channel: Any, *, retry_ttl_ms: int = 5000) -> None:
    pika = _pika()
    if type(retry_ttl_ms) is not int or retry_ttl_ms <= 0:
        raise ValueError("retry_ttl_ms must be positive")
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=DEAD_LETTER_EXCHANGE, exchange_type="direct", durable=True)
    quorum = {"x-queue-type": "quorum"}
    channel.queue_declare(
        queue=PROCESS_QUEUE,
        durable=True,
        arguments={
            **quorum,
            "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
            "x-dead-letter-routing-key": RETRY_ROUTING_KEY,
        },
    )
    channel.queue_declare(
        queue=RETRY_QUEUE,
        durable=True,
        arguments={
            **quorum,
            "x-message-ttl": retry_ttl_ms,
            "x-dead-letter-exchange": EXCHANGE,
            "x-dead-letter-routing-key": ROUTING_KEY,
        },
    )
    channel.queue_declare(queue=DLQ, durable=True, arguments=quorum)
    channel.queue_bind(queue=PROCESS_QUEUE, exchange=EXCHANGE, routing_key=ROUTING_KEY)
    channel.queue_bind(queue=RETRY_QUEUE, exchange=DEAD_LETTER_EXCHANGE, routing_key=RETRY_ROUTING_KEY)
    channel.queue_bind(queue=DLQ, exchange=DEAD_LETTER_EXCHANGE, routing_key=DLQ_ROUTING_KEY)
    channel.basic_qos(prefetch_count=1)


class RabbitMQPublisher:
    def __init__(self, url: str, *, retry_ttl_ms: int = 5000) -> None:
        self._url = url
        self._retry_ttl_ms = retry_ttl_ms
        self._connection = None
        self._channel = None

    @property
    def is_connected(self) -> bool:
        return bool(
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and self._channel.is_open
        )

    def connect(self) -> None:
        pika = _pika()
        params = pika.URLParameters(self._url)
        params.connection_attempts = 1
        params.retry_delay = 0
        params.blocked_connection_timeout = 10
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.confirm_delivery()
        declare_topology(self._channel, retry_ttl_ms=self._retry_ttl_ms)

    def close(self) -> None:
        try:
            if self._connection is not None and not self._connection.is_closed:
                self._connection.close()
        finally:
            self._connection = None
            self._channel = None

    def publish(self, message: ProcessingMessage) -> None:
        if self._channel is None:
            raise BrokerError("RabbitMQ publisher is not connected")
        pika = _pika()
        confirmed = self._channel.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                message_id=message.event_key,
                headers={"processing_version": message.processing_version},
            ),
            mandatory=True,
        )
        if confirmed is False:
            raise BrokerError("RabbitMQ publisher confirmation failed")

    def publish_retry(self, message: ProcessingMessage, *, retry_count: int) -> None:
        if retry_count < 0:
            raise ValueError("retry_count must not be negative")
        if self._channel is None:
            raise BrokerError("RabbitMQ publisher is not connected")
        pika = _pika()
        confirmed = self._channel.basic_publish(
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key=RETRY_ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                message_id=message.event_key,
                headers={"broker_retry_count": retry_count},
            ),
            mandatory=True,
        )
        if confirmed is False:
            raise BrokerError("RabbitMQ retry confirmation failed")

    def publish_dlq(self, message: ProcessingMessage, *, reason: str) -> None:
        if self._channel is None:
            raise BrokerError("RabbitMQ publisher is not connected")
        pika = _pika()
        confirmed = self._channel.basic_publish(
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key=DLQ_ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                message_id=message.event_key,
                headers={"dlq_reason": reason[:80]},
            ),
            mandatory=True,
        )
        if confirmed is False:
            raise BrokerError("RabbitMQ DLQ confirmation failed")


class RabbitMQConsumer:
    """Manual-ack consumer with explicit bounded retry outcomes."""

    def __init__(
        self,
        url: str,
        *,
        retry_ttl_ms: int = 5000,
        max_broker_redeliveries: int = 5,
    ) -> None:
        self._url = url
        self._retry_ttl_ms = retry_ttl_ms
        self._max_broker_redeliveries = max_broker_redeliveries
        self._connection = None
        self._channel = None
        self._stop_requested = False

    def _publish_retry(self, message: ProcessingMessage, retry_count: int) -> None:
        pika = _pika()
        confirmed = self._channel.basic_publish(
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key=RETRY_ROUTING_KEY,
            body=message.to_json(),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                message_id=message.event_key,
                headers={"broker_retry_count": retry_count},
            ),
            mandatory=True,
        )
        if confirmed is False:
            raise BrokerError("RabbitMQ retry confirmation failed")

    def _publish_raw_dlq(self, body: bytes, reason: str) -> None:
        pika = _pika()
        confirmed = self._channel.basic_publish(
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key=DLQ_ROUTING_KEY,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
                headers={"dlq_reason": reason[:80]},
            ),
            mandatory=True,
        )
        if confirmed is False:
            raise BrokerError("RabbitMQ DLQ confirmation failed")

    def run(self, handler: Callable[[ProcessingMessage, Mapping[str, Any]], str]) -> None:
        self._stop_requested = False
        reconnect_delay = 1.0
        while not self._stop_requested:
            try:
                self._run_once(handler)
                reconnect_delay = 1.0
            except Exception as exc:
                self.close()
                if self._stop_requested:
                    break
                logger.warning("broker_reconnect_scheduled error_type=%s", type(exc).__name__)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, 30.0)
            else:
                break

    def _run_once(self, handler: Callable[[ProcessingMessage, Mapping[str, Any]], str]) -> None:
        pika = _pika()
        params = pika.URLParameters(self._url)
        params.connection_attempts = 1
        params.retry_delay = 0
        params.blocked_connection_timeout = 10
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.confirm_delivery()
        declare_topology(self._channel, retry_ttl_ms=self._retry_ttl_ms)

        def callback(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            headers = getattr(properties, "headers", None) or {}
            try:
                message = ProcessingMessage.from_json(body)
            except BrokerMessageError:
                try:
                    self._publish_raw_dlq(body, "malformed_message")
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                return
            try:
                outcome = handler(message, headers)
            except Exception:
                outcome = "retry"
            if outcome == "ack":
                channel.basic_ack(delivery_tag=method.delivery_tag)
            elif outcome == "retry":
                try:
                    retry_count = int(headers.get("broker_retry_count", 0) or 0)
                except (TypeError, ValueError):
                    retry_count = 0
                try:
                    if retry_count >= self._max_broker_redeliveries:
                        self._publish_raw_dlq(body, "max_broker_redeliveries")
                    else:
                        self._publish_retry(message, retry_count + 1)
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            elif outcome == "dlq":
                try:
                    self._publish_raw_dlq(body, "handler_dlq")
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                except Exception:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            else:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self._channel.basic_consume(queue=PROCESS_QUEUE, on_message_callback=callback, auto_ack=False)
        try:
            self._channel.start_consuming()
        finally:
            self.close()

    def stop(self) -> None:
        self._stop_requested = True
        if self._channel is not None and self._channel.is_open:
            self._channel.stop_consuming()

    def close(self) -> None:
        try:
            if self._connection is not None and not self._connection.is_closed:
                self._connection.close()
        finally:
            self._connection = None
            self._channel = None
