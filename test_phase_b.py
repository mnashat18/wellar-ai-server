from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

from ai_outbox_dispatcher import OutboxDispatcher
from ai_worker import DurableWorker
from canonical_analysis import AnalysisInput, AnalysisOutcome, analyze_scan
from durable_config import DurableConfigurationError, DurableSettings
from processing import InMemoryProcessingRepository, JobSpec, JobStatus, utc_now
from processing_commit import ValidatedResultPayload
from rabbitmq_adapter import BrokerMessageError, ProcessingMessage
from validation import ValidationPolicy


class PhaseBFoundationTests(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryProcessingRepository()
        self.spec = JobSpec(
            scan_id="phase-b-synthetic-scan",
            processing_version="cie_v1_2",
            requester_user_id="phase-b-user",
            member_id="phase-b-member",
            business_profile_id="phase-b-workspace",
            trace_id="phase-b-trace",
        )

    def test_enqueue_creates_queued_job_and_minimal_event_once(self):
        first = self.repo.enqueue_job(
            self.spec,
            event_key="job:phase-b-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "phase-b-trace"},
        )
        second = self.repo.enqueue_job(
            self.spec,
            event_key="job:phase-b-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "phase-b-trace"},
        )
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.job.id, second.job.id)
        self.assertEqual(first.job.status, JobStatus.QUEUED)
        self.assertEqual(first.event.id, second.event.id)
        self.assertEqual(
            set(first.event.payload),
            {"event_key", "job_id", "processing_version", "trace_id"},
        )

    def test_processing_message_is_strictly_minimal(self):
        message = ProcessingMessage(
            event_key="event-1",
            job_id=uuid4(),
            processing_version="cie_v1_2",
            trace_id="trace-1",
        )
        decoded = ProcessingMessage.from_json(message.to_json())
        self.assertEqual(decoded, message)
        with self.assertRaises(BrokerMessageError):
            ProcessingMessage.from_mapping({**message.__dict__, "secret": "not-allowed"})

    def test_dispatcher_publishes_only_after_claim_and_confirms(self):
        event = self.repo.enqueue_job(
            self.spec,
            event_key="job:phase-b-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "phase-b-trace"},
        ).event

        class Publisher:
            is_connected = True

            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        publisher = Publisher()
        dispatcher = OutboxDispatcher(
            repository=self.repo,
            publisher=publisher,
            dispatcher_id="phase-b-dispatcher",
        )
        self.assertEqual(dispatcher.run_once(), 1)
        self.assertEqual(len(publisher.messages), 1)
        self.assertIsNotNone(self.repo._outbox[event.id].delivered_at)

    def test_dispatcher_releases_event_when_publish_fails(self):
        self.repo.enqueue_job(
            self.spec,
            event_key="job:phase-b-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "phase-b-trace"},
        )

        class Publisher:
            is_connected = True

            def publish(self, message):
                raise RuntimeError("broker unavailable")

        dispatcher = OutboxDispatcher(
            repository=self.repo,
            publisher=Publisher(),
            dispatcher_id="phase-b-dispatcher",
            retry_delay_seconds=2,
        )
        self.assertEqual(dispatcher.run_once(), 0)
        event = next(iter(self.repo._outbox.values()))
        self.assertIsNone(event.locked_by)
        self.assertEqual(event.attempt_count, 1)
        self.assertEqual(event.last_error_code, "broker_publish_failed")

    def test_dispatcher_reconciles_delivered_event_for_still_queued_job(self):
        self.repo.enqueue_job(
            self.spec,
            event_key="job:phase-b-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "phase-b-trace"},
        )

        class Publisher:
            is_connected = True

            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        publisher = Publisher()
        dispatcher = OutboxDispatcher(
            repository=self.repo,
            publisher=publisher,
            dispatcher_id="phase-b-dispatcher",
        )
        self.assertEqual(dispatcher.run_once(), 1)
        self.assertEqual(dispatcher.run_once(), 0)
        self.assertEqual(len(publisher.messages), 1)
        reopened = self.repo.reconcile_queued_job_events(
            now=utc_now() + timedelta(seconds=31),
            min_age_seconds=30,
        )
        self.assertEqual(len(reopened), 1)

    def test_worker_uses_coordinator_and_commits_normalized_result(self):
        lease_token = uuid4()

        class Coordinator:
            def __init__(self):
                self.assets = []
                self.commits = []
                self.failures = []

            def claim(self, job_id, worker_id, processing_version, trace_id):
                return {
                    "status": "processing",
                    "lease_token": str(lease_token),
                    "analysis_context": {"task_metrics": None, "baseline": None},
                }

            def heartbeat(self, job_id, worker_id, token):
                self.assert_token(token)

            def assert_token(self, token):
                if token != lease_token:
                    raise AssertionError("unexpected lease token")

            def asset(self, job_id, worker_id, token, role):
                self.assert_token(token)
                self.assets.append(role)
                content_type = "audio/wav" if role == "audio" else "application/octet-stream"
                return role.encode("ascii"), content_type

            def commit(self, job_id, payload):
                self.commits.append((job_id, payload))
                return {"status": "completed", "result_id": "result-ref"}

            def fail(self, job_id, payload):
                self.failures.append((job_id, payload))
                return {"status": "failed_retryable"}

        coordinator = Coordinator()

        def analyzer(request: AnalysisInput) -> AnalysisOutcome:
            self.assertTrue(all(request_path and Path(request_path).exists() for request_path in (
                request.video_path,
                request.audio_path,
                request.image_path,
            )))
            return AnalysisOutcome(
                ok=True,
                result={
                    "readiness_score": 75,
                    "confidence": 0.8,
                    "risk_level": "stable",
                    "explanation": "synthetic explanation",
                    "suggested_action": "synthetic action",
                },
            )

        worker = DurableWorker(
            coordinator=coordinator,
            consumer=object(),
            worker_id="phase-b-worker",
            analyzer=analyzer,
            policy=ValidationPolicy(require_video=True, require_audio=True, require_image=True),
        )
        message = ProcessingMessage(
            event_key="event-1",
            job_id=uuid4(),
            processing_version="cie_v1_2",
            trace_id="trace-1",
        )
        self.assertEqual(worker.handle_message(message, {}), "ack")
        self.assertEqual(coordinator.assets, ["video", "audio", "image"])
        self.assertEqual(len(coordinator.commits), 1)
        self.assertEqual(coordinator.failures, [])
        ValidatedResultPayload(
            {key: value for key, value in coordinator.commits[0][1].items() if key in {
                "readiness_score", "confidence", "risk_level", "explanation", "suggested_action"
            }}
        )

    def test_worker_discards_result_when_lease_is_lost(self):
        lease_token = uuid4()

        class Coordinator:
            def claim(self, job_id, worker_id, processing_version, trace_id):
                return {"status": "processing", "lease_token": str(lease_token), "analysis_context": {}}

            def asset(self, job_id, worker_id, token, role):
                raise RuntimeError("lease lost")

            def fail(self, job_id, payload):
                raise AssertionError("stale worker must not fail the job")

            def commit(self, job_id, payload):
                raise AssertionError("stale worker must not commit")

        worker = DurableWorker(
            coordinator=Coordinator(),
            consumer=object(),
            worker_id="phase-b-worker",
            analyzer=lambda request: (_ for _ in ()).throw(AssertionError("analyzer must not run")),
            policy=ValidationPolicy(require_video=True),
        )
        message = ProcessingMessage("event-1", uuid4(), "cie_v1_2", "trace-1")
        self.assertIn(worker.handle_message(message, {}), {"ack", "retry"})


class PhaseBConfigurationAndParityTests(unittest.TestCase):
    def test_durable_mode_is_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = DurableSettings.from_env(role="api")
        self.assertFalse(settings.enabled)

    def test_durable_mode_fails_closed_without_required_dependencies(self):
        with patch.dict(os.environ, {"DURABLE_PROCESSING_ENABLED": "true"}, clear=True):
            with self.assertRaises(DurableConfigurationError):
                DurableSettings.from_env(role="api")

    def test_durable_internal_secret_has_a_minimum_length(self):
        env = {
            "DURABLE_PROCESSING_ENABLED": "true",
            "AI_PROCESSING_DATABASE_DSN": "test-dsn",
            "RABBITMQ_URL": "amqp://test",
            "AI_INTERNAL_SERVICE_SECRET": "too-short",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(DurableConfigurationError):
                DurableSettings.from_env(role="api")

    def test_legacy_mode_does_not_require_rabbitmq(self):
        with patch.dict(os.environ, {"DURABLE_PROCESSING_ENABLED": "false"}, clear=True):
            settings = DurableSettings.from_env(role="worker")
        self.assertIsNone(settings.rabbitmq_url)

    def test_same_injected_analysis_pipeline_is_deterministic(self):
        class Runtime:
            def is_ready(self):
                return True

            def run_scan(self, scan_id, media, *, deadline_seconds):
                result = {"score": 0.75, "details": {"status": "ok"}}
                return {"video": result, "audio": result, "image": result}, {
                    "video": {"result_received": True},
                    "audio": {"result_received": True},
                    "image": {"result_received": True},
                }

        class Model:
            def local_model_required(self):
                return False

            def is_loaded(self):
                return True

            def predict(self, vector):
                return {"readiness_score": 75, "confidence": 0.8, "risk_level": "stable"}

        request = AnalysisInput("same-scan", None, None, None)
        first = analyze_scan(request, runtime=Runtime(), ml_runtime=Model(), policy=ValidationPolicy(require_video=False, require_audio=False, require_image=False))
        second = analyze_scan(request, runtime=Runtime(), ml_runtime=Model(), policy=ValidationPolicy(require_video=False, require_audio=False, require_image=False))
        self.assertEqual(first.ok, second.ok)
        self.assertEqual(first.result, second.result)


class PhaseBContractTests(unittest.TestCase):
    def test_internal_service_auth_is_fail_closed_and_constant_contract_is_separate(self):
        import main

        secret = "s" * 32
        settings = SimpleNamespace(internal_secret=secret)
        with patch.object(main, "DURABLE_PROCESSING_ENABLED", False):
            with self.assertRaises(main.HTTPException) as disabled:
                main._require_durable_internal_access(None)
            self.assertEqual(disabled.exception.status_code, 404)

        with patch.object(main, "DURABLE_PROCESSING_ENABLED", True), patch.object(
            main, "_durable_settings", return_value=settings
        ):
            with self.assertRaises(main.HTTPException) as missing:
                main._require_durable_internal_access(None)
            self.assertEqual(missing.exception.status_code, 401)
            with self.assertRaises(main.HTTPException) as wrong:
                main._require_durable_internal_access("w" * 32)
            self.assertEqual(wrong.exception.status_code, 401)
            self.assertIsNone(main._require_durable_internal_access(secret))

    def test_public_status_is_hidden_when_durable_mode_is_disabled(self):
        import main

        with patch.object(main, "DURABLE_PROCESSING_ENABLED", False):
            with self.assertRaises(main.HTTPException) as error:
                main.durable_process_status("synthetic-scan", None)
        self.assertEqual(error.exception.status_code, 404)

    def test_public_status_reauthorizes_before_reading_durable_state(self):
        import main

        with patch.object(main, "DURABLE_PROCESSING_ENABLED", True), patch.object(
            main, "_authenticate_process_user", return_value="qa-user"
        ), patch.object(
            main,
            "_authorized_durable_scan",
            side_effect=main.HTTPException(status_code=404, detail="not_found"),
        ):
            with self.assertRaises(main.HTTPException) as error:
                main.durable_process_status("synthetic-scan", "Bearer qa-token")
        self.assertEqual(error.exception.status_code, 404)

    def test_duplicate_broker_delivery_does_not_commit_twice(self):
        repo = InMemoryProcessingRepository()
        spec = JobSpec(
            scan_id="duplicate-delivery-scan",
            processing_version="cie_v1_2",
            requester_user_id="user",
            member_id="member",
            business_profile_id="workspace",
            trace_id="trace",
        )
        repo.enqueue_job(
            spec,
            event_key="job:duplicate-delivery-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "trace"},
        )

        class Coordinator:
            def __init__(self):
                self.commits = 0
                self.token = None

            def claim(self, job_id, worker_id, processing_version, trace_id):
                job = repo.get_job(job_id)
                if job.status is JobStatus.COMPLETED:
                    return {"status": "completed"}
                claim = repo.claim_job(job_id, worker_id)
                self.token = claim.attempt.lease_token
                return {
                    "status": "processing",
                    "scan_id": job.scan_id,
                    "lease_token": str(self.token),
                    "analysis_context": {},
                }

            def asset(self, job_id, worker_id, token, role):
                return role.encode("ascii"), "audio/wav" if role == "audio" else "application/octet-stream"

            def commit(self, job_id, payload):
                self.commits += 1
                repo.complete_job(
                    job_id,
                    "worker",
                    UUID(payload["lease_token"]),
                    result_ref="result-1",
                )
                return {"status": "completed"}

            def fail(self, job_id, payload):
                raise AssertionError("duplicate delivery must not fail a completed job")

        coordinator = Coordinator()
        worker = DurableWorker(
            coordinator=coordinator,
            consumer=object(),
            worker_id="worker",
            analyzer=lambda request: AnalysisOutcome(
                ok=True,
                result={
                    "readiness_score": 75,
                    "confidence": 0.8,
                    "risk_level": "stable",
                    "explanation": "synthetic",
                    "suggested_action": "synthetic",
                },
            ),
            policy=ValidationPolicy(require_video=True, require_audio=True, require_image=True),
        )
        event = repo.claim_outbox_events("dispatcher")[0]
        message = ProcessingMessage.from_mapping(event.payload)
        self.assertEqual(worker.handle_message(message, {}), "ack")
        self.assertEqual(worker.handle_message(message, {}), "ack")
        self.assertEqual(coordinator.commits, 1)
        self.assertEqual(repo.get_job(event.job_id).status, JobStatus.COMPLETED)

    def test_worker_validation_failure_is_terminal_and_not_retried(self):
        token = uuid4()

        class Coordinator:
            def __init__(self):
                self.failure = None

            def claim(self, job_id, worker_id, processing_version, trace_id):
                return {"status": "processing", "scan_id": "scan", "lease_token": str(token), "analysis_context": {}}

            def asset(self, job_id, worker_id, lease_token, role):
                return role.encode(), "audio/wav" if role == "audio" else "application/octet-stream"

            def fail(self, job_id, payload):
                self.failure = payload
                return {"status": "failed_terminal"}

        coordinator = Coordinator()
        worker = DurableWorker(
            coordinator=coordinator,
            consumer=object(),
            worker_id="worker",
            analyzer=lambda request: AnalysisOutcome(
                ok=False,
                failure_code="low_quality_media",
                error_type="ValidationError",
            ),
            policy=ValidationPolicy(require_video=True, require_audio=True, require_image=True),
        )
        message = ProcessingMessage("event", uuid4(), "cie_v1_2", "trace")
        self.assertEqual(worker.handle_message(message, {}), "ack")
        self.assertIsNotNone(coordinator.failure)
        self.assertTrue(coordinator.failure["terminal_failure"])

    def test_local_durable_flow_runs_dispatch_worker_and_commit_once(self):
        repo = InMemoryProcessingRepository()
        spec = JobSpec(
            scan_id="full-durable-synthetic-scan",
            processing_version="cie_v1_2",
            requester_user_id="qa-user",
            member_id="qa-member",
            business_profile_id="qa-workspace",
            trace_id="qa-trace",
        )
        repo.enqueue_job(
            spec,
            event_key="job:full-durable-synthetic-scan:process:cie_v1_2",
            event_type="job.process",
            payload={"processing_version": "cie_v1_2", "trace_id": "qa-trace"},
        )

        class Publisher:
            is_connected = True

            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        publisher = Publisher()
        dispatcher = OutboxDispatcher(
            repository=repo,
            publisher=publisher,
            dispatcher_id="qa-dispatcher",
        )

        class Coordinator:
            def claim(self, job_id, worker_id, processing_version, trace_id):
                job = repo.get_job(job_id)
                if job.status is JobStatus.COMPLETED:
                    return {"status": "completed"}
                claim = repo.claim_job(job_id, worker_id)
                return {
                    "status": "processing",
                    "scan_id": job.scan_id,
                    "lease_token": str(claim.attempt.lease_token),
                    "analysis_context": {},
                }

            def asset(self, job_id, worker_id, lease_token, role):
                return role.encode(), "audio/wav" if role == "audio" else "application/octet-stream"

            def commit(self, job_id, payload):
                repo.complete_job(
                    job_id,
                    "qa-worker",
                    UUID(payload["lease_token"]),
                    result_ref="qa-result",
                )
                return {"status": "completed"}

            def fail(self, job_id, payload):
                raise AssertionError("synthetic durable analysis should not fail")

        worker = DurableWorker(
            coordinator=Coordinator(),
            consumer=object(),
            worker_id="qa-worker",
            analyzer=lambda request: AnalysisOutcome(
                ok=True,
                result={
                    "readiness_score": 80,
                    "confidence": 0.9,
                    "risk_level": "stable",
                    "explanation": "synthetic",
                    "suggested_action": "synthetic",
                },
            ),
            policy=ValidationPolicy(require_video=True, require_audio=True, require_image=True),
        )
        self.assertEqual(dispatcher.run_once(), 1)
        self.assertEqual(worker.handle_message(publisher.messages[0], {}), "ack")
        self.assertEqual(repo.get_job(publisher.messages[0].job_id).status, JobStatus.COMPLETED)
        self.assertEqual(len(repo.list_attempts(publisher.messages[0].job_id)), 1)


class PhaseBRabbitAdapterTests(unittest.TestCase):
    class _Channel:
        is_open = True

        def __init__(self, *, publish_result=True, body=b"bad"):
            self.publish_result = publish_result
            self.body = body
            self.published = []
            self.acks = []
            self.nacks = []
            self.callback = None

        def exchange_declare(self, **kwargs):
            pass

        def queue_declare(self, **kwargs):
            pass

        def queue_bind(self, **kwargs):
            pass

        def basic_qos(self, **kwargs):
            pass

        def confirm_delivery(self):
            pass

        def basic_publish(self, **kwargs):
            self.published.append(kwargs)
            return self.publish_result

        def basic_consume(self, *, queue, on_message_callback, auto_ack):
            self.callback = on_message_callback

        def start_consuming(self):
            self.callback(
                self,
                SimpleNamespace(delivery_tag=1),
                SimpleNamespace(headers={}),
                self.body,
            )

        def basic_ack(self, *, delivery_tag):
            self.acks.append(delivery_tag)

        def basic_nack(self, *, delivery_tag, requeue):
            self.nacks.append((delivery_tag, requeue))

        def stop_consuming(self):
            pass

    class _Connection:
        is_closed = False

        def __init__(self, channel):
            self._channel = channel

        def channel(self):
            return self._channel

        def close(self):
            self.is_closed = True

    class _Pika:
        class URLParameters:
            def __init__(self, url):
                self.url = url

        class BasicProperties:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

    def test_publisher_confirmation_failure_is_not_success(self):
        from rabbitmq_adapter import BrokerError, RabbitMQPublisher

        channel = self._Channel(publish_result=False)
        publisher = RabbitMQPublisher("amqp://synthetic")
        publisher._channel = channel
        with patch("rabbitmq_adapter._pika", return_value=self._Pika):
            with self.assertRaises(BrokerError):
                publisher.publish(ProcessingMessage("event", uuid4(), "v1", "trace"))

    def test_malformed_message_is_dead_lettered_and_acknowledged(self):
        from rabbitmq_adapter import RabbitMQConsumer

        channel = self._Channel(body=b"{not-json")
        connection = self._Connection(channel)
        consumer = RabbitMQConsumer("amqp://synthetic")
        with patch("rabbitmq_adapter._pika", return_value=self._Pika), patch(
            "rabbitmq_adapter.pika", self._Pika, create=True
        ):
            with patch.object(self._Pika, "BlockingConnection", return_value=connection, create=True):
                consumer._run_once(lambda message, headers: "ack")
        self.assertEqual(channel.acks, [1])
        self.assertEqual(channel.nacks, [])
        self.assertEqual(len(channel.published), 1)

    def test_retry_outcome_is_published_to_retry_queue_before_ack(self):
        from rabbitmq_adapter import RabbitMQConsumer

        message = ProcessingMessage("event", uuid4(), "v1", "trace").to_json()
        channel = self._Channel(body=message)
        connection = self._Connection(channel)
        consumer = RabbitMQConsumer("amqp://synthetic")
        with patch("rabbitmq_adapter._pika", return_value=self._Pika), patch.object(
            self._Pika, "BlockingConnection", return_value=connection, create=True
        ):
            consumer._run_once(lambda received, headers: "retry")
        self.assertEqual(channel.acks, [1])
        self.assertEqual(channel.nacks, [])
        self.assertEqual(len(channel.published), 1)
        self.assertEqual(channel.published[0]["routing_key"], "retry")


if __name__ == "__main__":
    unittest.main()
