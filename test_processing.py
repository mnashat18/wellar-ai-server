import concurrent.futures
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from processing import (
    InMemoryProcessingRepository,
    InvalidJobTransition,
    JobSpec,
    JobStatus,
    LeaseConflict,
    OutboxConflict,
)
from processing_commit import (
    CommitProcessingResultRequest,
    ProcessingResultCommitter,
    ValidatedResultPayload,
)
from processing import (
    DatabaseOperationError,
    PostgresProcessingRepository,
    _retry_transaction_method,
)


UTC = timezone.utc


class PhaseAProcessingTests(unittest.TestCase):
    def setUp(self):
        self.repo = InMemoryProcessingRepository()
        self.start = datetime(2026, 1, 1, tzinfo=UTC)

    def _job(self, *, version="model-v1", max_attempts=3):
        return self.repo.create_or_get_job(
            JobSpec(
                scan_id="synthetic-scan",
                processing_version=version,
                requester_user_id="synthetic-user",
                member_id="synthetic-member",
                business_profile_id="synthetic-workspace",
                trace_id="trace-1",
                max_attempts=max_attempts,
                created_at=self.start,
            )
        )

    def _queued_job(self, **kwargs):
        job = self._job(**kwargs)
        return self.repo.transition_job(
            job.id,
            JobStatus.ACCEPTED,
            JobStatus.QUEUED,
            now=self.start,
        )

    def test_migration_contains_required_tables_and_constraints(self):
        migration = Path(__file__).parent / "sql" / "2026_09_05_phase_a_ai_processing.sql"
        sql = migration.read_text(encoding="utf-8")
        for table in (
            "ai_processing_jobs",
            "ai_processing_attempts",
            "ai_processing_outbox",
            "ai_worker_leases",
        ):
            self.assertIn(f"ai_processing.{table}", sql)
        self.assertIn("UNIQUE (scan_id, processing_version)", sql)
        self.assertNotIn("UNIQUE (scan_id) WHERE status = 'completed'", sql)

    def test_create_or_get_job_is_idempotent(self):
        first = self._job()
        second = self._job()
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.status, JobStatus.ACCEPTED)

    def test_concurrent_duplicate_creation_returns_one_job(self):
        def create():
            return self._job().id

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(executor.map(lambda _: create(), range(32)))
        self.assertEqual({job_id for job_id in ids}, {ids[0]})

    def test_new_processing_version_is_a_distinct_job(self):
        first = self._job(version="model-v1")
        second = self._job(version="model-v2")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.scan_id, second.scan_id)

    def test_state_machine_accepts_only_defined_transitions(self):
        job = self._job()
        queued = self.repo.transition_job(job.id, JobStatus.ACCEPTED, JobStatus.QUEUED, now=self.start)
        claim = self.repo.claim_job(job.id, "worker-a", now=self.start)
        self.assertEqual(queued.status, JobStatus.QUEUED)
        self.assertEqual(claim.job.status, JobStatus.PROCESSING)
        completed = self.repo.complete_job(job.id, "worker-a", claim.attempt.lease_token, now=self.start)
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        with self.assertRaises(InvalidJobTransition):
            self.repo.transition_job(job.id, JobStatus.COMPLETED, JobStatus.QUEUED, now=self.start)

    def test_atomic_claim_and_heartbeat(self):
        job = self._queued_job()
        claim = self.repo.claim_job(job.id, "worker-a", now=self.start, lease_seconds=60)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.attempt.attempt_number, 1)
        renewed = self.repo.heartbeat(
            job.id,
            "worker-a",
            claim.attempt.lease_token,
            now=self.start + timedelta(seconds=10),
            lease_seconds=60,
        )
        self.assertEqual(renewed.lease_owner, "worker-a")
        self.assertGreater(renewed.lease_expires_at, claim.job.lease_expires_at)

    def test_expired_lease_is_recovered_and_reclaimed(self):
        job = self._queued_job()
        first = self.repo.claim_job(job.id, "worker-a", now=self.start, lease_seconds=60)
        recovered = self.repo.recover_expired_leases(now=self.start + timedelta(seconds=61))
        self.assertEqual(recovered[0].status, JobStatus.FAILED_RETRYABLE)
        queued = self.repo.queue_due_retries(now=self.start + timedelta(seconds=61))
        self.assertEqual(queued[0].status, JobStatus.QUEUED)
        second = self.repo.claim_job(job.id, "worker-b", now=self.start + timedelta(seconds=61))
        self.assertNotEqual(first.attempt.lease_token, second.attempt.lease_token)
        self.assertEqual(second.attempt.attempt_number, 2)

    def test_stale_worker_cannot_complete_after_reassignment(self):
        job = self._queued_job()
        first = self.repo.claim_job(job.id, "worker-a", now=self.start, lease_seconds=60)
        recovery_time = self.start + timedelta(seconds=61)
        self.repo.recover_expired_leases(now=recovery_time)
        self.repo.queue_due_retries(now=recovery_time)
        second = self.repo.claim_job(job.id, "worker-b", now=recovery_time)
        with self.assertRaises(LeaseConflict):
            self.repo.complete_job(job.id, "worker-a", first.attempt.lease_token, now=recovery_time)
        self.assertEqual(self.repo.get_job(job.id).lease_owner, "worker-b")
        self.repo.complete_job(job.id, "worker-b", second.attempt.lease_token, now=recovery_time)

    def test_heartbeat_after_expiry_is_denied(self):
        job = self._queued_job()
        claim = self.repo.claim_job(job.id, "worker-a", now=self.start, lease_seconds=60)
        with self.assertRaises(LeaseConflict):
            self.repo.heartbeat(
                job.id,
                "worker-a",
                claim.attempt.lease_token,
                now=self.start + timedelta(seconds=61),
            )

    def test_attempt_numbering_and_max_attempts(self):
        job = self._queued_job(max_attempts=3)
        for number in (1, 2, 3):
            claim = self.repo.claim_job(
                job.id,
                f"worker-{number}",
                now=self.start + timedelta(seconds=number),
            )
            self.assertEqual(claim.attempt.attempt_number, number)
            result = self.repo.schedule_retry(
                job.id,
                f"worker-{number}",
                claim.attempt.lease_token,
                failure_code="dependency_timeout",
                error_type="TimeoutError",
                retry_delay_seconds=1,
                now=self.start + timedelta(seconds=number),
            )
            if number < 3:
                self.assertEqual(result.status, JobStatus.FAILED_RETRYABLE)
                self.repo.queue_due_retries(now=self.start + timedelta(seconds=number + 1))
            else:
                self.assertEqual(result.status, JobStatus.FAILED_TERMINAL)
        self.assertEqual(len(self.repo.list_attempts(job.id)), 3)

    def test_retry_is_not_queued_before_schedule(self):
        job = self._queued_job()
        claim = self.repo.claim_job(job.id, "worker-a", now=self.start)
        retry_at = self.start + timedelta(seconds=30)
        self.repo.schedule_retry(
            job.id,
            "worker-a",
            claim.attempt.lease_token,
            failure_code="temporary_dependency",
            error_type="ConnectionError",
            retry_delay_seconds=30,
            now=self.start,
        )
        self.assertEqual(len(self.repo.queue_due_retries(now=self.start + timedelta(seconds=29))), 0)
        self.assertEqual(
            self.repo.queue_due_retries(now=retry_at)[0].status,
            JobStatus.QUEUED,
        )

    def test_outbox_unique_claim_and_publisher_failure_retry(self):
        job = self._job()
        event = self.repo.add_outbox_event(
            event_key="job:synthetic-scan:model-v1:dispatch",
            job_id=job.id,
            event_type="job.dispatch",
            payload={
                "job_id": str(job.id),
                "scan_id": job.scan_id,
                "processing_version": job.processing_version,
                "status": "accepted",
            },
            available_at=self.start,
        )
        duplicate = self.repo.add_outbox_event(
            event_key=event.event_key,
            job_id=job.id,
            event_type=event.event_type,
            payload=event.payload,
            available_at=self.start,
        )
        self.assertEqual(event.id, duplicate.id)
        claimed = self.repo.claim_outbox_events("dispatcher-a", now=self.start)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(self.repo.claim_outbox_events("dispatcher-b", now=self.start), ())
        released = self.repo.release_outbox_event(
            event.id,
            "dispatcher-a",
            failure_code="publisher_unavailable",
            retry_delay_seconds=5,
            now=self.start,
        )
        self.assertEqual(released.attempt_count, 1)
        claimed_again = self.repo.claim_outbox_events(
            "dispatcher-b", now=self.start + timedelta(seconds=5)
        )
        delivered = self.repo.mark_outbox_delivered(
            claimed_again[0].id,
            "dispatcher-b",
            now=self.start + timedelta(seconds=5),
        )
        self.assertIsNotNone(delivered.delivered_at)

    def test_outbox_conflicting_duplicate_is_rejected(self):
        job = self._job()
        self.repo.add_outbox_event(
            event_key="same-event",
            job_id=job.id,
            event_type="job.dispatch",
            payload={"status": "accepted"},
            available_at=self.start,
        )
        with self.assertRaises(OutboxConflict):
            self.repo.add_outbox_event(
                event_key="same-event",
                job_id=job.id,
                event_type="job.dispatch",
                payload={"status": "completed"},
                available_at=self.start,
            )

    def test_transaction_rolls_back_on_database_failure_simulation(self):
        with self.assertRaisesRegex(RuntimeError, "simulated database failure"):
            with self.repo.transaction():
                self._job()
                raise RuntimeError("simulated database failure")
        with self.assertRaises(Exception):
            self.repo.get_job(next(iter(self.repo._jobs), uuid4()))

    def test_postgres_repository_wraps_database_failure_and_rolls_back(self):
        class FakeConnection:
            def __init__(self):
                self.autocommit = True
                self.rolled_back = False
                self.closed = False
                self.statements = []

            class Cursor:
                def __init__(self, owner):
                    self.owner = owner

                def execute(self, statement):
                    self.statement = statement
                    self.owner.statements.append(statement)

                def close(self):
                    pass

            def cursor(self):
                return self.Cursor(self)

            def commit(self):
                raise RuntimeError("simulated database failure")

            def rollback(self):
                self.rolled_back = True

            def close(self):
                self.closed = True

        connection = FakeConnection()
        repository = PostgresProcessingRepository(lambda: connection)
        with self.assertRaises(DatabaseOperationError):
            with repository._transaction():
                pass
        self.assertFalse(connection.autocommit)
        self.assertEqual(connection.statements, ["BEGIN"])
        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)

    def test_retry_policy_retries_only_serialization_and_deadlock(self):
        class RetryProbe:
            def __init__(self, sqlstate):
                self.calls = 0
                self.sqlstate = sqlstate

            @_retry_transaction_method
            def run(self):
                self.calls += 1
                if self.calls < 3:
                    raise DatabaseOperationError(
                        "transient transaction failure", sqlstate=self.sqlstate
                    )
                return "ok"

        for sqlstate in ("40001", "40P01"):
            probe = RetryProbe(sqlstate)
            with patch("processing.time.sleep") as sleep:
                self.assertEqual(probe.run(), "ok")
            self.assertEqual(probe.calls, 3)
            self.assertEqual(sleep.call_count, 2)

        probe = RetryProbe("23505")
        with patch("processing.time.sleep") as sleep:
            with self.assertRaises(DatabaseOperationError):
                probe.run()
        self.assertEqual(probe.calls, 1)
        sleep.assert_not_called()

    def test_completed_job_is_idempotent(self):
        job = self._queued_job()
        claim = self.repo.claim_job(job.id, "worker-a", now=self.start)
        first = self.repo.complete_job(
            job.id, "worker-a", claim.attempt.lease_token, result_ref="result-1", now=self.start
        )
        second = self.repo.complete_job(
            job.id, "worker-b", uuid4(), result_ref="result-2", now=self.start
        )
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.result_ref, "result-1")

    def test_result_commit_contract_requires_lease_and_validated_payload(self):
        class FakeCommitter:
            def __init__(self):
                self.received = None

            def commit_processing_result(self, request):
                self.received = request
                return {"status": "completed"}

        fake: ProcessingResultCommitter = FakeCommitter()
        request = CommitProcessingResultRequest(
            job_id=uuid4(),
            scan_id="synthetic-scan",
            processing_version="model-v1",
            lease_token=uuid4(),
            result_id=uuid4(),
            result=ValidatedResultPayload(
                {
                    "readiness_score": 75,
                    "confidence": 0.8,
                    "risk_level": "stable",
                    "explanation": "synthetic validated explanation",
                    "suggested_action": "synthetic validated action",
                }
            ),
        )
        response = fake.commit_processing_result(request)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(fake.received, request)

    def test_result_payload_rejects_non_finite_or_invalid_values(self):
        with self.assertRaises(ValueError):
            ValidatedResultPayload(
                {
                    "readiness_score": 75,
                    "confidence": float("nan"),
                    "risk_level": "stable",
                    "explanation": "explanation",
                    "suggested_action": "action",
                }
            )
        with self.assertRaises(ValueError):
            ValidatedResultPayload(
                {
                    "readiness_score": 101,
                    "confidence": 0.8,
                    "risk_level": "stable",
                    "explanation": "explanation",
                    "suggested_action": "action",
                }
            )
        with self.assertRaises(ValueError):
            ValidatedResultPayload(
                {
                    "readiness_score": 75,
                    "confidence": 0.8,
                    "risk_level": "unsafe",
                    "explanation": "explanation",
                    "suggested_action": "action",
                }
            )


if __name__ == "__main__":
    unittest.main()
