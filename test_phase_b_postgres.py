from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import unittest
from uuid import uuid4

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

from processing import JobSpec, JobStatus, PostgresProcessingRepository


TEST_DSN = os.environ.get("AI_TEST_POSTGRES_DSN")
POSTGRES_AVAILABLE = psycopg is not None and bool(TEST_DSN)


@unittest.skipUnless(
    POSTGRES_AVAILABLE,
    "AI_TEST_POSTGRES_DSN is required for disposable PostgreSQL integration tests",
)
class PhaseBPostgresIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_processing_postgres import PostgresProcessingIntegrationTests

        migrations = sorted((Path(__file__).parent / "sql").glob("2026_09_*.sql"))
        PostgresProcessingIntegrationTests.migrations = migrations
        PostgresProcessingIntegrationTests._create_product_schema()
        PostgresProcessingIntegrationTests._apply_migrations()
        cls.repo = PostgresProcessingRepository(cls._connect)

    @staticmethod
    def _connect():
        return psycopg.connect(TEST_DSN)

    def setUp(self):
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                """
                TRUNCATE TABLE
                    public.scan_results,
                    public.wellness_scans,
                    ai_processing.ai_processing_effects,
                    ai_processing.processing_result_history,
                    ai_processing.ai_processing_outbox,
                    ai_processing.ai_processing_attempts,
                    ai_processing.ai_processing_jobs,
                    ai_processing.ai_worker_leases
                CASCADE
                """
            )

    def _spec(self, *, version="cie_v1_2", max_attempts=3):
        return JobSpec(
            scan_id=str(uuid4()),
            processing_version=version,
            requester_user_id=str(uuid4()),
            member_id="phase-b-member",
            business_profile_id=str(uuid4()),
            trace_id=str(uuid4()),
            max_attempts=max_attempts,
        )

    def _count(self, query, params=()):
        with psycopg.connect(TEST_DSN) as connection:
            return connection.execute(query, params).fetchone()[0]

    def test_enqueue_is_atomic_and_database_idempotent(self):
        spec = self._spec()
        event_key = "job:phase-b:process:one"

        def enqueue():
            return self.repo.enqueue_job(
                spec,
                event_key=event_key,
                event_type="job.process",
                payload={"processing_version": spec.processing_version, "trace_id": spec.trace_id},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: enqueue(), range(16)))
        self.assertEqual(len({result.job.id for result in results}), 1)
        self.assertEqual(sum(result.created for result in results), 1)
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM ai_processing.ai_processing_jobs WHERE scan_id = %s",
                (spec.scan_id,),
            ),
            1,
        )
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM ai_processing.ai_processing_outbox WHERE event_key = %s",
                (event_key,),
            ),
            1,
        )

    def test_retry_event_becomes_claimable_only_after_job_is_requeued(self):
        spec = self._spec()
        result = self.repo.enqueue_job(
            spec,
            event_key="job:phase-b:process:retry",
            event_type="job.process",
            payload={"processing_version": spec.processing_version, "trace_id": spec.trace_id},
        )
        initial_event = self.repo.claim_outbox_events("phase-b-dispatcher", event_type="job.process")[0]
        self.repo.mark_outbox_delivered(initial_event.id, "phase-b-dispatcher")
        claim = self.repo.claim_job(result.job.id, "phase-b-worker", lease_seconds=60)
        failed = self.repo.schedule_retry(
            result.job.id,
            "phase-b-worker",
            claim.attempt.lease_token,
            failure_code="temporary_failure",
            error_type="TimeoutError",
            retry_delay_seconds=0,
        )
        self.assertEqual(failed.status, JobStatus.FAILED_RETRYABLE)
        self.assertEqual(self.repo.claim_outbox_events("phase-b-dispatcher", event_type="job.process"), ())
        queued = self.repo.queue_due_retries(limit=1)
        self.assertEqual(queued[0].status, JobStatus.QUEUED)
        self.assertEqual(len(self.repo.claim_outbox_events("phase-b-dispatcher", event_type="job.process")), 1)

    def test_competing_claimers_receive_distinct_jobs_and_attempts(self):
        specs = [self._spec() for _ in range(2)]
        for spec in specs:
            self.repo.enqueue_job(
                spec,
                event_key=f"job:{spec.scan_id}:process",
                event_type="job.process",
                payload={"processing_version": spec.processing_version, "trace_id": spec.trace_id},
            )
        jobs = [self.repo.get_job_by_scan(spec.scan_id, spec.processing_version) for spec in specs]

        def claim(job, worker):
            return self.repo.claim_job(job.id, worker, lease_seconds=60)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda args: claim(*args), zip(jobs, ("worker-a", "worker-b"))))
        self.assertEqual({claim.attempt.attempt_number for claim in claims}, {1})


if __name__ == "__main__":
    unittest.main()
