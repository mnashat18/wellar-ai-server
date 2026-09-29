import concurrent.futures
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

try:
    import psycopg
except ImportError:  # pragma: no cover - dependency is declared in requirements
    psycopg = None

from processing import (
    DatabaseOperationError,
    InvalidJobTransition,
    JobSpec,
    JobStatus,
    LeaseConflict,
    OutboxConflict,
    PostgresProcessingRepository,
)
from processing_commit import (
    CommitProcessingResultRequest,
    PostgresProcessingResultCommitter,
    ValidatedResultPayload,
)


TEST_DSN = os.environ.get("AI_TEST_POSTGRES_DSN")
POSTGRES_AVAILABLE = psycopg is not None and bool(TEST_DSN)
UTC = timezone.utc


@unittest.skipUnless(
    POSTGRES_AVAILABLE,
    "AI_TEST_POSTGRES_DSN is required for disposable PostgreSQL integration tests",
)
class PostgresProcessingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.migrations = sorted(
            (Path(__file__).parent / "sql").glob("2026_09_*.sql")
        )
        cls._create_product_schema()
        cls._apply_migrations()
        cls.repo = PostgresProcessingRepository(cls._connect)
        cls.committer = PostgresProcessingResultCommitter(cls._connect)

    @classmethod
    def tearDownClass(cls):
        # The database is disposable and owned by the test runner.  No
        # production object is touched or dropped by this suite.
        pass

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

    @staticmethod
    def _connect():
        return psycopg.connect(TEST_DSN)

    @classmethod
    def _create_product_schema(cls):
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS public.wellness_scans (
                    id uuid PRIMARY KEY,
                    status varchar(64) NOT NULL,
                    "user" uuid,
                    member integer,
                    business_profile uuid,
                    completed_at timestamptz
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS public.scan_results (
                    id uuid PRIMARY KEY,
                    scan_id uuid UNIQUE,
                    readiness_score numeric,
                    confidence numeric,
                    risk_level varchar(64),
                    explanation text,
                    suggested_action text,
                    ai_model_version varchar(100)
                )
                """
            )

    @classmethod
    def _apply_migrations(cls):
        for migration in cls.migrations:
            sql = migration.read_text(encoding="utf-8")
            with psycopg.connect(TEST_DSN, autocommit=True) as connection:
                connection.execute(sql)

    def _new_job(
        self,
        *,
        scan_id: UUID | None = None,
        processing_version: str = "model-v1",
        requester_user_id: UUID | None = None,
        member_id: int = 7,
        business_profile_id: UUID | None = None,
        max_attempts: int = 3,
    ):
        scan_id = scan_id or uuid4()
        requester_user_id = requester_user_id or uuid4()
        business_profile_id = business_profile_id or uuid4()
        job = self.repo.create_or_get_job(
            JobSpec(
                scan_id=str(scan_id),
                processing_version=processing_version,
                requester_user_id=str(requester_user_id),
                member_id=str(member_id),
                business_profile_id=str(business_profile_id),
                max_attempts=max_attempts,
            )
        )
        return job, scan_id, requester_user_id, business_profile_id, member_id

    def _queue(self, job):
        return self.repo.transition_job(job.id, JobStatus.ACCEPTED, JobStatus.QUEUED)

    def _expire_job(self, job_id: UUID):
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                """
                UPDATE ai_processing.ai_processing_jobs
                SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second'
                WHERE id = %s
                """,
                (job_id,),
            )

    def _insert_scan(
        self,
        scan_id: UUID,
        requester_user_id: UUID,
        business_profile_id: UUID,
        member_id: int,
        *,
        status: str = "processing",
    ):
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                """
                INSERT INTO public.wellness_scans
                    (id, status, "user", member, business_profile)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    scan_id,
                    status,
                    requester_user_id,
                    member_id,
                    business_profile_id,
                ),
            )

    @staticmethod
    def _result():
        return ValidatedResultPayload(
            {
                "readiness_score": 75,
                "confidence": 0.8,
                "risk_level": "stable",
                "explanation": "synthetic validated explanation",
                "suggested_action": "synthetic validated action",
            }
        )

    def _commit_request(self, job, scan_id, token, *, version=None):
        version = version or job.processing_version
        return CommitProcessingResultRequest(
            job_id=job.id,
            scan_id=str(scan_id),
            processing_version=version,
            lease_token=token,
            result_id=uuid4(),
            result=self._result(),
        )

    def _scalar(self, query, params=()):
        with psycopg.connect(TEST_DSN) as connection:
            row = connection.execute(query, params).fetchone()
            return row[0] if row else None

    def test_migrations_apply_and_roles_are_scoped(self):
        self._apply_migrations()
        self.assertEqual(
            self._scalar(
                "SELECT to_regclass('ai_processing.processing_result_history')"
            ),
            "ai_processing.processing_result_history",
        )
        self.assertEqual(
            self._scalar(
                "SELECT to_regprocedure('ai_processing.commit_processing_result(uuid,uuid,character varying,uuid,uuid,integer,numeric,character varying,text,text,character varying)')"
            ),
            "ai_processing.commit_processing_result(uuid,uuid,character varying,uuid,uuid,integer,numeric,character varying,text,text,character varying)",
        )
        self.assertTrue(
            self._scalar(
                "SELECT has_function_privilege('ai_processing_commit', 'ai_processing.commit_processing_result(uuid,uuid,character varying,uuid,uuid,integer,numeric,character varying,text,text,character varying)', 'EXECUTE')"
            )
        )
        self.assertFalse(
            self._scalar(
                "SELECT has_table_privilege('ai_processing_commit', 'public.scan_results', 'UPDATE')"
            )
        )
        self.assertFalse(
            self._scalar(
                "SELECT has_table_privilege('ai_processing_runtime', 'public.wellness_scans', 'UPDATE')"
            )
        )

    def test_concurrent_create_or_get_job_is_database_idempotent(self):
        scan_id = uuid4()
        spec = JobSpec(
            scan_id=str(scan_id),
            processing_version="model-v1",
            requester_user_id=str(uuid4()),
            member_id="7",
            business_profile_id=str(uuid4()),
        )

        def create_one():
            return PostgresProcessingRepository(self._connect).create_or_get_job(spec).id

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(executor.map(lambda _: create_one(), range(24)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM ai_processing.ai_processing_jobs WHERE scan_id = %s AND processing_version = %s",
                (str(scan_id), "model-v1"),
            ),
            1,
        )

    def test_concurrent_claim_has_one_owner_and_one_attempt(self):
        job, *_ = self._new_job()
        self._queue(job)

        def claim(worker):
            return PostgresProcessingRepository(self._connect).claim_job(
                job.id, worker, lease_seconds=60
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, ("worker-a", "worker-b")))
        winners = [claim for claim in claims if claim is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(
            self._scalar(
                "SELECT attempt_count FROM ai_processing.ai_processing_jobs WHERE id = %s",
                (job.id,),
            ),
            1,
        )
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM ai_processing.ai_processing_attempts WHERE job_id = %s",
                (job.id,),
            ),
            1,
        )

    def test_claim_and_attempt_insert_roll_back_together(self):
        job, *_ = self._new_job()
        self._queue(job)

        class FaultCursor:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, statement, params=None):
                if "INSERT INTO ai_processing.ai_processing_attempts" in statement:
                    raise RuntimeError("injected attempt insert failure")
                return self._cursor.execute(statement, params)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        class FaultConnection:
            def __init__(self, connection):
                self._connection = connection

            def cursor(self, *args, **kwargs):
                return FaultCursor(self._connection.cursor(*args, **kwargs))

            def __getattr__(self, name):
                return getattr(self._connection, name)

            def __setattr__(self, name, value):
                if name == "_connection":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._connection, name, value)

        def faulty_connection():
            return FaultConnection(psycopg.connect(TEST_DSN))

        with self.assertRaises(DatabaseOperationError):
            PostgresProcessingRepository(faulty_connection).claim_job(
                job.id, "worker-fault", lease_seconds=60
            )
        current = self.repo.get_job(job.id)
        self.assertEqual(current.status, JobStatus.QUEUED)
        self.assertEqual(current.attempt_count, 0)
        self.assertEqual(self.repo.list_attempts(job.id), ())

    def test_postgres_clock_rejects_skewed_expired_heartbeat(self):
        job, *_ = self._new_job()
        self._queue(job)
        claim = self.repo.claim_job(
            job.id,
            "worker-clock",
            now=datetime(2099, 1, 1, tzinfo=UTC),
            lease_seconds=1,
        )
        renewed = self.repo.heartbeat(
            job.id,
            "worker-clock",
            claim.attempt.lease_token,
            now=datetime(2000, 1, 1, tzinfo=UTC),
            lease_seconds=60,
        )
        self.assertGreater(renewed.lease_expires_at, renewed.heartbeat_at)
        self._expire_job(job.id)
        with self.assertRaises(LeaseConflict):
            self.repo.heartbeat(
                job.id,
                "worker-clock",
                claim.attempt.lease_token,
                now=datetime(2000, 1, 1, tzinfo=UTC),
                lease_seconds=60,
            )

    def test_expired_lease_reclaim_denies_stale_worker(self):
        job, *_ = self._new_job()
        self._queue(job)
        first = self.repo.claim_job(job.id, "worker-a", lease_seconds=60)
        self._expire_job(job.id)
        recovered = self.repo.recover_expired_leases(now=datetime(2099, 1, 1, tzinfo=UTC))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(self.repo.queue_due_retries(now=datetime(2099, 1, 1, tzinfo=UTC))[0].status, JobStatus.QUEUED)
        second = self.repo.claim_job(job.id, "worker-b", now=datetime(2000, 1, 1, tzinfo=UTC))
        with self.assertRaises(LeaseConflict):
            self.repo.complete_job(job.id, "worker-a", first.attempt.lease_token, now=datetime(2000, 1, 1, tzinfo=UTC))
        self.repo.complete_job(job.id, "worker-b", second.attempt.lease_token)

    def test_max_attempts_and_worker_liveness(self):
        job, *_ = self._new_job(max_attempts=2)
        self._queue(job)
        for number in (1, 2):
            claim = self.repo.claim_job(job.id, f"worker-{number}", lease_seconds=60)
            self._expire_job(job.id)
            if number == 2:
                recovered = self.repo.recover_expired_leases()
                self.assertEqual(recovered[0].status, JobStatus.FAILED_TERMINAL)
            else:
                self.repo.recover_expired_leases()
                self.repo.queue_due_retries()
        self.assertEqual(self.repo.get_job(job.id).attempt_count, 2)
        self.assertEqual(len(self.repo.list_attempts(job.id)), 2)

        worker = self.repo.register_worker("worker-live", "test-worker", capacity=2)
        heartbeat = self.repo.heartbeat_worker(
            worker.worker_id, worker.lease_token, active_slots=1
        )
        self.assertEqual(heartbeat.active_slots, 1)
        self.assertTrue(
            self.repo.set_worker_draining(
                worker.worker_id, worker.lease_token, draining=True
            ).draining
        )
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                "UPDATE ai_processing.ai_worker_leases SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE worker_id = %s",
                (worker.worker_id,),
            )
        with self.assertRaises(LeaseConflict):
            self.repo.heartbeat_worker(
                worker.worker_id, worker.lease_token, active_slots=0
            )
        self.assertEqual(len(self.repo.reap_expired_workers()), 1)

    def test_database_trigger_rejects_illegal_transition(self):
        job, *_ = self._new_job()
        with psycopg.connect(TEST_DSN) as connection:
            with self.assertRaises(psycopg.Error):
                connection.execute(
                    "UPDATE ai_processing.ai_processing_jobs SET status = 'completed' WHERE id = %s",
                    (job.id,),
                )
            connection.rollback()
        self.assertEqual(self.repo.get_job(job.id).status, JobStatus.ACCEPTED)

    def test_outbox_competing_claimers_and_expired_release(self):
        job, *_ = self._new_job()
        event = self.repo.add_outbox_event(
            event_key=f"test:{job.id}:dispatch",
            job_id=job.id,
            event_type="job.dispatch",
            payload={"job_id": str(job.id), "status": "accepted"},
        )

        def claim(dispatcher):
            return PostgresProcessingRepository(self._connect).claim_outbox_events(
                dispatcher, lock_seconds=1
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(claim, ("dispatcher-a", "dispatcher-b")))
        winners = [rows for rows in claims if rows]
        self.assertEqual(len(winners), 1)
        owner = winners[0][0].locked_by
        time.sleep(1.2)
        with self.assertRaises(OutboxConflict):
            self.repo.release_outbox_event(
                event.id,
                owner,
                failure_code="publisher_unavailable",
                retry_delay_seconds=0,
            )
        redelivered = self.repo.claim_outbox_events("dispatcher-recovery")
        self.assertEqual(len(redelivered), 1)
        self.repo.mark_outbox_delivered(redelivered[0].id, "dispatcher-recovery")
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM ai_processing.ai_processing_outbox WHERE event_key = %s AND delivered_at IS NOT NULL",
                (event.event_key,),
            ),
            1,
        )

    def test_atomic_final_commit_persistence_and_duplicate(self):
        job, scan_id, user_id, workspace_id, member_id = self._new_job()
        self._insert_scan(scan_id, user_id, workspace_id, member_id)
        self._queue(job)
        claim = self.repo.claim_job(job.id, "worker-commit")
        request = self._commit_request(job, scan_id, claim.attempt.lease_token)
        response = self.committer.commit_processing_result(request)
        self.assertEqual(response.status, "completed")
        self.assertIsNotNone(response.result_ref)
        self.assertEqual(
            self._scalar("SELECT status FROM ai_processing.ai_processing_jobs WHERE id = %s", (job.id,)),
            "completed",
        )
        self.assertEqual(
            self._scalar("SELECT status FROM public.wellness_scans WHERE id = %s", (scan_id,)),
            "completed",
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.processing_result_history WHERE scan_id = %s", (scan_id,)),
            1,
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM public.scan_results WHERE scan_id = %s", (scan_id,)),
            1,
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.ai_processing_effects WHERE job_id = %s", (job.id,)),
            1,
        )
        duplicate = self.committer.commit_processing_result(request)
        self.assertEqual(duplicate.status, "completed")
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.processing_result_history WHERE scan_id = %s", (scan_id,)),
            1,
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.ai_processing_outbox WHERE job_id = %s", (job.id,)),
            1,
        )
        with psycopg.connect(TEST_DSN) as connection:
            with self.assertRaises(psycopg.Error):
                connection.execute(
                    "UPDATE ai_processing.processing_result_history SET explanation = 'tampered' WHERE job_id = %s",
                    (job.id,),
                )
            connection.rollback()

    def test_versioned_reprocessing_keeps_history_and_current_projection(self):
        job, scan_id, user_id, workspace_id, member_id = self._new_job()
        self._insert_scan(scan_id, user_id, workspace_id, member_id)
        self._queue(job)
        first = self.repo.claim_job(job.id, "worker-v1")
        self.committer.commit_processing_result(
            self._commit_request(job, scan_id, first.attempt.lease_token)
        )

        second_job, *_ = self._new_job(
            scan_id=scan_id,
            processing_version="model-v2",
            requester_user_id=user_id,
            member_id=member_id,
            business_profile_id=workspace_id,
        )
        self._queue(second_job)
        second = self.repo.claim_job(second_job.id, "worker-v2")
        self.committer.commit_processing_result(
            self._commit_request(
                second_job,
                scan_id,
                second.attempt.lease_token,
                version="model-v2",
            )
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.processing_result_history WHERE scan_id = %s", (scan_id,)),
            2,
        )
        self.assertEqual(
            self._scalar("SELECT current_processing_version FROM public.wellness_scans WHERE id = %s", (scan_id,)),
            "model-v2",
        )
        self.assertEqual(
            self._scalar("SELECT ai_model_version FROM public.scan_results WHERE scan_id = %s", (scan_id,)),
            "model-v2",
        )

    def test_final_commit_rejects_stale_lease_and_wrong_binding(self):
        job, scan_id, user_id, workspace_id, member_id = self._new_job()
        self._insert_scan(scan_id, user_id, workspace_id, member_id)
        self._queue(job)
        first = self.repo.claim_job(job.id, "worker-stale")
        self._expire_job(job.id)
        self.repo.recover_expired_leases()
        self.repo.queue_due_retries()
        second = self.repo.claim_job(job.id, "worker-current")
        with self.assertRaises(DatabaseOperationError):
            self.committer.commit_processing_result(
                self._commit_request(job, scan_id, first.attempt.lease_token)
            )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM public.scan_results WHERE scan_id = %s", (scan_id,)),
            0,
        )
        self.repo.schedule_retry(
            job.id,
            "worker-current",
            second.attempt.lease_token,
            failure_code="test_abort",
            error_type="TestError",
            retry_delay_seconds=0,
        )

        bad_job, bad_scan, bad_user, bad_workspace, bad_member = self._new_job()
        self._insert_scan(bad_scan, uuid4(), bad_workspace, bad_member)
        self._queue(bad_job)
        bad_claim = self.repo.claim_job(bad_job.id, "worker-binding")
        with self.assertRaises(DatabaseOperationError):
            self.committer.commit_processing_result(
                self._commit_request(bad_job, bad_scan, bad_claim.attempt.lease_token)
            )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.processing_result_history WHERE scan_id = %s", (bad_scan,)),
            0,
        )
        self.assertEqual(self.repo.get_job(bad_job.id).status, JobStatus.PROCESSING)

    def test_final_commit_rolls_back_after_product_write_failure(self):
        job, scan_id, user_id, workspace_id, member_id = self._new_job()
        self._insert_scan(scan_id, user_id, workspace_id, member_id)
        self._queue(job)
        claim = self.repo.claim_job(job.id, "worker-rollback")
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute(
                """
                CREATE OR REPLACE FUNCTION public.phase_a_test_fail_scan_result()
                RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'test product write failure';
                END;
                $$
                """
            )
            connection.execute(
                """
                CREATE TRIGGER phase_a_test_fail_scan_result_trigger
                BEFORE INSERT OR UPDATE ON public.scan_results
                FOR EACH ROW EXECUTE FUNCTION public.phase_a_test_fail_scan_result()
                """
            )
        try:
            with self.assertRaises(DatabaseOperationError):
                self.committer.commit_processing_result(
                    self._commit_request(job, scan_id, claim.attempt.lease_token)
                )
        finally:
            with psycopg.connect(TEST_DSN, autocommit=True) as connection:
                connection.execute(
                    "DROP TRIGGER phase_a_test_fail_scan_result_trigger ON public.scan_results"
                )
                connection.execute("DROP FUNCTION public.phase_a_test_fail_scan_result()")
        self.assertEqual(self.repo.get_job(job.id).status, JobStatus.PROCESSING)
        self.assertEqual(
            self._scalar("SELECT count(*) FROM ai_processing.processing_result_history WHERE scan_id = %s", (scan_id,)),
            0,
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM public.scan_results WHERE scan_id = %s", (scan_id,)),
            0,
        )
        self.assertEqual(
            self._scalar("SELECT status FROM public.wellness_scans WHERE id = %s", (scan_id,)),
            "processing",
        )


if __name__ == "__main__":
    unittest.main()
