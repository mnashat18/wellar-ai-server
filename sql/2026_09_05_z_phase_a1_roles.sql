BEGIN;

-- Administrative migration. These are NOLOGIN capability roles only; no
-- credentials or service-user assignments are created here.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_processing_owner') THEN
        CREATE ROLE ai_processing_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_processing_runtime') THEN
        CREATE ROLE ai_processing_runtime NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_processing_commit') THEN
        CREATE ROLE ai_processing_commit NOLOGIN;
    END IF;
END $$;

ALTER SCHEMA ai_processing OWNER TO ai_processing_owner;

ALTER TABLE ai_processing.ai_processing_jobs OWNER TO ai_processing_owner;
ALTER TABLE ai_processing.ai_processing_attempts OWNER TO ai_processing_owner;
ALTER TABLE ai_processing.ai_processing_outbox OWNER TO ai_processing_owner;
ALTER TABLE ai_processing.ai_worker_leases OWNER TO ai_processing_owner;

REVOKE ALL ON SCHEMA ai_processing FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA ai_processing FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA ai_processing FROM PUBLIC;

GRANT USAGE ON SCHEMA ai_processing TO ai_processing_runtime;
GRANT USAGE ON SCHEMA ai_processing TO ai_processing_commit;

GRANT SELECT, INSERT, UPDATE ON
    ai_processing.ai_processing_jobs,
    ai_processing.ai_processing_attempts,
    ai_processing.ai_processing_outbox,
    ai_processing.ai_worker_leases
TO ai_processing_runtime;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ai_processing
TO ai_processing_runtime;

-- The owner role is the only role that will own the SECURITY DEFINER
-- completion function and its product-table capability.
GRANT USAGE ON SCHEMA public TO ai_processing_owner;
GRANT SELECT, INSERT, UPDATE ON public.wellness_scans, public.scan_results
TO ai_processing_owner;

COMMIT;
