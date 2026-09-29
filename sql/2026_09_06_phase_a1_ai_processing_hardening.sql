BEGIN;

ALTER TABLE public.wellness_scans
    ADD COLUMN IF NOT EXISTS current_processing_version varchar(100);

CREATE TABLE IF NOT EXISTS ai_processing.processing_result_history (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES ai_processing.ai_processing_jobs(id),
    scan_id uuid NOT NULL,
    processing_version varchar(100) NOT NULL,
    readiness_score integer NOT NULL,
    confidence numeric NOT NULL,
    risk_level varchar(32) NOT NULL,
    explanation text NOT NULL,
    suggested_action text NOT NULL,
    result_hash varchar(128) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT processing_result_history_score_check
        CHECK (readiness_score BETWEEN 0 AND 100),
    CONSTRAINT processing_result_history_confidence_check
        CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT processing_result_history_risk_check
        CHECK (risk_level IN (
            'stable', 'low_focus', 'elevated_fatigue', 'high_risk', 'unknown'
        )),
    CONSTRAINT processing_result_history_job_unique
        UNIQUE (job_id),
    CONSTRAINT processing_result_history_scan_version_unique
        UNIQUE (scan_id, processing_version)
);

CREATE INDEX IF NOT EXISTS processing_result_history_scan_idx
    ON ai_processing.processing_result_history (scan_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_processing.ai_processing_effects (
    effect_key varchar(255) PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES ai_processing.ai_processing_jobs(id),
    effect_type varchar(100) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ai_processing_effects_job_idx
    ON ai_processing.ai_processing_effects (job_id, effect_type);

ALTER TABLE ai_processing.processing_result_history OWNER TO ai_processing_owner;
ALTER TABLE ai_processing.ai_processing_effects OWNER TO ai_processing_owner;
REVOKE ALL ON TABLE
    ai_processing.processing_result_history,
    ai_processing.ai_processing_effects
FROM PUBLIC;

CREATE OR REPLACE FUNCTION ai_processing.prevent_processing_history_mutation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, ai_processing
AS $$
BEGIN
    RAISE EXCEPTION USING
        ERRCODE = 'P0001',
        MESSAGE = 'processing result history is immutable';
END;
$$;

DROP TRIGGER IF EXISTS processing_result_history_immutable_trigger
    ON ai_processing.processing_result_history;

CREATE TRIGGER processing_result_history_immutable_trigger
BEFORE UPDATE OR DELETE ON ai_processing.processing_result_history
FOR EACH ROW
EXECUTE FUNCTION ai_processing.prevent_processing_history_mutation();

CREATE OR REPLACE FUNCTION ai_processing.enforce_job_status_transition()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, ai_processing
AS $$
BEGIN
    IF NEW.status IS DISTINCT FROM OLD.status
       AND NOT (
            (OLD.status = 'accepted' AND NEW.status = 'queued')
         OR (OLD.status = 'queued' AND NEW.status = 'processing')
         OR (OLD.status = 'processing' AND NEW.status IN (
                'completed', 'failed_retryable', 'failed_terminal'
            ))
         OR (OLD.status = 'failed_retryable' AND NEW.status = 'queued')
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'illegal ai_processing job status transition';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS ai_processing_jobs_status_transition_trigger
    ON ai_processing.ai_processing_jobs;

CREATE TRIGGER ai_processing_jobs_status_transition_trigger
BEFORE UPDATE OF status ON ai_processing.ai_processing_jobs
FOR EACH ROW
EXECUTE FUNCTION ai_processing.enforce_job_status_transition();

CREATE OR REPLACE FUNCTION ai_processing.commit_processing_result(
    p_job_id uuid,
    p_scan_id uuid,
    p_processing_version varchar(100),
    p_lease_token uuid,
    p_result_id uuid,
    p_readiness_score integer,
    p_confidence numeric,
    p_risk_level varchar(32),
    p_explanation text,
    p_suggested_action text,
    p_result_hash varchar(128)
)
RETURNS TABLE(status varchar, result_id uuid, history_id uuid)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, ai_processing
AS $$
DECLARE
    v_job ai_processing.ai_processing_jobs%ROWTYPE;
    v_scan public.wellness_scans%ROWTYPE;
    v_history ai_processing.processing_result_history%ROWTYPE;
    v_result_id uuid;
    v_history_id uuid;
    v_attempt_rows integer;
BEGIN
    IF p_job_id IS NULL OR p_scan_id IS NULL OR p_lease_token IS NULL
       OR p_result_id IS NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'processing commit identifiers are required';
    END IF;
    IF p_processing_version IS NULL OR length(btrim(p_processing_version)) = 0
       OR p_readiness_score IS NULL OR p_readiness_score < 0
       OR p_readiness_score > 100 OR p_confidence IS NULL
       OR p_confidence < 0 OR p_confidence > 1
       OR p_confidence::text IN ('NaN', 'Infinity', '-Infinity')
       OR p_risk_level IS NULL OR p_risk_level NOT IN (
            'stable', 'low_focus', 'elevated_fatigue', 'high_risk', 'unknown'
       )
       OR p_explanation IS NULL OR length(btrim(p_explanation)) = 0
       OR length(p_explanation) > 65535
       OR p_suggested_action IS NULL OR length(btrim(p_suggested_action)) = 0
       OR length(p_suggested_action) > 65535
       OR p_result_hash IS NULL OR p_result_hash !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'validated processing result is invalid';
    END IF;

    SELECT * INTO v_job
    FROM ai_processing.ai_processing_jobs
    WHERE id = p_job_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0002',
            MESSAGE = 'processing job was not found';
    END IF;

    -- A repeated final commit is successful only when it matches the already
    -- committed immutable history row for this exact job/version.
    IF v_job.status = 'completed' THEN
        SELECT * INTO v_history
        FROM ai_processing.processing_result_history
        WHERE job_id = p_job_id
          AND scan_id = p_scan_id
          AND processing_version = p_processing_version
        FOR UPDATE;
        IF NOT FOUND
           OR v_history.readiness_score <> p_readiness_score
           OR v_history.confidence <> p_confidence
           OR v_history.risk_level <> p_risk_level
           OR v_history.explanation <> p_explanation
           OR v_history.suggested_action <> p_suggested_action
           OR v_history.result_hash <> p_result_hash THEN
            RAISE EXCEPTION USING
                ERRCODE = 'P0001',
                MESSAGE = 'completed processing job has conflicting result';
        END IF;
        SELECT id INTO v_result_id
        FROM public.scan_results
        WHERE scan_id = p_scan_id
        FOR UPDATE;
        RETURN QUERY SELECT 'completed'::varchar, v_result_id, v_history.id;
        RETURN;
    END IF;

    IF v_job.status <> 'processing'
       OR v_job.scan_id <> p_scan_id::text
       OR v_job.processing_version <> p_processing_version
       OR v_job.lease_owner IS NULL
       OR v_job.lease_token IS DISTINCT FROM p_lease_token
       OR v_job.lease_expires_at <= CURRENT_TIMESTAMP THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'processing job lease is invalid';
    END IF;

    SELECT * INTO v_scan
    FROM public.wellness_scans
    WHERE id = p_scan_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0002',
            MESSAGE = 'wellness scan was not found';
    END IF;
    IF v_scan."user"::text IS DISTINCT FROM v_job.requester_user_id
       OR v_scan.member::text IS DISTINCT FROM v_job.member_id
       OR v_scan.business_profile::text IS DISTINCT FROM v_job.business_profile_id
       OR v_scan.status NOT IN ('processing', 'media_ready', 'completed') THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'processing job and wellness scan binding is invalid';
    END IF;

    INSERT INTO ai_processing.processing_result_history (
        id, job_id, scan_id, processing_version, readiness_score, confidence,
        risk_level, explanation, suggested_action, result_hash
    ) VALUES (
        p_result_id, p_job_id, p_scan_id, p_processing_version,
        p_readiness_score, p_confidence, p_risk_level, p_explanation,
        p_suggested_action, p_result_hash
    )
    ON CONFLICT (scan_id, processing_version) DO NOTHING
    RETURNING id INTO v_history_id;

    IF v_history_id IS NULL THEN
        SELECT * INTO v_history
        FROM ai_processing.processing_result_history
        WHERE scan_id = p_scan_id
          AND processing_version = p_processing_version
        FOR UPDATE;
        IF NOT FOUND
           OR v_history.job_id <> p_job_id
           OR v_history.readiness_score <> p_readiness_score
           OR v_history.confidence <> p_confidence
           OR v_history.risk_level <> p_risk_level
           OR v_history.explanation <> p_explanation
           OR v_history.suggested_action <> p_suggested_action
           OR v_history.result_hash <> p_result_hash THEN
            RAISE EXCEPTION USING
                ERRCODE = 'P0001',
                MESSAGE = 'processing result history conflicts with existing result';
        END IF;
        v_history_id := v_history.id;
    END IF;

    SELECT id INTO v_result_id
    FROM public.scan_results
    WHERE scan_id = p_scan_id
    FOR UPDATE;

    IF v_result_id IS NULL THEN
        INSERT INTO public.scan_results (
            id, scan_id, readiness_score, confidence, risk_level,
            explanation, suggested_action, ai_model_version
        ) VALUES (
            p_result_id, p_scan_id, p_readiness_score, p_confidence, p_risk_level,
            p_explanation, p_suggested_action, p_processing_version
        )
        ON CONFLICT (scan_id) DO UPDATE SET
            readiness_score = EXCLUDED.readiness_score,
            confidence = EXCLUDED.confidence,
            risk_level = EXCLUDED.risk_level,
            explanation = EXCLUDED.explanation,
            suggested_action = EXCLUDED.suggested_action,
            ai_model_version = EXCLUDED.ai_model_version
        RETURNING id INTO v_result_id;
    ELSE
        UPDATE public.scan_results
        SET readiness_score = p_readiness_score,
            confidence = p_confidence,
            risk_level = p_risk_level,
            explanation = p_explanation,
            suggested_action = p_suggested_action,
            ai_model_version = p_processing_version
        WHERE id = v_result_id;
    END IF;

    UPDATE public.wellness_scans
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP,
        current_processing_version = p_processing_version
    WHERE id = p_scan_id;

    UPDATE ai_processing.ai_processing_attempts
    SET finished_at = CURRENT_TIMESTAMP, outcome = 'completed'
    WHERE job_id = p_job_id
      AND attempt_number = v_job.attempt_count
      AND lease_token = p_lease_token;
    GET DIAGNOSTICS v_attempt_rows = ROW_COUNT;
    IF v_attempt_rows <> 1 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'processing attempt could not be completed';
    END IF;

    UPDATE ai_processing.ai_processing_jobs AS jobs
    SET status = 'completed', result_ref = v_result_id::text,
        lease_owner = NULL, lease_token = NULL,
        lease_expires_at = NULL, heartbeat_at = NULL,
        completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
    WHERE jobs.id = p_job_id
      AND jobs.status = 'processing'
      AND jobs.lease_token = p_lease_token
      AND jobs.lease_expires_at > CURRENT_TIMESTAMP;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING
            ERRCODE = 'P0001',
            MESSAGE = 'processing job completion compare-and-set failed';
    END IF;

    INSERT INTO ai_processing.ai_processing_outbox (
        event_key, job_id, event_type, payload, available_at
    ) VALUES (
        'job:' || p_job_id::text || ':completed',
        p_job_id,
        'job.completed',
        jsonb_build_object(
            'job_id', p_job_id::text,
            'scan_id', p_scan_id::text,
            'processing_version', p_processing_version,
            'status', 'completed',
            'result_ref', v_result_id::text
        ),
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (event_key) DO NOTHING;

    INSERT INTO ai_processing.ai_processing_effects (
        effect_key, job_id, effect_type
    ) VALUES (
        'scan:' || p_scan_id::text || ':completed:' || p_processing_version,
        p_job_id,
        'scan.completed'
    )
    ON CONFLICT (effect_key) DO NOTHING;

    RETURN QUERY SELECT 'completed'::varchar, v_result_id, v_history_id;
END;
$$;

ALTER FUNCTION ai_processing.enforce_job_status_transition()
    OWNER TO ai_processing_owner;
ALTER FUNCTION ai_processing.prevent_processing_history_mutation()
    OWNER TO ai_processing_owner;
ALTER FUNCTION ai_processing.commit_processing_result(
    uuid, uuid, varchar, uuid, uuid, integer, numeric, varchar, text, text, varchar
)
    OWNER TO ai_processing_owner;

REVOKE ALL ON FUNCTION ai_processing.enforce_job_status_transition() FROM PUBLIC;
REVOKE ALL ON FUNCTION ai_processing.prevent_processing_history_mutation() FROM PUBLIC;
REVOKE ALL ON FUNCTION ai_processing.commit_processing_result(
    uuid, uuid, varchar, uuid, uuid, integer, numeric, varchar, text, text, varchar
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION ai_processing.commit_processing_result(
    uuid, uuid, varchar, uuid, uuid, integer, numeric, varchar, text, text, varchar
) TO ai_processing_commit;

GRANT SELECT, INSERT ON
    ai_processing.processing_result_history,
    ai_processing.ai_processing_effects
TO ai_processing_owner;

COMMIT;
