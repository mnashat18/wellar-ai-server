from __future__ import annotations

import concurrent.futures
import multiprocessing
from datetime import datetime, timezone
import math
import os
import queue as queue_module
import shutil
import subprocess
import tempfile
import threading
import time
import sys
from multiprocessing.connection import wait as multiprocessing_wait
from typing import Any
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
from requests import HTTPError, RequestException

from analysis_runtime import AnalysisRuntimeStartupError, AnalysisRuntimeUnavailable, get_runtime as get_analyzer_runtime
from baseline import (
    baseline_ready_for_personalized_scoring,
    baseline_signal_payload,
    baseline_status_payload,
    evaluate_baseline_eligibility,
)
from config import MODEL_VERSION
from directus_client import DirectusClient
from logger import get_logger
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from quality import assess_quality
from scoring import compute_result
from utils import download_temp_file, is_url, remove_temp_file, safe_number, sanitize_payload, sanitize_text
from validation import ValidationPolicy, fail_validation, failure_message, validate_scan_inputs
from durable_config import DurableConfigurationError, DurableSettings, durable_enabled


app = FastAPI()
logger = get_logger()
ml_runtime = MLRuntime()
_ANALYZER_CHILD_ENV = "AI_SERVER_ANALYSIS_CHILD"
_ANALYZER_CHILD_MODE = os.getenv(_ANALYZER_CHILD_ENV) == "1"
if not _ANALYZER_CHILD_MODE:
    ml_runtime.load()
directus = None if _ANALYZER_CHILD_MODE else DirectusClient()

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean environment value")


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value: Any = default if raw is None else raw.strip()
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a positive finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{name} must be a positive finite number") from None
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RuntimeError(f"{name} must be a positive finite number")
    return numeric


VALIDATION_POLICY = None if _ANALYZER_CHILD_MODE else ValidationPolicy.from_env()
DURABLE_PROCESSING_ENABLED = durable_enabled()
AI_PROCESSING_VERSION = os.getenv("AI_PROCESSING_VERSION", "cie_v1_2").strip() or "cie_v1_2"
_DURABLE_COORDINATOR = None
AI_SERVER_ENV = os.getenv("AI_SERVER_ENV", "production").strip().lower()
DEBUG_SCAN_ENDPOINT_ENABLED = (
    AI_SERVER_ENV in {"dev", "development", "local", "test"}
    and _env_flag("DEBUG_SCAN_ENDPOINT_ENABLED", False)
)
OPTIONAL_PHRASE_TIMEOUT_SECONDS = 1.5
TRANSCRIPTION_TIMEOUT_SECONDS = _env_positive_float(
    "AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS",
    4.0,
)
FAST_SCAN_MODE = _env_flag("FAST_SCAN_MODE", True)
FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS = _env_positive_float(
    "FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS",
    2.5,
)
MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS = _env_positive_float(
    "MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS",
    8.0,
)
AUDIO_ANALYSIS_WORKER_TIMEOUT_SECONDS = _env_positive_float(
    "AUDIO_ANALYSIS_WORKER_TIMEOUT_SECONDS",
    6.5,
)

_ACTIVE_SCAN_IDS: set[str] = set()
_ACTIVE_SCAN_IDS_LOCK = threading.Lock()

SCAN_STATUS_PENDING = "pending"
SCAN_STATUS_MEDIA_READY = "media_ready"
SCAN_STATUS_PROCESSING = "processing"
SCAN_STATUS_COMPLETED = "completed"
SCAN_STATUS_FAILED = "failed"

FAILURE_REASON_MISSING_MEDIA = "missing_media"
FAILURE_REASON_VIDEO_MISSING = "video_missing"
FAILURE_REASON_VIDEO_TOO_DARK = "video_too_dark"
FAILURE_REASON_VIDEO_BLURRY = "video_blurry"
FAILURE_REASON_VIDEO_TOO_SHORT = "video_too_short"
FAILURE_REASON_UNSTABLE_VIDEO = "unstable_video"
FAILURE_REASON_IMAGE_MISSING = "image_missing"
FAILURE_REASON_IMAGE_TOO_DARK = "image_too_dark"
FAILURE_REASON_IMAGE_BLURRY = "image_blurry"
FAILURE_REASON_AUDIO_MISSING = "audio_missing"
FAILURE_REASON_AUDIO_TOO_QUIET = "audio_too_quiet"
FAILURE_REASON_AUDIO_TOO_NOISY = "audio_too_noisy"
FAILURE_REASON_SPEECH_NOT_DETECTED = "speech_not_detected"
FAILURE_REASON_AUDIO_TOO_SHORT = "audio_too_short"
FAILURE_REASON_FACE_NOT_VISIBLE = "face_not_visible"
FAILURE_REASON_PHRASE_MISMATCH = "phrase_mismatch"
FAILURE_REASON_TRANSCRIPTION_FAILED = "transcription_failed"
FAILURE_REASON_EXPECTED_PHRASE_MISSING = "expected_phrase_missing"
FAILURE_REASON_LOW_QUALITY_MEDIA = "low_quality_media"
FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT = "audio_validation_timeout"
FAILURE_REASON_MODEL_NOT_LOADED = "model_not_loaded"
FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED = "directus_download_failed"
FAILURE_REASON_ANALYSIS_EXCEPTION = "analysis_exception"
FAILURE_REASON_WRITEBACK_FAILED = "writeback_failed"

_FULL_EVIDENCE_RESULT_KEY_ALLOWLIST = (
    "face_metrics",
    "fusion_details",
    "modality_scores",
    "voice_confidence",
    "voice_metrics",
)

OPTIONAL_SCAN_RESULT_FIELDS = [
    "analysis_metadata",
    "media_quality",
    "warnings",
    "modality_scores",
    "fusion_details",
    "internal_analysis",
    "result_status",
    "capture_quality_score",
    "measurement_reliability_score",
    "observed_fatigue_score",
    "fatigue_evidence_score",
    "personal_deviation_score",
    "task_completion_status",
    "baseline_status_at_inference",
    "baseline_confidence",
    "baseline_eligible",
    "hard_gates_triggered",
    "explainable_reasons",
]

SCAN_RESULT_NUMERIC_FIELDS: dict[str, bool] = {
    "readiness_score": True,
    "observed_fatigue_score": True,
    "confidence": False,
    "camera_confidence": False,
    "voice_confidence": False,
    "task_performance_score": True,
    "confidence_drift": False,
    "phrase_match_score": False,
    "audio_quality_score": False,
    "video_quality_score": False,
    "image_quality_score": False,
    "capture_quality_score": False,
    "measurement_reliability_score": False,
    "personal_deviation_score": False,
    "baseline_confidence": False,
    "fatigue_evidence_score": False,
}

SCAN_RESULT_CHOICE_ALIASES: dict[str, dict[str, list[str]]] = {
    "risk_level": {
        "stable": ["Stable"],
        "low_focus": ["Low Focus"],
        "elevated_fatigue": ["Elevated Fatigue"],
        "high_risk": ["High Risk"],
    },
    "baseline_status": {
        "collecting": ["Collecting"],
        "provisional": ["Provisional"],
        "active": ["Active"],
        "needs_review": ["Needs Review"],
        "disabled": ["Disabled"],
    },
    "result_status": {
        "scored": ["Scored"],
        "retake_required": ["Retake Required"],
        "incomplete": ["Incomplete"],
        "low_confidence": ["Low Confidence"],
        "failed": ["Failed"],
    },
    "suggested_action": {
        "continue_normal_activity": ["Continue Normal Activity"],
        "review_required": ["Review Required"],
        "rescan_recommended": ["Rescan Recommended"],
        "rest_advised": ["Rest Advised"],
        "manager_review": ["Manager Review"],
    },
    "task_completion_status": {
        "completed": ["Completed"],
        "incomplete_required_speech": ["Incomplete Required Speech"],
        "incomplete_required_task": ["Incomplete Required Task"],
        "not_required": ["Not Required"],
    },
}


class Media(BaseModel):
    image: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    audio: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    video: str | None = Field(None, description="Local path, URL, or Directus asset ID")


class Task(BaseModel):
    reaction_time: float | None = None
    errors: int | None = None
    attempts: int | None = None


class ScanRequest(BaseModel):
    scan_id: str = Field(..., min_length=1, max_length=255)

    class Config:
        extra = "ignore"


class BaselineRequest(BaseModel):
    scan_id: str | None = Field(default=None, max_length=255)
    media: Media | None = None
    manually_unreliable: bool = False


class ProcessResponse(BaseModel):
    ok: bool = True
    status: str
    scan_id: str


class ScanResultResponse(BaseModel):
    status: str
    retake_required: bool = False
    failure_reason: str | None = None
    readiness_score: int | None = None
    risk_level: str | None = None
    confidence: float | None = None
    camera_confidence: float | None = None
    voice_confidence: float | None = None
    task_performance_score: int | None = None
    baseline_used: bool = False
    confidence_drift: float | None = None
    face_metrics: dict | None = None
    voice_metrics: dict | None = None
    reaction_metrics: dict | None = None
    explanation: str
    suggested_action: str


class DurableClaimRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    processing_version: str = Field(..., min_length=1, max_length=100)
    trace_id: str = Field(..., min_length=1, max_length=255)

    class Config:
        extra = "forbid"


class DurableHeartbeatRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    lease_token: UUID

    class Config:
        extra = "forbid"


class DurableCommitRequest(BaseModel):
    scan_id: str = Field(..., min_length=1, max_length=255)
    processing_version: str = Field(..., min_length=1, max_length=100)
    lease_token: UUID
    result_id: UUID
    readiness_score: int = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., min_length=1, max_length=32)
    explanation: str = Field(..., min_length=1, max_length=65535)
    suggested_action: str = Field(..., min_length=1, max_length=65535)

    class Config:
        extra = "forbid"


class DurableFailRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    lease_token: UUID
    failure_code: str = Field(..., min_length=1, max_length=100)
    error_type: str = Field(..., min_length=1, max_length=100)
    retry_delay_seconds: int = Field(default=5, ge=0, le=3600)
    terminal_failure: bool = False

    class Config:
        extra = "forbid"


class DurableWorkerRegisterRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    worker_version: str = Field(..., min_length=1, max_length=100)
    capacity: int = Field(default=1, ge=1, le=64)

    class Config:
        extra = "forbid"


class DurableWorkerHeartbeatRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    lease_token: UUID
    active_slots: int = Field(..., ge=0, le=64)

    class Config:
        extra = "forbid"


class DurableWorkerDrainRequest(BaseModel):
    worker_id: str = Field(..., min_length=1, max_length=255)
    lease_token: UUID

    class Config:
        extra = "forbid"


@app.on_event("startup")
def _startup_analyzer_runtime() -> None:
    runtime = get_analyzer_runtime()
    try:
        runtime.start_background()
    except AnalysisRuntimeStartupError:
        logger.error("analyzer_runtime_start_failed error_type=AnalysisRuntimeStartupError")
        raise


@app.on_event("shutdown")
def _shutdown_analyzer_runtime() -> None:
    try:
        get_analyzer_runtime().shutdown()
    except Exception as exc:
        logger.error("analyzer_runtime_shutdown_failed error_type=%s", type(exc).__name__)


class BaselineStatusResponse(BaseModel):
    is_active: bool
    scan_count: int
    scans_remaining: int
    is_provisional: bool
    needs_morning_scan: bool
    needs_evening_scan: bool
    message: str


class ProcessingError(RuntimeError):
    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason
        self.message = message or reason


class SchemaValidationError(ProcessingError):
    def __init__(self, collection: str, field_name: str, actual_length: int, max_length: int):
        super().__init__(
            FAILURE_REASON_WRITEBACK_FAILED,
            (
                f"{collection}.{field_name} exceeds Directus max length "
                f"({actual_length}>{max_length})"
            ),
        )
        self.collection = collection
        self.field_name = field_name
        self.actual_length = actual_length
        self.max_length = max_length


@app.get("/")
def root():
    return {"status": "ok"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relation_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id", value.get("uuid"))
    return value


def _model_to_dict(model: BaseModel | None) -> dict:
    if not model:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _scan_timestamp(scan_context: dict) -> str:
    return (
        scan_context.get("completed_at")
        or scan_context.get("started_at")
        or scan_context.get("date_created")
        or _utc_now()
    )


def _safe_string(value: Any, max_len: int = 255, fallback: str | None = None) -> str | None:
    return sanitize_text(value, fallback=fallback, max_len=max_len)


def _field_length_checked_string(
    collection: str,
    field_name: str,
    value: Any,
    *,
    fallback: str | None = None,
    default_max_len: int = 65535,
    required: bool = False,
) -> str | None:
    text = sanitize_text(value, fallback=fallback, max_len=default_max_len)
    if text is None:
        return None
    max_length = directus.get_field_max_length(collection, field_name)
    if max_length is None or len(text) <= max_length:
        return text
    logger.warning(
        "directus_field_too_long collection=%s field=%s max_length=%s actual_length=%s required=%s",
        collection,
        field_name,
        max_length,
        len(text),
        required,
    )
    if required:
        raise SchemaValidationError(collection, field_name, len(text), max_length)
    return None


def _normalize_choice_token(value: Any) -> str:
    text = _safe_string(value, max_len=255, fallback="") or ""
    normalized = []
    for char in text.lower():
        normalized.append(char if char.isalnum() else "_")
    compact = "".join(normalized).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact


def _safe_numeric(value: Any, *, integer: bool) -> int | float | None:
    if value is None or isinstance(value, bool) or type(value) not in {int, float}:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if integer:
        if not numeric.is_integer():
            return None
        return int(numeric)
    return round(numeric, 6)


def _is_json_field(collection: str, field_name: str) -> bool:
    schema_type = directus.get_field_schema_type(collection, field_name) or ""
    return schema_type in {"json", "jsonb"} or "json" in schema_type


def _truncate_string_to_schema(
    collection: str,
    field_name: str,
    value: Any,
    *,
    fallback: str | None = None,
    default_max_len: int = 65535,
) -> str | None:
    text = sanitize_text(value, fallback=fallback, max_len=default_max_len)
    if text is None:
        return None
    max_length = directus.get_field_max_length(collection, field_name)
    if max_length is None or len(text) <= max_length:
        return text
    logger.warning(
        "directus_field_truncated collection=%s field=%s max_length=%s actual_length=%s",
        collection,
        field_name,
        max_length,
        len(text),
    )
    return text[:max_length]


def _coerce_json_field(collection: str, field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        schema_type = directus.get_field_schema_type(collection, field_name)
        if not schema_type or _is_json_field(collection, field_name):
            return value
        logger.warning(
            "directus_json_field_skipped collection=%s field=%s schema_type=%s value_type=%s",
            collection,
            field_name,
            schema_type,
            type(value).__name__,
        )
        return None
    return value


def _scan_result_payload_diagnostics(payload: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    string_lengths = {key: len(value) for key, value in payload.items() if isinstance(value, str)}
    field_types = {key: type(value).__name__ for key, value in payload.items()}
    return string_lengths, field_types


def _log_scan_result_payload_ready(scan_id: str, payload: dict[str, Any]) -> None:
    string_lengths, field_types = _scan_result_payload_diagnostics(payload)
    ai_model_version = payload.get("ai_model_version")
    logger.info(
        "scan_id=%s step=scan_result_payload_ready ai_model_version=%r ai_model_version_type=%s ai_model_version_len=%s payload_keys=%s payload_types=%s payload_string_lengths=%s",
        scan_id,
        ai_model_version,
        type(ai_model_version).__name__ if ai_model_version is not None else None,
        len(str(ai_model_version)) if ai_model_version is not None else 0,
        sorted(payload.keys()),
        field_types,
        string_lengths,
    )


def _coerce_scan_result_choice(field_name: str, value: Any) -> Any:
    text = _safe_string(value, max_len=255)
    if text is None:
        return None
    choices = directus.get_field_choices("scan_results", field_name)
    if not choices:
        return text

    normalized_allowed: dict[str, Any] = {}
    allowed_values: list[Any] = []
    for choice in choices:
        actual = choice.get("value")
        label = choice.get("label")
        if actual is None:
            continue
        allowed_values.append(actual)
        for token in [actual, label]:
            normalized = _normalize_choice_token(token)
            if normalized:
                normalized_allowed[normalized] = actual

    normalized_text = _normalize_choice_token(text)
    if text in allowed_values:
        return text
    if normalized_text in normalized_allowed:
        return normalized_allowed[normalized_text]

    for alias in SCAN_RESULT_CHOICE_ALIASES.get(field_name, {}).get(text, []):
        normalized_alias = _normalize_choice_token(alias)
        if normalized_alias in normalized_allowed:
            mapped = normalized_allowed[normalized_alias]
            logger.info(
                "scan_result_choice_mapped field=%s source=%s mapped=%s allowed=%s",
                field_name,
                text,
                mapped,
                allowed_values,
            )
            return mapped

    logger.warning(
        "scan_result_choice_unmapped field=%s value=%s allowed=%s",
        field_name,
        text,
        allowed_values,
    )
    return None


def _scan_result_scan_id_value(scan_id: Any) -> str | int | None:
    value = _relation_id(scan_id)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _safe_string(value, max_len=255)
    return text


def _schema_aware_scan_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)

    candidate["scan_id"] = _scan_result_scan_id_value(candidate.get("scan_id"))

    for field_name, integer in SCAN_RESULT_NUMERIC_FIELDS.items():
        if field_name not in candidate:
            continue
        raw_value = candidate.get(field_name)
        coerced = _safe_numeric(raw_value, integer=integer)
        if raw_value is not None and coerced is None:
            candidate.pop(field_name, None)
        else:
            candidate[field_name] = coerced

    for field_name in [
        "risk_level",
        "suggested_action",
        "result_status",
        "task_completion_status",
        "baseline_status_at_inference",
    ]:
        if field_name in candidate:
            choice_field = "baseline_status" if field_name == "baseline_status_at_inference" else field_name
            candidate[field_name] = _coerce_scan_result_choice(choice_field, candidate.get(field_name))

    candidate["explanation"] = _truncate_string_to_schema(
        "scan_results",
        "explanation",
        candidate.get("explanation"),
        fallback="Analysis completed.",
        default_max_len=65535,
    )
    candidate["suggested_action"] = _field_length_checked_string(
        "scan_results",
        "suggested_action",
        candidate.get("suggested_action"),
        default_max_len=255,
        required=True,
    )
    candidate["ai_model_version"] = _field_length_checked_string(
        "scan_results",
        "ai_model_version",
        MODEL_VERSION,
        fallback=MODEL_VERSION,
        default_max_len=255,
        required=True,
    )

    for field_name in ["spoken_transcript", "expected_phrase"]:
        if field_name in candidate:
            candidate[field_name] = _field_length_checked_string(
                "scan_results",
                field_name,
                candidate.get(field_name),
                default_max_len=65535,
                required=False,
            )

    for field_name in [
        "face_metrics",
        "voice_metrics",
        "reaction_metrics",
        "analysis_metadata",
        "media_quality",
        "warnings",
        "modality_scores",
        "fusion_details",
        "validation_warnings",
        "hard_gates_triggered",
        "explainable_reasons",
    ]:
        if field_name in candidate:
            candidate[field_name] = _coerce_json_field("scan_results", field_name, candidate.get(field_name))

    filtered = _scan_result_payload(candidate)

    required_missing: list[str] = []
    for field_name in [
        "scan_id",
        "risk_level",
        "confidence",
        "readiness_score",
        "explanation",
        "suggested_action",
        "ai_model_version",
    ]:
        value = filtered.get(field_name)
        missing_value = value is None or (
            isinstance(value, str) and not value.strip()
        )
        if field_name not in filtered or missing_value:
            required = directus.is_field_required("scan_results", field_name)
            if required:
                required_missing.append(field_name)
    if required_missing:
        raise ProcessingError(
            FAILURE_REASON_WRITEBACK_FAILED,
            f"scan_results required fields missing after schema validation: {', '.join(required_missing)}",
        )
    return filtered


def _directus_payload_preserve_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _directus_payload_preserve_nulls(raw) for key, raw in value.items()}
    if isinstance(value, list):
        return [_directus_payload_preserve_nulls(raw) for raw in value]
    if isinstance(value, tuple):
        return [_directus_payload_preserve_nulls(raw) for raw in value]
    if isinstance(value, float):
        return safe_number(value, digits=6)
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def _log_step(scan_id: str, step: str, **details: Any) -> None:
    if details:
        logger.info("scan_id=%s step=%s details=%s", scan_id, step, sanitize_payload(details))
        return
    logger.info("scan_id=%s step=%s", scan_id, step)


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _log_perf(scan_id: str, metric: str, elapsed_ms: int | float | None) -> None:
    if isinstance(elapsed_ms, bool) or type(elapsed_ms) not in {int, float}:
        value = 0
    else:
        numeric = float(elapsed_ms)
        value = int(round(numeric)) if math.isfinite(numeric) and numeric >= 0.0 else 0
    logger.info("[PERF] %s scan_id=%s value=%s", metric, scan_id, value)


def _timing_ms_value(value: Any) -> int:
    if isinstance(value, bool) or type(value) not in {int, float}:
        return 0
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        return 0
    return int(round(numeric))


def _claim_active_scan(scan_id: str) -> bool:
    with _ACTIVE_SCAN_IDS_LOCK:
        if scan_id in _ACTIVE_SCAN_IDS:
            return False
        _ACTIVE_SCAN_IDS.add(scan_id)
        return True


def _release_active_scan(scan_id: str) -> None:
    with _ACTIVE_SCAN_IDS_LOCK:
        _ACTIVE_SCAN_IDS.discard(scan_id)


def _cleanup_download_future(future: concurrent.futures.Future) -> None:
    try:
        path, is_temp = future.result()
    except Exception:
        return
    if path and is_temp:
        remove_temp_file(path)


def _defer_temp_file_cleanup(
    future: concurrent.futures.Future,
    path: str | None,
    temp_files: list[str],
) -> None:
    if not path or path not in temp_files:
        return
    temp_files.remove(path)
    future.add_done_callback(lambda _future, deferred_path=path: remove_temp_file(deferred_path))


def _is_retryable_directus_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        return status_code in {408, 425, 429} or (
            isinstance(status_code, int) and 500 <= status_code <= 599
        )
    return isinstance(exc, RequestException)


def _retry_directus_call(operation: str, fn, *, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not _is_retryable_directus_error(exc):
                raise
            delay_seconds = 0.15 * (2 ** (attempt - 1))
            logger.warning(
                "directus_operation_retry operation=%s attempt=%s error_type=%s",
                operation,
                attempt,
                type(exc).__name__,
            )
            time.sleep(delay_seconds)
    raise last_error or RuntimeError(f"{operation} failed")


def _log_validation_decision(
    scan_id: str,
    *,
    valid_modalities: list[str],
    timed_out_modalities: list[str],
    analyzer_error_modalities: list[str] | None = None,
    terminal_reason: str,
) -> None:
    logger.info("[VALIDATION_DECISION] valid_modalities scan_id=%s value=%s", scan_id, valid_modalities)
    logger.info("[VALIDATION_DECISION] timed_out_modalities scan_id=%s value=%s", scan_id, timed_out_modalities)
    logger.info(
        "[VALIDATION_DECISION] analyzer_error_modalities scan_id=%s value=%s",
        scan_id,
        analyzer_error_modalities or [],
    )
    logger.info("[VALIDATION_DECISION] terminal_reason scan_id=%s value=%s", scan_id, terminal_reason)


def _log_validation_lifecycle(scan_id: str, *, all_workers_terminal: bool, running_modalities: list[str]) -> None:
    logger.info(
        "[VALIDATION_LIFECYCLE] all_workers_terminal=%s running_modalities=%s scan_id=%s",
        str(bool(all_workers_terminal)).lower(),
        running_modalities,
        scan_id,
    )


def _shutdown_executor(executor: concurrent.futures.ThreadPoolExecutor | None, *, wait: bool = False) -> None:
    if executor is None:
        return
    executor.shutdown(wait=wait, cancel_futures=True)


def _analysis_worker_entry(
    result_conn: Any,
    analyzer_name: str,
    path: str | None,
    missing_warning: str,
    scan_id: str,
    parent_started_at: float,
) -> None:
    child_started_at = time.perf_counter()
    analyzer_execution_started_at = None
    try:
        if analyzer_name == "video":
            from video import analyze_video

            analyzer_execution_started_at = time.perf_counter()
            result = _safe_analyze(analyze_video, path, missing_warning)
        elif analyzer_name == "audio":
            from audio import analyze_audio

            analyzer_execution_started_at = time.perf_counter()
            result = _safe_analyze(analyze_audio, path, missing_warning)
        elif analyzer_name == "image":
            from vision import analyze_face

            analyzer_execution_started_at = time.perf_counter()
            result = _safe_analyze(analyze_face, path, missing_warning)
        else:
            result = {"score": None, "details": {"status": "invalid"}}
        payload = {
            "ok": True,
            "result": result,
            "metrics": {
                "child_boot_ms": _elapsed_ms(parent_started_at),
                "analyzer_execution_ms": _elapsed_ms(analyzer_execution_started_at or child_started_at),
            },
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "metrics": {
                "child_boot_ms": _elapsed_ms(parent_started_at),
            },
        }

    result_publish_started_at = time.perf_counter()
    try:
        result_conn.send(payload)
    except Exception:
        pass
    finally:
        result_publish_ms = _elapsed_ms(result_publish_started_at)
        total_worker_ms = _elapsed_ms(parent_started_at)
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        child_boot_ms = metrics.get("child_boot_ms")
        analyzer_execution_ms = metrics.get("analyzer_execution_ms")
        _log_perf(scan_id, f"{analyzer_name}_child_boot_ms", child_boot_ms)
        _log_perf(scan_id, f"{analyzer_name}_analyzer_execution_ms", analyzer_execution_ms)
        _log_perf(scan_id, f"{analyzer_name}_result_publish_ms", result_publish_ms)
        _log_perf(scan_id, f"{analyzer_name}_total_worker_ms", total_worker_ms)
        try:
            result_conn.close()
        except Exception:
            pass


def _analysis_worker_placeholder(analyzer_name: str) -> dict:
    return _analysis_timeout_placeholder(analyzer_name)


def _analysis_error_placeholder(analyzer_name: str) -> dict:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")
    status = "error"
    return {
        "score": None,
        "details": {
            "status": status,
            warning_key: [f"{analyzer_name}_analysis_error"],
        },
    }


def _finalize_analysis_process(process: multiprocessing.Process, *, timeout_seconds: float = 1.0) -> dict[str, Any]:
    state = {
        "terminated": False,
        "killed": False,
        "alive": process.is_alive(),
        "exitcode": process.exitcode,
    }
    if process.is_alive():
        process.terminate()
        state["terminated"] = True
        process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        state["killed"] = True
        process.join(timeout_seconds)
    state["alive"] = process.is_alive()
    state["exitcode"] = process.exitcode
    try:
        process.close()
    except Exception:
        pass
    return state


def _finalize_analysis_worker(
    *,
    scan_id: str,
    analyzer_name: str,
    process: multiprocessing.Process,
    result_conn: Any,
    started_at: float,
    parent_process_start_ms: int | None,
    timeout_seconds: float | None,
    timed_out: bool,
    result_ready: bool = False,
) -> tuple[dict, dict[str, Any]]:
    metric_name = f"{analyzer_name}_validation_ms"
    elapsed_ms = _elapsed_ms(started_at)
    state: dict[str, Any] = {
        "state": "starting",
        "child_started": True,
        "result_ready": False,
        "result_received": False,
        "analyzer_error": False,
        "process_exited": False,
        "timed_out": timed_out,
        "alive": process.is_alive(),
        "terminated": False,
        "killed": False,
        "process_exitcode": process.exitcode,
        "final_alive": process.is_alive(),
        "finalized": False,
        "parent_process_start_ms": parent_process_start_ms,
        "process_start_begin_ms": None,
        "process_start_return_ms": None,
        "child_entry_ms": None,
        "analyzer_import_ms": None,
        "analyzer_execution_ms": None,
        "result_send_ms": None,
        "result_receive_ms": None,
        "process_exit_ms": None,
        "child_boot_ms": None,
        "total_worker_ms": None,
    }

    if timed_out:
        logger.warning(
            "%s_analysis_worker_timeout scan_id=%s timeout_seconds=%s",
            analyzer_name,
            scan_id,
            timeout_seconds,
        )
        state.update(_finalize_analysis_process(process))
        try:
            result_conn.close()
        except Exception:
            pass
        _log_step(
            scan_id,
            "analysis_worker_finalized",
            analyzer=analyzer_name,
            timed_out=True,
            process_exitcode=state["process_exitcode"],
            terminated=state["terminated"],
            killed=state["killed"],
            alive=state["alive"],
            result_received=False,
            analyzer_error=False,
        )
        state["process_exited"] = not state["alive"]
        state["final_alive"] = state["alive"]
        state["process_exit_ms"] = elapsed_ms
        state["finalized"] = True
        _log_perf(scan_id, metric_name, elapsed_ms)
        return _analysis_worker_placeholder(analyzer_name), state

    payload = None
    try:
        if result_ready:
            if result_conn.poll(0):
                payload = result_conn.recv()
        elif result_conn.poll(0.5):
            payload = result_conn.recv()
    except EOFError:
        payload = None
    finally:
        try:
            result_conn.close()
        except Exception:
            pass
        state.update(_finalize_analysis_process(process))
        state["process_exited"] = not state["alive"]
        state["final_alive"] = state["alive"]
        state["process_exit_ms"] = elapsed_ms
        state["finalized"] = True

    result_received = False
    analyzer_error = False
    if isinstance(payload, dict) and payload.get("ok"):
        result = payload.get("result")
        if isinstance(result, dict):
            result_received = True
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            state["result_ready"] = True
            state["result_received"] = True
            state["child_entry_ms"] = metrics.get("child_entry_ms")
            state["analyzer_import_ms"] = metrics.get("analyzer_import_ms")
            state["analyzer_execution_ms"] = metrics.get("analyzer_execution_ms")
            state["result_send_ms"] = metrics.get("result_send_ms") or metrics.get("result_publish_ms")
            state["total_worker_ms"] = metrics.get("total_worker_ms")
            state["process_exitcode"] = state["process_exitcode"]
            _log_step(
                scan_id,
                "analysis_worker_finalized",
                analyzer=analyzer_name,
                timed_out=False,
                process_exitcode=state["process_exitcode"],
                terminated=state["terminated"],
                killed=state["killed"],
                alive=state["alive"],
                result_received=True,
                analyzer_error=False,
            )
            _log_perf(scan_id, metric_name, elapsed_ms)
            _log_perf(scan_id, f"{analyzer_name}_child_entry_ms", state["child_entry_ms"])
            _log_perf(scan_id, f"{analyzer_name}_analyzer_import_ms", state["analyzer_import_ms"])
            _log_perf(scan_id, f"{analyzer_name}_analyzer_execution_ms", state["analyzer_execution_ms"])
            _log_perf(scan_id, f"{analyzer_name}_result_send_ms", state["result_send_ms"])
            _log_perf(scan_id, f"{analyzer_name}_result_receive_ms", elapsed_ms)
            _log_perf(scan_id, f"{analyzer_name}_process_exit_ms", state["process_exit_ms"])
            return result, state
        analyzer_error = True
    elif isinstance(payload, dict) and payload.get("error_type"):
        analyzer_error = True
        logger.warning(
            "analysis_worker_error scan_id=%s analyzer=%s error_type=%s",
            scan_id,
            analyzer_name,
            payload.get("error_type"),
        )
    elif payload is None:
        analyzer_error = True

    _log_step(
        scan_id,
        "analysis_worker_finalized",
        analyzer=analyzer_name,
        timed_out=False,
        process_exitcode=state["process_exitcode"],
        terminated=state["terminated"],
        killed=state["killed"],
        alive=state["alive"],
        result_received=result_received,
        analyzer_error=analyzer_error,
    )
    _log_perf(scan_id, metric_name, elapsed_ms)
    _log_perf(scan_id, f"{analyzer_name}_child_entry_ms", state["child_entry_ms"])
    _log_perf(scan_id, f"{analyzer_name}_analyzer_import_ms", state["analyzer_import_ms"])
    _log_perf(scan_id, f"{analyzer_name}_analyzer_execution_ms", state["analyzer_execution_ms"])
    _log_perf(scan_id, f"{analyzer_name}_result_send_ms", state["result_send_ms"])
    _log_perf(scan_id, f"{analyzer_name}_result_receive_ms", elapsed_ms if result_received else None)
    _log_perf(scan_id, f"{analyzer_name}_process_exit_ms", state["process_exit_ms"])
    if analyzer_error:
        state["analyzer_error"] = True
        state["result_received"] = bool(payload)
        state["result_ready"] = bool(payload)
        result_received = bool(payload)
        return _analysis_error_placeholder(analyzer_name), state
    return _analysis_worker_placeholder(analyzer_name), state


def _run_parallel_analysis(
    scan_id: str,
    media: Media,
) -> tuple[dict, dict[str, dict[str, Any]]]:
    runtime = get_analyzer_runtime()
    runtime_started = time.perf_counter()
    analysis_results, worker_states = runtime.run_scan(
        scan_id,
        media,
        deadline_seconds=MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
    )
    _log_perf(scan_id, "analyzer_runtime_call_ms", _elapsed_ms(runtime_started))
    return {
        "video": analysis_results.get("video") or _analysis_worker_placeholder("video"),
        "audio": analysis_results.get("audio") or _analysis_worker_placeholder("audio"),
        "image": analysis_results.get("image") or _analysis_worker_placeholder("image"),
    }, worker_states


def _has_meaningful_evidence(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if type(value) in {int, float}:
        numeric = float(value)
        return math.isfinite(numeric) and not math.isclose(numeric, 0.0, abs_tol=0.0)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_meaningful_evidence(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_evidence(item) for item in value)
    return False


def _finite_result_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and type(value) in {int, float}
        and math.isfinite(float(value))
    )


def _result_has_valid_evidence(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False

    profiles = ((result.get("fusion_details") or {}).get("signal_profiles") or {})
    if isinstance(profiles, dict):
        for profile in profiles.values():
            if (
                isinstance(profile, dict)
                and profile.get("present") is True
                and _finite_result_score(profile.get("score"))
            ):
                return True

    modality_scores = result.get("modality_scores")
    if isinstance(modality_scores, dict):
        if any(_finite_result_score(value) for value in modality_scores.values()):
            return True

    for field_name, score_names in {
        "face_metrics": ("face_score", "image_score", "video_score"),
        "voice_metrics": ("voice_score",),
        "reaction_metrics": ("reaction_score",),
    }.items():
        metrics = result.get(field_name)
        if isinstance(metrics, dict) and any(
            _finite_result_score(metrics.get(score_name))
            for score_name in score_names
        ):
            return True
    return False


def _raw_analysis_has_valid_evidence(
    modality: str,
    result: dict[str, Any],
    worker_state: dict[str, Any],
) -> bool:
    if not isinstance(result, dict) or not isinstance(worker_state, dict):
        return False
    if (
        worker_state.get("result_received") is not True
        or worker_state.get("timed_out") is True
        or worker_state.get("analyzer_error") is True
        or worker_state.get("final_alive") is True
    ):
        return False

    details = result.get("details")
    if not isinstance(details, dict) or details.get("status") != "ok":
        return False

    confidence_key = {
        "video": "visual_confidence",
        "audio": "audio_confidence",
        "image": "image_confidence",
    }.get(modality)
    quality_key = {
        "video": "visual_quality_score",
        "audio": "audio_quality_score",
        "image": "image_quality_score",
    }.get(modality)
    if confidence_key is None or quality_key is None:
        return False
    if not _finite_result_score(details.get(confidence_key)) or not _finite_result_score(details.get(quality_key)):
        return False
    if modality in {"video", "audio"}:
        duration = details.get("duration_seconds")
        if duration is None and modality == "audio":
            duration = details.get("duration_sec")
        if not _finite_result_score(duration) or float(duration) < 0.0:
            return False
    return True


def _fused_result_valid_modalities(
    result: dict[str, Any],
    *,
    raw_valid_modalities: list[str],
) -> list[str]:
    raw_valid = set(raw_valid_modalities)
    profiles = ((result.get("fusion_details") or {}).get("signal_profiles") or {})
    modality_scores = result.get("modality_scores") or {}
    face_metrics = result.get("face_metrics") or {}
    voice_metrics = result.get("voice_metrics") or {}
    metric_scores = {
        "video": face_metrics.get("video_score"),
        "audio": voice_metrics.get("voice_score"),
        "image": face_metrics.get("image_score"),
    }

    valid: list[str] = []
    for modality in ("video", "audio", "image"):
        profile = profiles.get(modality) if isinstance(profiles, dict) else None
        if (
            modality in raw_valid
            and isinstance(profile, dict)
            and profile.get("present") is True
            and _finite_result_score(profile.get("score"))
            and _finite_result_score(modality_scores.get(modality))
            and _finite_result_score(metric_scores[modality])
        ):
            valid.append(modality)
    return valid


def _log_full_evidence_gate(
    scan_id: str,
    *,
    result: dict[str, Any],
    worker_states: dict[str, dict[str, Any]],
    valid_modalities: list[str],
) -> None:
    valid = set(valid_modalities)
    voice_confidence = result.get("voice_confidence")
    logger.info(
        "[FULL_EVIDENCE_GATE] scan_id=%s video_evidence_present=%s audio_evidence_present=%s "
        "image_evidence_present=%s voice_confidence_present=%s voice_confidence_numeric=%s "
        "video_timeout=%s audio_timeout=%s image_timeout=%s video_analyzer_error=%s "
        "audio_analyzer_error=%s image_analyzer_error=%s final_result_keys=%s",
        scan_id,
        "video" in valid,
        "audio" in valid,
        "image" in valid,
        voice_confidence is not None,
        _finite_result_score(voice_confidence),
        bool((worker_states.get("video") or {}).get("timed_out")),
        bool((worker_states.get("audio") or {}).get("timed_out")),
        bool((worker_states.get("image") or {}).get("timed_out")),
        bool((worker_states.get("video") or {}).get("analyzer_error")),
        bool((worker_states.get("audio") or {}).get("analyzer_error")),
        bool((worker_states.get("image") or {}).get("analyzer_error")),
        [key for key in _FULL_EVIDENCE_RESULT_KEY_ALLOWLIST if key in result],
    )


def _build_scan_result_response(
    *,
    ok: bool,
    scan_id: str,
    status: str | None = None,
    error: str | None = None,
    current_status: str | None = None,
    status_code: int = 200,
) -> JSONResponse:
    payload: dict[str, Any] = {"ok": ok, "scan_id": scan_id}
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
    if current_status is not None:
        payload["current_status"] = current_status
    return JSONResponse(status_code=status_code, content=payload)


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def _ids_equal(left: Any, right: Any) -> bool:
    left_id = _relation_id(left)
    right_id = _relation_id(right)
    if left_id is None or right_id is None:
        return False
    return str(left_id) == str(right_id)


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if type(value) in {int, float}:
        numeric = float(value)
        return math.isfinite(numeric) and numeric != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "active",
            "enabled",
            "approved",
            "published",
        }
    return False


def _membership_row_is_active(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status not in (None, ""):
        status_value = str(_relation_id(status)).strip().lower()
        if status_value not in {"active", "enabled", "approved", "published"}:
            return False
    for field_name in ["is_active", "active"]:
        if field_name in row and row.get(field_name) is not None and not _truthy_flag(row.get(field_name)):
            return False
    return True


def _membership_row_matches_scan_scope(row: dict[str, Any], scan_context: dict[str, Any]) -> bool:
    scan_member_id = _relation_id(scan_context.get("member"))
    scan_department_id = _relation_id(scan_context.get("department"))

    if scan_member_id:
        member_candidates = [_relation_id(row.get("id")), _relation_id(row.get("member"))]
        if not any(_ids_equal(candidate, scan_member_id) for candidate in member_candidates if candidate is not None):
            return False

    row_department_id = _relation_id(row.get("department"))
    if scan_department_id and row_department_id and not _ids_equal(row_department_id, scan_department_id):
        return False

    return True


def _authenticate_process_user(authorization: str | None, scan_id: str) -> Any:
    token = _bearer_token(authorization)
    if not token:
        logger.info("process_auth_failed scan_id=%s reason=missing_or_malformed_authorization", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")

    try:
        user = directus.get_current_user(token)
    except HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code in {401, 403}:
            logger.info("process_auth_failed scan_id=%s reason=invalid_or_expired_token status_code=%s", scan_id, status_code)
            raise HTTPException(status_code=401, detail="invalid_authorization") from exc
        logger.warning("process_identity_request_failed scan_id=%s status_code=%s", scan_id, status_code)
        raise HTTPException(status_code=502, detail="directus_identity_request_failed") from exc
    except Exception as exc:
        logger.warning("process_identity_request_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail="directus_identity_request_failed") from exc

    user_id = _relation_id((user or {}).get("id"))
    if not user_id:
        logger.info("process_auth_failed scan_id=%s reason=missing_user_id", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")
    user_status = str(_relation_id((user or {}).get("status")) or "").strip().lower()
    if not user_status or user_status != "active":
        logger.info("process_auth_failed scan_id=%s reason=inactive_user", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")
    return user_id


def _resolve_scan_auth_context(scan_id: str) -> dict[str, Any]:
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    try:
        scan_context = directus.get_scan_auth_context(scan_id)
    except HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            raise HTTPException(status_code=404, detail="wellness_scans record not found") from exc
        logger.warning("scan_auth_context_failed scan_id=%s status_code=%s", scan_id, status_code)
        raise HTTPException(status_code=502, detail="scan_auth_context_failed") from exc
    except Exception as exc:
        logger.exception("scan_auth_context_failed scan_id=%s error=%s", scan_id, exc)
        raise HTTPException(status_code=502, detail="scan_auth_context_failed") from exc

    if not scan_context or not _relation_id(scan_context.get("id")):
        raise HTTPException(status_code=404, detail="wellness_scans record not found")
    return scan_context


def _authorize_scan_access(scan_context: dict[str, Any], authenticated_user_id: Any) -> None:
    scan_id = _relation_id(scan_context.get("id"))
    scan_user_id = _relation_id(scan_context.get("user"))
    if not scan_user_id or not _ids_equal(scan_user_id, authenticated_user_id):
        logger.info("process_scan_not_found_or_not_owned scan_id=%s authenticated_user_id=%s", scan_id, authenticated_user_id)
        raise HTTPException(status_code=404, detail="wellness_scans record not found")

    business_profile_id = _relation_id(scan_context.get("business_profile"))
    if not business_profile_id:
        logger.info("process_membership_denied scan_id=%s user_id=%s reason=missing_business_profile", scan_id, authenticated_user_id)
        raise HTTPException(status_code=403, detail="active_membership_required")

    try:
        membership_rows = directus.list_business_profile_members(authenticated_user_id, business_profile_id)
    except Exception as exc:
        logger.warning(
            "process_membership_lookup_failed scan_id=%s user_id=%s business_profile_id=%s error=%s",
            scan_id,
            authenticated_user_id,
            business_profile_id,
            exc,
        )
        raise HTTPException(status_code=403, detail="active_membership_required") from exc

    for row in membership_rows:
        if not _ids_equal(row.get("user"), authenticated_user_id):
            continue
        if not _ids_equal(row.get("business_profile"), business_profile_id):
            continue
        if not _membership_row_is_active(row):
            continue
        if _membership_row_matches_scan_scope(row, scan_context):
            return

    logger.info(
        "process_membership_denied scan_id=%s user_id=%s business_profile_id=%s member_id=%s department_id=%s",
        scan_id,
        authenticated_user_id,
        business_profile_id,
        _relation_id(scan_context.get("member")),
        _relation_id(scan_context.get("department")),
    )
    raise HTTPException(status_code=403, detail="active_membership_required")


def _ensure_scan_media_ready(scan_id: str) -> None:
    try:
        media_row = directus.get_scan_media(scan_id)
    except Exception as exc:
        logger.warning("process_scan_media_lookup_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise HTTPException(status_code=409, detail="scan_media_not_ready") from exc

    if not media_row:
        logger.info("process_scan_media_not_ready scan_id=%s reason=missing_scan_media", scan_id)
        raise HTTPException(status_code=409, detail="scan_media_not_ready")

    required_fields = []
    if VALIDATION_POLICY.require_video:
        required_fields.append("video_file")
    if VALIDATION_POLICY.require_audio:
        required_fields.append("audio_file")
    if getattr(VALIDATION_POLICY, "require_image", False):
        required_fields.append("thumbnail")

    missing_fields = [field for field in required_fields if not _relation_id(media_row.get(field))]
    if missing_fields:
        logger.info("process_scan_media_not_ready scan_id=%s missing_fields=%s", scan_id, missing_fields)
        raise HTTPException(status_code=409, detail="scan_media_not_ready")


def _model_health() -> dict[str, Any]:
    model_path = ml_runtime.model_path
    runtime = get_analyzer_runtime()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "ml_loaded": ml_runtime.is_loaded(),
        "ml_error": ml_runtime.error,
        "configured_model_path": model_path,
        "model_file_exists": bool(model_path and os.path.exists(model_path)),
        "local_model_required": ml_runtime.local_model_required(),
        "directus_configured": directus.is_configured(),
        "validation": {
            "require_video": VALIDATION_POLICY.require_video,
            "require_audio": VALIDATION_POLICY.require_audio,
            "require_face": VALIDATION_POLICY.require_face,
            "require_image": VALIDATION_POLICY.require_image,
            "require_phrase_match": VALIDATION_POLICY.require_phrase_match,
            "phrase_match_threshold": VALIDATION_POLICY.phrase_match_threshold,
        },
        "analyzer_runtime": runtime.health(),
    }


def _should_convert_audio(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() not in [".wav", ".wave"]


def _convert_audio_to_wav(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise ProcessingError(FAILURE_REASON_ANALYSIS_EXCEPTION, "ffmpeg not installed")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-t",
        "3.0",
        "-ac",
        "1",
        "-ar",
        "16000",
        out,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3.5,
        )
    except subprocess.TimeoutExpired as exc:
        remove_temp_file(out)
        raise ProcessingError(FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, "audio conversion timed out") from exc
    except Exception as exc:
        remove_temp_file(out)
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            f"audio conversion failed: {type(exc).__name__}",
        ) from exc
    try:
        if os.path.getsize(out) <= 0:
            raise OSError("empty converted audio")
    except OSError as exc:
        remove_temp_file(out)
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "audio conversion produced no usable output",
        ) from exc
    return out


def _validated_directus_asset_id(asset_id: Any) -> str:
    if not isinstance(asset_id, str):
        raise ValueError("Directus asset id must be a string")
    value = asset_id.strip()
    if (
        not value
        or len(value) > 255
        or any(character.isspace() or ord(character) < 32 for character in value)
        or any(character in value for character in ("/", "\\", "?", "#", "\x00"))
    ):
        raise ValueError("Directus asset id is invalid")
    return value


def _download_directus_asset(asset_id: str, suffix: str) -> str:
    if not directus.is_configured():
        raise ProcessingError(
            FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
            "Directus credentials are not configured",
        )
    normalized_asset_id = _validated_directus_asset_id(asset_id)
    url = f"{directus.base_url}/assets/{normalized_asset_id}"
    download_timeout = (3, 5) if FAST_SCAN_MODE else (10, 30)
    try:
        return download_temp_file(
            url,
            suffix,
            timeout=download_timeout,
        )
    except Exception as exc:
        raise ProcessingError(
            FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
            f"Directus asset download failed: {type(exc).__name__}",
        ) from exc


def _resolve_media_input(
    value: str | None,
    suffix: str,
    media_kind: str,
    *,
    allow_url: bool = True,
    allow_local_path: bool = True,
) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if not isinstance(value, str) or not value.strip():
        raise ProcessingError(
            FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
            f"{media_kind} reference is invalid",
        )
    value = value.strip()
    if allow_local_path and os.path.exists(value):
        return value, False
    try:
        if allow_url and is_url(value):
            path = download_temp_file(value, suffix, timeout=(3, 5) if FAST_SCAN_MODE else (10, 30))
            return path, True
        if is_url(value) or os.path.isabs(value) or any(sep in value for sep in ["/", "\\"]):
            raise ProcessingError(
                FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
                f"{media_kind} must reference a Directus file id",
            )
        path = _download_directus_asset(value, suffix)
        return path, True
    except Exception as exc:
        raise ProcessingError(
            FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
            f"{media_kind} download failed: {exc}",
        ) from exc


def _resolve_media_input_timed(
    scan_id: str,
    value: str | None,
    suffix: str,
    media_kind: str,
    *,
    allow_url: bool = True,
    allow_local_path: bool = True,
) -> tuple[str | None, bool]:
    started_at = time.perf_counter()
    try:
        return _resolve_media_input(
            value,
            suffix,
            media_kind,
            allow_url=allow_url,
            allow_local_path=allow_local_path,
        )
    finally:
        _log_perf(scan_id, f"{media_kind}_resolution_ms", _elapsed_ms(started_at))


def _resolve_scan_context(scan_id: str) -> dict:
    if not directus.is_configured():
        raise HTTPException(
            status_code=500,
            detail="Directus credentials are not configured",
        )
    try:
        scan_context = directus.get_scan_context(scan_id)
    except HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="wellness_scans record not found",
            ) from exc
        logger.warning(
            "scan_context_failed scan_id=%s status_code=%s",
            scan_id,
            status_code,
        )
        raise HTTPException(status_code=502, detail="scan_context_failed") from exc
    except Exception as exc:
        logger.exception(
            "scan_context_failed scan_id=%s error_type=%s",
            scan_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="scan_context_failed") from exc

    if not isinstance(scan_context, dict) or not _relation_id(scan_context.get("id")):
        raise HTTPException(
            status_code=404,
            detail="wellness_scans record not found",
        )
    return scan_context


def _merge_media(scan_context: dict) -> Media:
    directus_values = scan_context.get("resolved_media", {})
    return Media(
        image=directus_values.get("image"),
        audio=directus_values.get("audio"),
        video=directus_values.get("video"),
    )


def _merge_task(scan_context: dict) -> Task | None:
    task_metrics = scan_context.get("task_metrics")
    if not isinstance(task_metrics, dict):
        return None

    reaction_time = task_metrics.get("reaction_time")
    if (
        isinstance(reaction_time, bool)
        or type(reaction_time) not in {int, float}
        or not math.isfinite(float(reaction_time))
        or float(reaction_time) <= 0.0
    ):
        reaction_time = None
    else:
        reaction_time = float(reaction_time)

    errors = task_metrics.get("errors")
    if type(errors) is not int or errors < 0:
        errors = None

    attempts = task_metrics.get("attempts")
    if type(attempts) is not int or attempts <= 0:
        return None

    return Task(
        reaction_time=reaction_time,
        errors=errors,
        attempts=attempts,
    )


def _baseline_for_member(member_id: str | None, business_profile_id: str | None) -> dict | None:
    rows = _baseline_rows_for_member(member_id, business_profile_id)
    if len(rows) != 1:
        if len(rows) > 1:
            logger.warning(
                "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
                member_id,
                business_profile_id,
                len(rows),
            )
        return None
    return rows[0]


def _baseline_rows_for_member(member_id: str | None, business_profile_id: str | None) -> list[dict]:
    if not directus.is_configured() or not member_id or not business_profile_id:
        return []
    try:
        return directus.get_employee_baselines(member_id, business_profile_id)
    except Exception as exc:
        logger.warning("baseline_fetch_failed member_id=%s error=%s", member_id, exc)
        return []


def _authenticate_baseline_status_user(
    authorization: str | None = Header(default=None),
) -> Any:
    """Authenticate a baseline-status caller without trusting query IDs."""
    if not _bearer_token(authorization):
        raise HTTPException(status_code=401, detail="invalid_authorization")
    if directus is None or not directus.is_configured():
        logger.warning("baseline_status_auth_unavailable reason=directus_not_configured")
        raise HTTPException(status_code=503, detail="directus_unavailable")
    return _authenticate_process_user(authorization, "baseline-status")


def _authorize_baseline_status_access(
    authenticated_user_id: Any,
    member_id: str,
    business_profile_id: str,
) -> None:
    """Require an active server-resolved membership for the requested relation."""
    try:
        membership_rows = directus.list_business_profile_members(
            authenticated_user_id,
            business_profile_id,
            require_schema=True,
        )
    except Exception as exc:
        logger.warning(
            "baseline_status_membership_lookup_failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="baseline_status_dependency_failure",
        ) from exc

    target_scope = {"member": member_id}
    for row in membership_rows or []:
        if not _ids_equal(row.get("user"), authenticated_user_id):
            continue
        if not _ids_equal(row.get("business_profile"), business_profile_id):
            continue
        if not _membership_row_is_active(row):
            continue
        if _membership_row_matches_scan_scope(row, target_scope):
            return

    logger.info("baseline_status_access_denied")
    raise HTTPException(status_code=404, detail="baseline_status_not_found")


def _authorized_baseline_for_status(
    member_id: str,
    business_profile_id: str,
) -> dict | None:
    """Read baseline data only after access has been authorized."""
    try:
        return directus.get_employee_baseline(member_id, business_profile_id)
    except Exception as exc:
        logger.warning(
            "baseline_status_lookup_failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="baseline_status_dependency_failure",
        ) from exc


def _directus_processing_ready() -> bool:
    if directus is None or not directus.is_configured():
        logger.warning("readiness_directus_unavailable reason=not_configured")
        return False
    try:
        directus.check_processing_readiness()
    except Exception as exc:
        logger.warning(
            "readiness_directus_check_failed error_type=%s",
            type(exc).__name__,
        )
        return False
    return True


def _durable_settings() -> DurableSettings:
    if not DURABLE_PROCESSING_ENABLED:
        raise DurableConfigurationError("durable processing is disabled")
    return DurableSettings.from_env(role="api")


def _durable_coordinator():
    global _DURABLE_COORDINATOR
    if _DURABLE_COORDINATOR is not None:
        return _DURABLE_COORDINATOR
    settings = _durable_settings()
    if directus is None:
        raise DurableConfigurationError("Directus coordinator is unavailable")
    try:
        import psycopg
        from durable_coordinator import DurableCoordinator
        from processing import PostgresProcessingRepository
        from processing_commit import PostgresProcessingResultCommitter
    except ImportError as exc:
        raise DurableConfigurationError("durable processing dependencies are unavailable") from exc
    connection_factory = lambda: psycopg.connect(settings.database_dsn)
    repository = PostgresProcessingRepository(connection_factory)
    _DURABLE_COORDINATOR = DurableCoordinator(
        repository=repository,
        directus=directus,
        committer=PostgresProcessingResultCommitter(connection_factory),
        internal_secret=settings.internal_secret,
        processing_version=settings.processing_version,
        lease_seconds=settings.lease_seconds,
        heartbeat_seconds=settings.heartbeat_seconds,
    )
    return _DURABLE_COORDINATOR


def _durable_processing_ready() -> bool:
    if not DURABLE_PROCESSING_ENABLED:
        return True
    try:
        _durable_coordinator().check_ready()
    except Exception as exc:
        logger.warning("durable_readiness_failed error_type=%s", type(exc).__name__)
        return False
    return True


def _public_readiness_response(*, ready: bool) -> JSONResponse:
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})


def _authorized_durable_scan(scan_id: str, authenticated_user_id: Any) -> dict[str, Any]:
    scan_context = _resolve_scan_auth_context(scan_id)
    _authorize_scan_access(scan_context, authenticated_user_id)
    return _resolve_scan_context(scan_id)


def _process_scan_durable(scan_id: str, authenticated_user_id: Any) -> JSONResponse:
    try:
        coordinator = _durable_coordinator()
        scan_context = _authorized_durable_scan(scan_id, authenticated_user_id)
        current_status = str(scan_context.get("status") or "").strip()
        if current_status == SCAN_STATUS_COMPLETED:
            return _build_scan_result_response(
                ok=True,
                scan_id=scan_id,
                status="already_completed",
                status_code=200,
            )
        if current_status == SCAN_STATUS_PROCESSING:
            return _build_scan_result_response(
                ok=True,
                scan_id=scan_id,
                status="already_processing",
                status_code=202,
            )
        if current_status != SCAN_STATUS_MEDIA_READY:
            return _build_scan_result_response(
                ok=False,
                scan_id=scan_id,
                error="scan_not_ready",
                current_status=current_status,
                status_code=409,
            )
        _ensure_scan_media_ready(scan_id)
        submission = coordinator.submit_authorized(
            scan_context=scan_context,
            requester_user_id=authenticated_user_id,
        )
        logger.info("durable_job_accepted status=%s", submission.job.status.value)
        return _build_scan_result_response(
            ok=True,
            scan_id=scan_id,
            status="accepted" if submission.created else submission.job.status.value,
            status_code=202,
        )
    except DurableConfigurationError as exc:
        logger.warning("durable_submission_configuration_failed error_type=%s", type(exc).__name__)
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="durable_processing_unavailable",
            status_code=503,
        )
    except HTTPException as exc:
        error = "scan_not_found" if exc.status_code == 404 else "durable_processing_unavailable"
        if exc.status_code == 409:
            error = str(exc.detail)
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error=error,
            status_code=exc.status_code,
        )
    except Exception as exc:
        logger.warning("durable_submission_failed error_type=%s", type(exc).__name__)
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="durable_processing_unavailable",
            status_code=503,
        )


def _require_durable_internal_access(
    x_ai_internal_secret: str | None = Header(default=None),
) -> None:
    if not DURABLE_PROCESSING_ENABLED:
        raise HTTPException(status_code=404, detail="not_found")
    try:
        settings = _durable_settings()
    except DurableConfigurationError:
        raise HTTPException(status_code=503, detail="durable_processing_unavailable") from None
    from durable_coordinator import verify_internal_secret

    if not verify_internal_secret(x_ai_internal_secret, settings.internal_secret):
        raise HTTPException(status_code=401, detail="invalid_internal_authorization")


def _internal_job_payload(job: Any) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "scan_id": job.scan_id,
        "processing_version": job.processing_version,
        "requester_user_id": job.requester_user_id,
        "member_id": job.member_id,
        "business_profile_id": job.business_profile_id,
        "trace_id": job.trace_id,
        "status": job.status.value,
    }


def _identifier_payload(scan_context: dict) -> dict:
    return {
        "user_id": _relation_id(scan_context.get("user")),
        "member_id": _relation_id(scan_context.get("member")),
        "business_profile_id": _relation_id(scan_context.get("business_profile")),
        "department_id": _relation_id(scan_context.get("department")),
    }


def _high_risk_evidence_summary(result: dict, internal_analysis: dict) -> str:
    quality = (internal_analysis or {}).get("quality") or {}
    warnings = set(quality.get("warnings") or result.get("validation_warnings") or [])
    parts: list[str] = []
    if "sustained_eye_closure" in warnings:
        parts.append("sustained_eye_closure")
    if "speech_not_detected" in warnings:
        parts.append("speech_not_detected")
    if "too_much_silence" in warnings:
        parts.append("too_much_silence")
    if not parts:
        parts.append("score_confidence_policy")
    return ",".join(parts[:4])


def _dispatch_high_risk_notifications(
    *,
    scan_id: str,
    alert: dict | None,
    identifiers: dict,
    risk_level: str,
) -> str:
    if not alert:
        return "not_available"
    alert_id = _relation_id(alert.get("id"))
    if not alert_id:
        return "not_available"
    recipients = directus.list_readiness_alert_recipients(
        business_profile_id=identifiers.get("business_profile_id"),
        target_user_id=identifiers.get("user_id"),
    )
    if not recipients:
        return "not_available"
    created = 0
    for user_id in recipients:
        directus.create_notification(
            user_id=user_id,
            business_profile_id=identifiers.get("business_profile_id"),
            alert_id=alert_id,
            scan_id=scan_id,
            member_id=identifiers.get("member_id"),
            risk_level=risk_level,
        )
        created += 1
    return "attempted" if created else "not_available"


def _expected_phrase(scan_context: dict) -> str | None:
    return _safe_string(scan_context.get("expected_phrase"), max_len=500)


def _wellness_scan_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    directus.clear_schema_cache("wellness_scans")
    original = dict(payload)
    candidate = dict(payload)
    if "failure_reason" in candidate:
        candidate["failure_reason"] = _field_length_checked_string(
            "wellness_scans",
            "failure_reason",
            candidate.get("failure_reason"),
            default_max_len=255,
            required=False,
        )
    if "failure_message" in candidate:
        candidate["failure_message"] = _field_length_checked_string(
            "wellness_scans",
            "failure_message",
            candidate.get("failure_message"),
            default_max_len=65535,
            required=False,
        )
    if "ai_model_version" in candidate:
        candidate["ai_model_version"] = _field_length_checked_string(
            "wellness_scans",
            "ai_model_version",
            MODEL_VERSION,
            fallback=MODEL_VERSION,
            default_max_len=255,
            required=False,
        )

    filtered = directus.filter_payload_fields("wellness_scans", candidate)
    fallback_field = directus.first_supported_field("wellness_scans", ["failure_message", "user_message"])
    if "failure_message" in candidate and "failure_message" not in filtered and fallback_field:
        filtered[fallback_field] = candidate["failure_message"]
    cleaned = sanitize_payload(filtered)
    for field_name in ["failure_reason", "failure_message", "completed_at", "ai_model_version", "status"]:
        if field_name in filtered and filtered[field_name] is None and original.get(field_name) is None:
            cleaned[field_name] = None
    if fallback_field and fallback_field in filtered and filtered[fallback_field] is None and original.get("failure_message") is None:
        cleaned[fallback_field] = None
    return cleaned


def _scan_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered = directus.filter_payload_fields("scan_results", payload)
    metadata_field = directus.first_supported_field("scan_results", ["analysis_metadata"])
    if metadata_field and metadata_field not in filtered:
        extras = {}
        for key in [
            "spoken_transcript",
            "expected_phrase",
            "phrase_match_score",
            "audio_quality_score",
            "video_quality_score",
            "image_quality_score",
            "validation_warnings",
        ]:
            if key in payload:
                extras[key] = payload[key]
        if extras:
            filtered[metadata_field] = extras
    return _directus_payload_preserve_nulls(filtered)


def _safe_analyze(fn, path: str | None, missing_warning: str) -> dict:
    if not path:
        return {"score": None, "details": {"status": "missing", "warnings": [missing_warning]}}
    try:
        result = fn(path)
        return result if isinstance(result, dict) else {"score": None, "details": {"status": "invalid"}}
    except Exception as exc:
        logger.error(
            "analyzer_error analyzer=%s error_type=%s",
            getattr(fn, "__name__", "analyzer"),
            type(exc).__name__,
        )
    return {"score": None, "details": {"status": "error", "warnings": [missing_warning]}}


def _timed_safe_analyze(
    scan_id: str,
    metric_name: str,
    fn,
    path: str | None,
    missing_warning: str,
) -> dict:
    started = time.perf_counter()
    result = _safe_analyze(fn, path, missing_warning)
    _log_perf(scan_id, metric_name, _elapsed_ms(started))
    return result


def _analysis_timeout_placeholder(media_kind: str) -> dict:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(media_kind, "warnings")
    status = "load_failed" if media_kind in {"video", "audio"} else "invalid_image"
    return {
        "score": None,
        "details": {
            "status": status,
            warning_key: [f"{media_kind}_timeout"],
        },
    }


def _analyze_audio_file(path: str | None) -> dict:
    if not path:
        return {"score": None, "details": {"status": "missing", "audio_warnings": ["audio_missing"]}}

    from audio import analyze_audio_worker

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=analyze_audio_worker, args=(path, result_queue), daemon=True)
    process.start()
    process.join(AUDIO_ANALYSIS_WORKER_TIMEOUT_SECONDS)

    if process.is_alive():
        logger.warning(
            "audio_analysis_worker_timeout path=%s timeout_seconds=%s",
            path,
            AUDIO_ANALYSIS_WORKER_TIMEOUT_SECONDS,
        )
        process.terminate()
        process.join(1.0)
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        return {"score": None, "details": {"status": "load_failed", "audio_warnings": ["audio_decode_timeout"]}}

    try:
        payload = result_queue.get(timeout=0.5)
    except queue_module.Empty:
        payload = None
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    if isinstance(payload, dict) and payload.get("ok"):
        result = payload.get("result")
        if isinstance(result, dict):
            return result
        return {"score": None, "details": {"status": "invalid", "audio_warnings": ["audio_decode_failed"]}}

    if process.exitcode not in (0, None):
        logger.warning("audio_analysis_worker_failed exitcode=%s path=%s", process.exitcode, path)
    if isinstance(payload, dict) and payload.get("error"):
        logger.warning("audio_analysis_worker_error path=%s error=%s", path, payload.get("error"))
    return {"score": None, "details": {"status": "load_failed", "audio_warnings": ["audio_decode_failed"]}}


def _transcribe_audio_file(path: str, timeout_seconds: float | None = None) -> str:
    if not path:
        raise RuntimeError("audio_missing")
    if timeout_seconds is None:
        effective_timeout = TRANSCRIPTION_TIMEOUT_SECONDS
    elif (
        isinstance(timeout_seconds, bool)
        or type(timeout_seconds) not in {int, float}
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    else:
        effective_timeout = float(timeout_seconds)
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from audio import transcribe_audio; "
            "print(transcribe_audio(sys.argv[1]))"
        ),
        path,
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("transcription_timeout") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "transcription_failed") from exc
    transcript = (completed.stdout or "").strip()
    if not transcript:
        raise RuntimeError("transcription_failed")
    return transcript


def _transcribe_audio_file_optional(path: str | None, timeout_seconds: float = OPTIONAL_PHRASE_TIMEOUT_SECONDS) -> tuple[str | None, str]:
    if not path:
        return None, "audio_missing"
    try:
        return _transcribe_audio_file(path, timeout_seconds=timeout_seconds), "completed"
    except TimeoutError:
        logger.warning("phrase_transcription_timeout timeout_seconds=%s", timeout_seconds)
        return None, "timeout"
    except Exception as exc:
        logger.warning("phrase_transcription_error error=%s", exc)
        return None, "error"


def _analyze_video_file(path: str | None) -> dict:
    from video import analyze_video

    return analyze_video(path)


def _analyze_face_image(path: str | None) -> dict:
    from vision import analyze_face

    return analyze_face(path)


def _analyze_media(scan_id: str, media: Media) -> tuple[dict, list[str]]:
    if not any([media.image, media.audio, media.video]):
        raise ProcessingError(
            FAILURE_REASON_MISSING_MEDIA,
            "No media linked to scan",
        )

    temp_files: list[str] = []
    try:
        image_path, image_temp = _resolve_media_input(
            media.image,
            ".jpg",
            "image",
        )
        audio_path, audio_temp = _resolve_media_input(
            media.audio,
            ".bin",
            "audio",
        )
        video_path, video_temp = _resolve_media_input(
            media.video,
            ".mp4",
            "video",
        )
        for path, is_temp in [
            (image_path, image_temp),
            (audio_path, audio_temp),
            (video_path, video_temp),
        ]:
            if path and is_temp:
                temp_files.append(path)

        if audio_path and _should_convert_audio(audio_path):
            converted = _convert_audio_to_wav(audio_path)
            temp_files.append(converted)
            audio_path = converted

        _log_step(scan_id, "video_analysis_start", has_video=bool(video_path))
        video = _safe_analyze(
            _analyze_video_file,
            video_path,
            "video_missing",
        )
        _log_step(scan_id, "video_analysis_done", score=video.get("score"))

        _log_step(scan_id, "audio_analysis_start", has_audio=bool(audio_path))
        voice = _safe_analyze(
            _analyze_audio_file,
            audio_path,
            "audio_missing",
        )
        _log_step(scan_id, "audio_analysis_done", score=voice.get("score"))

        _log_step(scan_id, "image_analysis_start", has_image=bool(image_path))
        camera = _safe_analyze(
            _analyze_face_image,
            image_path,
            "image_missing",
        )
        _log_step(scan_id, "image_analysis_done", score=camera.get("score"))

        return {
            "camera": camera,
            "video": video,
            "voice": voice,
        }, temp_files
    except Exception:
        for path in temp_files:
            remove_temp_file(path)
        raise


def _mark_scan_failed(scan_id: str, reason: str, message: str | None = None) -> dict[str, str]:
    payload = _wellness_scan_update_payload(
        {
            "status": SCAN_STATUS_FAILED,
            "failure_reason": _safe_string(reason, fallback=FAILURE_REASON_ANALYSIS_EXCEPTION),
            "failure_message": _safe_string(message or failure_message(reason), max_len=500),
            "completed_at": _utc_now(),
            "ai_model_version": MODEL_VERSION,
        }
    )
    try:
        _retry_directus_call(
            "mark_scan_failed",
            lambda: directus.update_wellness_scan(scan_id, payload),
        )
        _log_step(scan_id, "directus_writeback_done", writeback_status={"wellness_scan": "failed_updated"})
        return {"wellness_scan": "failed_updated"}
    except Exception as exc:
        logger.error("scan_id=%s step=directus_mark_failed_error error_type=%s", scan_id, type(exc).__name__)
        return {"wellness_scan": f"failed:{type(exc).__name__}"}


def _mark_scan_failed_terminal(scan_id: str, reason: str, message: str | None = None) -> dict[str, str]:
    writeback_status = _mark_scan_failed(scan_id, reason, message)
    if writeback_status.get("wellness_scan") != "failed_updated":
        logger.error(
            "scan_id=%s terminal_failure_writeback_unconfirmed reason=%s writeback_status=%s",
            scan_id,
            reason,
            writeback_status,
        )
    return writeback_status


def _quality_failure_response(scan_id: str, quality_result: dict, diagnostics: dict, writeback_status: dict) -> dict:
    explanation = "Available media was too weak for a reliable result. A re-scan is recommended."
    if diagnostics.get("quality", {}).get("media_quality", {}).get("image", {}).get("usable") and not diagnostics.get("quality", {}).get("media_quality", {}).get("video", {}).get("usable") and not diagnostics.get("quality", {}).get("media_quality", {}).get("audio", {}).get("usable"):
        explanation = "Only thumbnail data was available, so this result has limited confidence."
    return {
        "status": "failed",
        "retake_required": True,
        "failure_reason": quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
        "readiness_score": 35 if (quality_result.get("failure_reason") == FAILURE_REASON_MISSING_MEDIA) else 45,
        "risk_level": None,
        "confidence": 0.2,
        "camera_confidence": None,
        "voice_confidence": None,
        "task_performance_score": None,
        "baseline_used": False,
        "confidence_drift": None,
        "face_metrics": None,
        "voice_metrics": None,
        "reaction_metrics": None,
        "explanation": explanation,
        "suggested_action": "rescan_recommended",
        "ai_model_version": MODEL_VERSION,
        "diagnostics": diagnostics,
        "writeback_status": writeback_status,
    }


def _safe_internal_analysis_text(result: dict, internal_analysis: dict) -> str:
    quality = internal_analysis.get("quality") or {}
    warnings = [str(w).replace("_", " ") for w in (quality.get("warnings") or result.get("validation_warnings") or [])[:3]]
    if quality.get("failure_reason") in {"missing_media", "low_quality_media"}:
        summary = "reliable assessment unavailable"
        if warnings:
            summary = f"{summary}: {', '.join(warnings)}"
        return sanitize_text(summary, fallback="reliable assessment unavailable", max_len=255) or "reliable assessment unavailable"
    if warnings:
        return sanitize_text(f"analysis warnings: {', '.join(warnings)}", fallback="analysis available", max_len=255) or "analysis available"
    return "analysis available"


def _build_scan_result_payload(
    scan_id: str,
    result: dict,
    internal_analysis: dict,
) -> dict:
    payload = {
        "scan_id": scan_id,
        "readiness_score": result.get("readiness_score"),
        # This is part of the established scan-results payload contract. Directus
        # schema filtering still removes it on deployments that do not support it.
        "observed_fatigue_score": result.get("observed_fatigue_score"),
        "risk_level": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "camera_confidence": result.get("camera_confidence"),
        "voice_confidence": result.get("voice_confidence"),
        "task_performance_score": result.get("task_performance_score"),
        "explanation": sanitize_text(
            result.get("explanation"),
            fallback="Analysis completed.",
            max_len=65535,
        ),
        "suggested_action": sanitize_text(
            result.get("suggested_action"),
            fallback="review_required",
            max_len=255,
        ),
        "ai_model_version": MODEL_VERSION,
        "confidence_drift": result.get("confidence_drift"),
        "baseline_used": result.get("baseline_used"),
        "face_metrics": result.get("face_metrics"),
        "voice_metrics": result.get("voice_metrics"),
        "reaction_metrics": result.get("reaction_metrics"),
    }

    optional_fields = OPTIONAL_SCAN_RESULT_FIELDS + [
        "spoken_transcript",
        "expected_phrase",
        "phrase_match_score",
        "audio_quality_score",
        "video_quality_score",
        "image_quality_score",
        "validation_warnings",
    ]
    supported_optional = directus.supports_fields(
        "scan_results",
        optional_fields,
    )

    # Validate optional strings even when the current Directus schema does not
    # expose them. This preserves the established diagnostic warning for an
    # overlong value while still omitting unsupported fields from the write.
    optional_text_values = {
        "spoken_transcript": _field_length_checked_string(
            "scan_results",
            "spoken_transcript",
            result.get("spoken_transcript"),
            default_max_len=65535,
            required=False,
        ),
        "expected_phrase": _field_length_checked_string(
            "scan_results",
            "expected_phrase",
            result.get("expected_phrase"),
            default_max_len=65535,
            required=False,
        ),
    }

    result_field_map = {
        "fatigue_evidence_score": "fatigue_evidence_score",
        "phrase_match_score": "phrase_match_score",
        "audio_quality_score": "audio_quality_score",
        "video_quality_score": "video_quality_score",
        "image_quality_score": "image_quality_score",
        "validation_warnings": "validation_warnings",
        "result_status": "result_status",
        "capture_quality_score": "capture_quality_score",
        "measurement_reliability_score": "measurement_reliability_score",
        "personal_deviation_score": "personal_deviation_score",
        "task_completion_status": "task_completion_status",
        "baseline_status_at_inference": "baseline_status_at_inference",
        "baseline_confidence": "baseline_confidence",
        "baseline_eligible": "baseline_eligible",
        "hard_gates_triggered": "hard_gates_triggered",
        "explainable_reasons": "explainable_reasons",
    }
    for field_name, result_key in result_field_map.items():
        if field_name in supported_optional:
            payload[field_name] = result.get(result_key)

    for field_name, value in optional_text_values.items():
        if field_name in supported_optional and value is not None:
            payload[field_name] = value

    if "analysis_metadata" in supported_optional:
        payload["analysis_metadata"] = internal_analysis
    if "media_quality" in supported_optional:
        payload["media_quality"] = (
            internal_analysis.get("quality", {}).get("media_quality")
        )
    if "warnings" in supported_optional:
        payload["warnings"] = internal_analysis.get("warnings")
    if "modality_scores" in supported_optional:
        payload["modality_scores"] = result.get("modality_scores")
    if "fusion_details" in supported_optional:
        payload["fusion_details"] = result.get("fusion_details")
    if "internal_analysis" in supported_optional:
        payload["internal_analysis"] = _safe_internal_analysis_text(
            result,
            internal_analysis,
        )

    return _schema_aware_scan_result_payload(payload)


def _write_quality_failure(scan_id: str, quality_result: dict) -> dict:
    return _mark_scan_failed(
        scan_id,
        quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
    )


def _baseline_personal_deviation_score(result: dict) -> float | None:
    if not isinstance(result, dict):
        return None

    drifts: list[float] = []
    for field_name in ["face_metrics", "voice_metrics", "reaction_metrics"]:
        metrics = result.get(field_name)
        if not isinstance(metrics, dict):
            continue
        metric_drifts = metrics.get("baseline_drifts")
        if not isinstance(metric_drifts, dict):
            continue

        for drift_payload in metric_drifts.values():
            if not isinstance(drift_payload, dict):
                continue
            drift = drift_payload.get("drift")
            if isinstance(drift, bool) or type(drift) not in {int, float}:
                continue
            numeric = float(drift)
            if math.isfinite(numeric):
                drifts.append(abs(numeric))

    if not drifts:
        return None
    return round(sum(drifts) / len(drifts), 4)


def _result_status_from_outcome(
    *,
    quality_result: dict,
    validation_result: dict,
    result: dict,
    baseline_eligibility: dict,
) -> str:
    if result.get("risk_level") == "high_risk":
        return "scored"
    if quality_result.get("retake_required") or result.get("retake_required") or quality_result.get("failure_reason") in {"low_quality_media", "missing_media"}:
        return "retake_required"
    task_completion_status = baseline_eligibility.get("task_completion_status")
    if task_completion_status in {"incomplete_required_speech", "incomplete_required_task"}:
        return "incomplete"
    confidence = result.get("confidence")
    if (
        isinstance(confidence, bool)
        or type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or float(confidence) < 0.45
    ):
        return "low_confidence"
    return "scored"


def _completed_scan_update_payload() -> dict[str, Any]:
    return _wellness_scan_update_payload(
        {
            "status": SCAN_STATUS_COMPLETED,
            "completed_at": _utc_now(),
            "failure_reason": None,
            "failure_message": None,
            "ai_model_version": MODEL_VERSION,
        }
    )


def _recover_completed_scan_if_result_exists(scan_id: str) -> bool | None:
    """Recover a partial write where scan_results succeeded but scan completion did not.

    Returns True when recovery completed, False when no result exists, and None when
    Directus could not be checked or updated.
    """
    try:
        existing_result = _retry_directus_call(
            "check_existing_scan_result",
            lambda: directus.get_scan_result_by_scan_id(scan_id),
        )
    except Exception as exc:
        logger.warning(
            "scan_completion_recovery_lookup_failed scan_id=%s error_type=%s",
            scan_id,
            type(exc).__name__,
        )
        return None

    if not isinstance(existing_result, dict) or not existing_result.get("id"):
        return False

    try:
        completed_payload = _completed_scan_update_payload()
        _retry_directus_call(
            "recover_completed_wellness_scan",
            lambda: directus.update_wellness_scan(
                scan_id,
                completed_payload,
            ),
        )
    except Exception as exc:
        logger.warning(
            "scan_completion_recovery_update_failed scan_id=%s error_type=%s",
            scan_id,
            type(exc).__name__,
        )
        return None

    _log_step(scan_id, "scan_completion_recovered_from_existing_result")
    return True


def _validate_success_result_contract(result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "result payload is not a dictionary",
        )

    if not _result_has_valid_evidence(result):
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid_result_classification",
        )

    readiness_score = result.get("readiness_score")
    confidence = result.get("confidence")
    risk_level = result.get("risk_level")
    suggested_action = result.get("suggested_action")
    explanation = result.get("explanation")

    if type(readiness_score) is not int or not 0 <= readiness_score <= 100:
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid readiness score",
        )
    if (
        isinstance(confidence, bool)
        or type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid confidence",
        )
    if risk_level not in {
        "stable",
        "low_focus",
        "elevated_fatigue",
        "high_risk",
    }:
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid risk level",
        )
    if not isinstance(suggested_action, str) or not suggested_action.strip():
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid suggested action",
        )
    if not isinstance(explanation, str) or not explanation.strip():
        raise ProcessingError(
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            "invalid explanation",
        )


def _write_success(
    *,
    scan_id: str,
    scan_context: dict,
    identifiers: dict,
    result: dict,
    internal_analysis: dict,
) -> dict:
    _validate_success_result_contract(result)

    status: dict[str, Any] = {}
    try:
        scan_result_payload = _build_scan_result_payload(scan_id, result, internal_analysis)
        _log_scan_result_payload_ready(scan_id, scan_result_payload)
        write_mode, scan_result = _retry_directus_call(
            "upsert_scan_result",
            lambda: directus.upsert_scan_result(scan_id, scan_result_payload),
        )
        status["scan_result"] = f"{write_mode}:{_relation_id(scan_result.get('id')) or 'ok'}"
    except Exception as exc:
        logger.error("scan_result_write_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise ProcessingError(FAILURE_REASON_WRITEBACK_FAILED, type(exc).__name__) from exc

    try:
        completed_payload = _completed_scan_update_payload()
        _retry_directus_call(
            "complete_wellness_scan",
            lambda: directus.update_wellness_scan(
                scan_id,
                completed_payload,
            ),
        )
        status["wellness_scan"] = "updated"
    except Exception as exc:
        logger.error("wellness_scan_update_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise ProcessingError(FAILURE_REASON_WRITEBACK_FAILED, type(exc).__name__) from exc

    member_id = identifiers.get("member_id")
    if member_id:
        try:
            directus.update_member_last_result(
                member_id,
                {
                    "last_scan_at": _utc_now(),
                    "last_readiness_score": result.get("readiness_score"),
                    "last_risk_level": result.get("risk_level"),
                },
            )
            status["member"] = "updated"
        except Exception as exc:
            logger.warning("member_update_failed member_id=%s error=%s", member_id, exc)
            status["member"] = f"failed:{exc}"

    try:
        scan_request = directus.update_scan_request_if_needed(
            request_id=None,
            scan_context=scan_context,
            scan_id=scan_id,
        )
        status["scan_request"] = "updated" if scan_request else "skipped"
    except Exception as exc:
        logger.warning("scan_request_update_failed scan_id=%s error=%s", scan_id, exc)
        status["scan_request"] = f"failed:{exc}"

    high_risk_detected = result.get("risk_level") == "high_risk"
    high_risk_evidence = _high_risk_evidence_summary(result, internal_analysis)
    logger.info("[HIGH_RISK] detected=%s", str(bool(high_risk_detected)).lower())
    logger.info("[HIGH_RISK] evidence=%s", high_risk_evidence)
    logger.info("[HIGH_RISK] final_state=%s", result.get("risk_level") or "retake_required")

    alert_result = "skipped"
    notification_dispatch = "not_available"
    alert = None
    try:
        alert = directus.create_alert_if_needed(
            risk_level=result["risk_level"],
            confidence=float(result["confidence"] or 0.0),
            scan_id=scan_id,
            member_id=identifiers.get("member_id"),
            business_profile_id=identifiers.get("business_profile_id"),
            department_id=identifiers.get("department_id"),
            user_id=identifiers.get("user_id"),
        )
        alert_result = "created" if alert else "skipped"
        status["alert"] = alert_result
    except Exception as exc:
        logger.warning("alert_write_failed scan_id=%s error=%s", scan_id, exc)
        alert_result = "failed"
        notification_dispatch = "failed" if high_risk_detected else "not_available"
        status["alert"] = f"failed:{exc}"
    if high_risk_detected and alert:
        try:
            notification_dispatch = _dispatch_high_risk_notifications(
                scan_id=scan_id,
                alert=alert,
                identifiers=identifiers,
                risk_level=result["risk_level"],
            )
            status["notification_dispatch"] = notification_dispatch
        except Exception as exc:
            logger.warning("notification_dispatch_failed scan_id=%s error=%s", scan_id, exc)
            notification_dispatch = "failed"
            status["notification_dispatch"] = f"failed:{exc}"
    logger.info("[HIGH_RISK] alert_result=%s", alert_result)
    logger.info("[HIGH_RISK] notification_dispatch=%s", notification_dispatch)
    return status


def _critical_validation_errors_allow_result(critical_errors: list[str] | None) -> bool:
    errors = set(critical_errors or [])
    if not errors:
        return True
    return errors.issubset({"missing_media", "unreadable_media"})


def _face_eye_evidence_unreliable(warnings: list[str] | None, video_details: dict | None, image_details: dict | None) -> bool:
    warning_set = set(warnings or [])
    if warning_set & {"face_not_visible", "subject_not_visible", "landmark_detection_failed", "insufficient_usable_frames"}:
        return True
    video_details = video_details or {}
    image_details = image_details or {}
    face_frames = video_details.get("face_frames")
    valid_face_frames = (
        face_frames
        if type(face_frames) is int and face_frames >= 0
        else 0
    )
    if (
        video_details.get("reliable_eye_landmarks") is False
        and valid_face_frames <= 0
    ):
        return True
    if image_details and image_details.get("face_detected") is False and image_details.get("avg_ear") is None:
        return True
    return False


def _required_modality_gate(
    quality_result: dict,
    *,
    timed_out_modalities: list[str],
) -> tuple[str | None, list[str]]:
    media_quality = quality_result.get("media_quality") or {}
    required_modalities: list[str] = []
    for modality, required in [
        ("video", VALIDATION_POLICY.require_video),
        ("audio", VALIDATION_POLICY.require_audio),
        ("image", VALIDATION_POLICY.require_image),
    ]:
        if not required:
            continue
        required_modalities.append(modality)
        if modality in timed_out_modalities:
            if modality == "audio":
                return FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, required_modalities
            if modality == "video":
                return FAILURE_REASON_VIDEO_MISSING, required_modalities
            if modality == "image":
                return FAILURE_REASON_IMAGE_MISSING, required_modalities
            return FAILURE_REASON_MISSING_MEDIA, required_modalities
        modality_quality = media_quality.get(modality) or {}
        if modality == "audio" and "audio_decode_timeout" in set(modality_quality.get("warnings") or []):
            return FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, required_modalities
        if not modality_quality.get("present"):
            if modality == "audio":
                return FAILURE_REASON_AUDIO_MISSING, required_modalities
            if modality == "video":
                return FAILURE_REASON_VIDEO_MISSING, required_modalities
            if modality == "image":
                return FAILURE_REASON_IMAGE_MISSING, required_modalities
            return FAILURE_REASON_MISSING_MEDIA, required_modalities
        if not modality_quality.get("usable"):
            return FAILURE_REASON_LOW_QUALITY_MEDIA, required_modalities
    return None, required_modalities


def _required_full_multimodal_evidence_failure(
    *,
    result: dict[str, Any],
    worker_states: dict[str, dict[str, Any]],
    valid_modalities: list[str],
) -> str | None:
    if not (VALIDATION_POLICY.require_video and VALIDATION_POLICY.require_audio and VALIDATION_POLICY.require_image):
        return None
    required = {"video", "audio", "image"}
    valid = set(valid_modalities)
    if not required.issubset(valid):
        if "audio" not in valid:
            audio_state = worker_states.get("audio") or {}
            if audio_state.get("timed_out"):
                return FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT
            return FAILURE_REASON_AUDIO_MISSING
        if "video" not in valid:
            return FAILURE_REASON_VIDEO_MISSING
        if "image" not in valid:
            return FAILURE_REASON_IMAGE_MISSING
        return FAILURE_REASON_MISSING_MEDIA
    if not _finite_result_score(result.get("voice_confidence")):
        return FAILURE_REASON_AUDIO_MISSING
    return None


def _process_scan_sync(scan_id: str) -> dict[str, Any]:
    total_started = time.perf_counter()
    _log_step(scan_id, "validation_start")
    stage_started = time.perf_counter()
    scan_context = _resolve_scan_context(scan_id)
    _log_perf(scan_id, "scan_context_resolution_ms", _elapsed_ms(stage_started))
    _log_step(scan_id, "scan_context_loaded", status=scan_context.get("status"))

    media_row = scan_context.get("scan_media") or directus.get_scan_media(scan_id)
    _log_step(
        scan_id,
        "scan_media_loaded",
        media_found=media_row is not None,
        video_file_id=_relation_id((media_row or {}).get("video_file")),
        audio_file_id=_relation_id((media_row or {}).get("audio_file")),
        thumbnail_id=_relation_id((media_row or {}).get("thumbnail")),
    )

    if media_row is None:
        scan_context["scan_media"] = None
        scan_context["resolved_media"] = {"image": None, "audio": None, "video": None}
        _log_step(scan_id, "media_missing_continue_with_unknown_result")

    if ml_runtime.local_model_required() and not ml_runtime.is_loaded():
        validation_result = fail_validation(FAILURE_REASON_MODEL_NOT_LOADED)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()
        terminal_writeback_status = _mark_scan_failed_terminal(
            scan_id,
            validation_result["failure_reason"],
            validation_result["failure_message"],
        )
        validation_result["writeback_status"] = terminal_writeback_status
        _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {"status": SCAN_STATUS_FAILED, **validation_result}

    media = _merge_media(scan_context)
    task = _merge_task(scan_context)
    identifiers = _identifier_payload(scan_context)
    baseline_rows = _baseline_rows_for_member(identifiers.get("member_id"), identifiers.get("business_profile_id"))
    if len(baseline_rows) > 1:
        logger.warning(
            "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
            identifiers.get("member_id"),
            identifiers.get("business_profile_id"),
            len(baseline_rows),
        )
    baseline = baseline_rows[0] if len(baseline_rows) == 1 else None
    baseline_status = baseline_status_payload(baseline)
    expected_phrase = _expected_phrase(scan_context)

    temp_files: list[str] = []
    download_executor: concurrent.futures.ThreadPoolExecutor | None = None
    transcript: str | None = None
    phrase_score: float | None = None
    phrase_status = "not_required"
    try:
        _log_step(scan_id, "media_download_start")
        stage_started = time.perf_counter()
        download_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="media-download",
        )
        download_futures = {
            "image": download_executor.submit(_resolve_media_input_timed, scan_id, media.image, ".jpg", "image", allow_url=False, allow_local_path=False),
            "audio": download_executor.submit(_resolve_media_input_timed, scan_id, media.audio, ".bin", "audio", allow_url=False, allow_local_path=False),
            "video": download_executor.submit(_resolve_media_input_timed, scan_id, media.video, ".mp4", "video", allow_url=False, allow_local_path=False),
        }
        done, not_done = concurrent.futures.wait(
            list(download_futures.values()),
            timeout=FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS if FAST_SCAN_MODE else None,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        if not_done:
            logger.warning(
                "media_download_timeout scan_id=%s timeout_seconds=%s pending=%s",
                scan_id,
                FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS,
                len(not_done),
            )
            for future in not_done:
                future.add_done_callback(_cleanup_download_future)
        image_path = audio_path = video_path = None
        image_temp = audio_temp = video_temp = False
        for name, future in download_futures.items():
            if future not in done:
                continue
            try:
                path, is_temp = future.result()
            except Exception as exc:
                logger.warning("media_download_failed scan_id=%s media=%s error=%s", scan_id, name, exc)
                path, is_temp = None, False
            if name == "image":
                image_path, image_temp = path, is_temp
            elif name == "audio":
                audio_path, audio_temp = path, is_temp
            else:
                video_path, video_temp = path, is_temp
        download_executor.shutdown(wait=False, cancel_futures=True)
        download_executor = None
        for path, is_temp in [(image_path, image_temp), (audio_path, audio_temp), (video_path, video_temp)]:
            if path and is_temp:
                temp_files.append(path)
        if audio_path and _should_convert_audio(audio_path):
            conversion_started = time.perf_counter()
            converted = _convert_audio_to_wav(audio_path)
            _log_perf(scan_id, "audio_conversion_ms", _elapsed_ms(conversion_started))
            temp_files.append(converted)
            audio_path = converted
        else:
            _log_perf(scan_id, "audio_conversion_ms", 0)
        resolved_media = Media(image=image_path, audio=audio_path, video=video_path)
        media_resolution_ms = _elapsed_ms(stage_started)
        _log_perf(scan_id, "media_resolution_total_ms", media_resolution_ms)
        _log_perf(scan_id, "media_download_ms", media_resolution_ms)
        _log_step(
            scan_id,
            "media_download_done",
            has_video=bool(video_path),
            has_audio=bool(audio_path),
            has_image=bool(image_path),
        )

        _log_step(scan_id, "video_validation_start")
        media_validation_started = time.perf_counter()
        stage_started = media_validation_started
        analysis_results, worker_states = _run_parallel_analysis(
            scan_id,
            resolved_media,
        )
        validation_post_runtime_started = time.perf_counter()
        timed_out_modalities = [
            modality
            for modality, state in worker_states.items()
            if state.get("timed_out")
        ]
        video_result = analysis_results["video"]
        _log_step(scan_id, "video_validation_done", quality_score=((video_result.get("details") or {}).get("visual_quality_score")))

        _log_step(scan_id, "face_validation_start")
        stage_started = time.perf_counter()
        video_details = (video_result.get("details") or {})
        image_result = analysis_results.get("image")
        _log_step(
            scan_id,
            "face_validation_done",
            video_face_ratio=video_details.get("face_or_subject_visibility") or video_details.get("face_rate"),
            image_face_detected=((image_result or {}).get("details") or {}).get("face_detected"),
        )

        _log_step(scan_id, "audio_validation_start")
        stage_started = time.perf_counter()
        audio_result = analysis_results["audio"]
        audio_timings = ((audio_result.get("details") or {}).get("timings_ms") or {}) if isinstance(audio_result, dict) else {}
        _log_perf(scan_id, "audio_decode_ms", audio_timings.get("audio_decode_ms"))
        _log_perf(scan_id, "audio_quality_ms", audio_timings.get("audio_quality_ms"))
        _log_perf(scan_id, "voice_activity_ms", audio_timings.get("voice_activity_ms"))
        audio_remaining_ms = max(
            0,
            _elapsed_ms(stage_started)
            - _timing_ms_value(audio_timings.get("audio_decode_ms"))
            - _timing_ms_value(audio_timings.get("audio_quality_ms"))
            - _timing_ms_value(audio_timings.get("voice_activity_ms")),
        )
        if audio_remaining_ms:
            logger.info("[PERF] audio_validation_overhead_ms scan_id=%s value=%s", scan_id, audio_remaining_ms)
        if "audio" in timed_out_modalities:
            _log_step(scan_id, "audio_validation_timeout", quality_score=((audio_result.get("details") or {}).get("audio_quality_score")))
        else:
            _log_step(scan_id, "audio_validation_done", quality_score=((audio_result.get("details") or {}).get("audio_quality_score")))
        _log_perf(scan_id, "media_validation_wall_ms", _elapsed_ms(media_validation_started))
        _log_step(scan_id, "phrase_validation_start")
        stage_started = time.perf_counter()
        audio_analysis_timed_out = "audio" in timed_out_modalities
        if expected_phrase and audio_analysis_timed_out:
            phrase_status = "audio_timeout"
        elif expected_phrase and resolved_media.audio:
            if VALIDATION_POLICY.require_phrase_match:
                transcript, phrase_status = _transcribe_audio_file_optional(
                    resolved_media.audio
                )
            else:
                phrase_status = "skipped_optional"
        elif expected_phrase:
            phrase_status = "audio_missing"
        phrase_expected_for_validation = expected_phrase if (VALIDATION_POLICY.require_phrase_match or transcript) else None
        phrase_validation = validate_scan_inputs(
            policy=VALIDATION_POLICY,
            media=resolved_media,
            video_result=video_result,
            audio_result=audio_result,
            image_result=image_result,
            expected_phrase=phrase_expected_for_validation,
            transcript=transcript,
        )
        phrase_score = phrase_validation["quality_scores"].get("phrase_match")
        phrase_failure_reason = None
        if VALIDATION_POLICY.require_phrase_match and not transcript:
            phrase_failure_reason = phrase_validation.get("failure_reason") or FAILURE_REASON_TRANSCRIPTION_FAILED
        _log_perf(scan_id, "phrase_optional_ms", _elapsed_ms(stage_started))
        _log_step(scan_id, "phrase_validation_done", transcript_present=bool(transcript), phrase_match_score=phrase_score, phrase_status=phrase_status)

        _log_step(scan_id, "image_validation_start")
        _log_step(
            scan_id,
            "image_validation_done",
            quality_score=((image_result or {}).get("details") or {}).get("image_quality_score"),
        )

        raw_signals = {
            "camera": image_result or {"score": None, "details": {"status": "missing"}},
            "video": video_result,
            "voice": audio_result,
        }
        quality_result = assess_quality(
            raw_signals,
            task,
            speech_required=VALIDATION_POLICY.require_phrase_match or bool(expected_phrase),
        )
        combined_warnings = []
        combined_warnings.extend(quality_result.get("warnings") or [])
        combined_warnings.extend(phrase_validation.get("warnings") or [])
        combined_warnings.extend(phrase_validation.get("critical_errors") or [])
        if "unreadable_media" in (phrase_validation.get("critical_errors") or []):
            combined_warnings.append("missing_media")
        quality_result["warnings"] = list(dict.fromkeys(warning for warning in combined_warnings if warning))
        if phrase_validation.get("warnings") or phrase_validation.get("critical_errors"):
            quality_result["weak"] = True
            quality_result["status"] = "weak"
        if phrase_validation.get("failure_reason") in {"missing_media", "unreadable_media"}:
            quality_result["failure_reason"] = "missing_media"
            quality_result["retake_required"] = True
            quality_result["suggested_action"] = "rescan_recommended"

        raw_valid_modalities = [
            modality
            for modality, result in analysis_results.items()
            if _raw_analysis_has_valid_evidence(
                modality,
                result,
                worker_states.get(modality) or {},
            )
        ]
        valid_modalities = list(raw_valid_modalities)
        analyzer_error_modalities = [
            modality
            for modality, state in worker_states.items()
            if state.get("analyzer_error")
        ]
        running_modalities = [
            modality
            for modality, state in worker_states.items()
            if state.get("final_alive")
        ]
        all_workers_terminal = not running_modalities
        all_modalities_timed_out = bool(worker_states) and len(timed_out_modalities) == len(worker_states)
        terminal_reason = "validation_passed"
        if all_modalities_timed_out:
            terminal_failure_reason = "validation_timeout"
            terminal_reason = "validation_timeout"
            _required_modalities = ["video", "audio", "image"]
        elif analyzer_error_modalities and len(analyzer_error_modalities) == len(worker_states) and not valid_modalities:
            terminal_failure_reason = FAILURE_REASON_ANALYSIS_EXCEPTION
            terminal_reason = "analysis_worker_error"
        else:
            terminal_failure_reason, _required_modalities = _required_modality_gate(
                quality_result,
                timed_out_modalities=timed_out_modalities,
            )
            if phrase_failure_reason and terminal_failure_reason is None:
                terminal_failure_reason = phrase_failure_reason
                terminal_reason = "phrase_validation_failed"
            if terminal_failure_reason == FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT:
                terminal_reason = "audio_validation_timeout"
            elif terminal_failure_reason:
                terminal_reason = "validation_timeout" if timed_out_modalities else "validation_no_reliable_evidence"
            elif quality_result.get("usable_modalities", 0) <= 0:
                terminal_failure_reason = quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA
                terminal_reason = "validation_timeout" if timed_out_modalities else "validation_no_reliable_evidence"
        if not all_workers_terminal and terminal_failure_reason is None:
            terminal_failure_reason = FAILURE_REASON_ANALYSIS_EXCEPTION
            terminal_reason = "workers_not_terminal"
        _log_validation_decision(
            scan_id,
            valid_modalities=valid_modalities,
            timed_out_modalities=timed_out_modalities,
            analyzer_error_modalities=analyzer_error_modalities,
            terminal_reason=terminal_reason,
        )
        _log_validation_lifecycle(scan_id, all_workers_terminal=all_workers_terminal, running_modalities=running_modalities)
        _log_perf(scan_id, "validation_post_runtime_ms", _elapsed_ms(validation_post_runtime_started))

        if terminal_failure_reason:
            validation_result = fail_validation(terminal_failure_reason, warnings=quality_result.get("warnings"))
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            terminal_writeback_status = _mark_scan_failed_terminal(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            validation_result["writeback_status"] = terminal_writeback_status
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}

        critical_errors = phrase_validation.get("critical_errors") or []
        if critical_errors and not _critical_validation_errors_allow_result(critical_errors):
            validation_result = fail_validation(critical_errors[0], warnings=quality_result.get("warnings"))
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            terminal_writeback_status = _mark_scan_failed_terminal(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            validation_result["writeback_status"] = terminal_writeback_status
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}

        if terminal_failure_reason or _face_eye_evidence_unreliable(quality_result.get("warnings"), video_details, ((image_result or {}).get("details") or {})):
            quality_result["failure_reason"] = terminal_failure_reason or quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA
            quality_result["retake_required"] = True
            quality_result["suggested_action"] = "rescan_recommended"
            quality_result["status"] = "weak"
            quality_result["weak"] = True
            _log_step(
                scan_id,
                "validation_retake_required",
                reason=terminal_reason if terminal_failure_reason else "unreliable_face_eye_evidence",
                warnings=quality_result.get("warnings"),
            )

        _log_step(scan_id, "validation_completed", scores=phrase_validation["quality_scores"], warnings=quality_result.get("warnings"))
        _log_step(scan_id, "analysis_start")
        stage_started = time.perf_counter()
        fusion_started = time.perf_counter()
        feature_map, _ = features_from_signals(raw_signals, task=task)
        feature_vector = vector_from_features(feature_map)
        _log_perf(scan_id, "feature_fusion_ms", _elapsed_ms(fusion_started))
        classification_started = time.perf_counter()
        ml_result = ml_runtime.predict(feature_vector)
        preview_result = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality_result,
            ml_result=ml_result,
        )
        baseline_used = baseline_ready_for_personalized_scoring(
            baseline,
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=preview_result,
            task=task,
            expected_phrase=expected_phrase,
            unique_row=len(baseline_rows) == 1,
        )
        result = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=baseline_used,
            quality=quality_result,
            ml_result=ml_result,
        )
        _log_perf(scan_id, "classification_ms", _elapsed_ms(classification_started))
        result.update(
            {
                "spoken_transcript": transcript,
                "expected_phrase": expected_phrase,
                "phrase_match_score": phrase_score,
                "audio_quality_score": phrase_validation["quality_scores"].get("audio"),
                "video_quality_score": phrase_validation["quality_scores"].get("video"),
                "image_quality_score": phrase_validation["quality_scores"].get("image"),
                "validation_warnings": quality_result.get("warnings"),
            }
        )
        baseline_eligibility = evaluate_baseline_eligibility(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            signals=raw_signals,
            expected_phrase=expected_phrase,
            task=task,
            manually_unreliable=False,
        )
        result.update(
            {
                "capture_quality_score": baseline_eligibility.get("capture_quality_score"),
                "measurement_reliability_score": baseline_eligibility.get("measurement_reliability_score"),
                "personal_deviation_score": _baseline_personal_deviation_score(result),
                "task_completion_status": baseline_eligibility.get("task_completion_status"),
                "baseline_status_at_inference": baseline_status.get("baseline_status"),
                "baseline_confidence": baseline_status.get("baseline_confidence"),
                "baseline_eligible": baseline_eligibility.get("eligible"),
                "hard_gates_triggered": baseline_eligibility.get("hard_gates_triggered"),
                "explainable_reasons": baseline_eligibility.get("reasons"),
            }
        )
        result["result_status"] = _result_status_from_outcome(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            baseline_eligibility=baseline_eligibility,
        )
        valid_modalities = _fused_result_valid_modalities(
            result,
            raw_valid_modalities=raw_valid_modalities,
        )
        _log_full_evidence_gate(
            scan_id,
            result=result,
            worker_states=worker_states,
            valid_modalities=valid_modalities,
        )
        full_evidence_failure = _required_full_multimodal_evidence_failure(
            result=result,
            worker_states=worker_states,
            valid_modalities=valid_modalities,
        )
        if full_evidence_failure:
            validation_result = fail_validation(full_evidence_failure, warnings=quality_result.get("warnings"))
            _log_validation_decision(
                scan_id,
                valid_modalities=valid_modalities,
                timed_out_modalities=timed_out_modalities,
                terminal_reason="required_full_multimodal_evidence_missing",
            )
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            terminal_writeback_status = _mark_scan_failed_terminal(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            validation_result["writeback_status"] = terminal_writeback_status
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}
        if not _result_has_valid_evidence(result):
            validation_result = fail_validation(quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA, warnings=quality_result.get("warnings"))
            _log_validation_decision(
                scan_id,
                valid_modalities=valid_modalities,
                timed_out_modalities=timed_out_modalities,
                terminal_reason="result_without_real_evidence",
            )
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            terminal_writeback_status = _mark_scan_failed_terminal(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            validation_result["writeback_status"] = terminal_writeback_status
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}
        _log_perf(scan_id, "analysis_ms", _elapsed_ms(stage_started))
        _log_step(scan_id, "analysis_done", risk_level=result.get("risk_level"), confidence=result.get("confidence"))

        internal_analysis = sanitize_payload(
            {
                "quality": quality_result,
                "signals": raw_signals,
                "ml": ml_result,
                "baseline_status_before": baseline_status,
                "validation": phrase_validation,
                "phrase_optional": {
                    "status": phrase_status,
                    "timeout_seconds": OPTIONAL_PHRASE_TIMEOUT_SECONDS,
                    "transcript_present": bool(transcript),
                    "blocking_required": VALIDATION_POLICY.require_phrase_match,
                },
            }
        )

        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()
        writeback_status = _write_success(
            scan_id=scan_id,
            scan_context=scan_context,
            identifiers=identifiers,
            result=result,
            internal_analysis=internal_analysis,
        )
        _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
        _log_step(scan_id, "directus_writeback_done", writeback_status=writeback_status)

        if (
            identifiers.get("member_id")
            and identifiers.get("business_profile_id")
            and baseline_eligibility.get("eligible") is True
        ):
            if len(baseline_rows) > 1:
                logger.warning(
                    "baseline_write_skipped_duplicate member_id=%s business_profile_id=%s scan_id=%s",
                    identifiers.get("member_id"),
                    identifiers.get("business_profile_id"),
                    scan_id,
                )
                writeback_status["baseline"] = "skipped_duplicate"
            else:
                try:
                    baseline_payload = baseline_signal_payload(
                        baseline,
                        signals=raw_signals,
                        scanned_at=_scan_timestamp(scan_context),
                    )
                    baseline_payload["member"] = identifiers["member_id"]
                    baseline_payload["business_profile"] = identifiers[
                        "business_profile_id"
                    ]
                    updated_baseline = directus.upsert_employee_baseline(
                        _relation_id((baseline or {}).get("id")),
                        baseline_payload,
                    )
                    writeback_status["baseline"] = (
                        f"updated:{_relation_id((updated_baseline or {}).get('id')) or 'ok'}"
                    )
                except Exception as exc:
                    logger.warning(
                        "baseline_write_failed scan_id=%s optional=true error_type=%s",
                        scan_id,
                        type(exc).__name__,
                    )
                    writeback_status["baseline"] = (
                        f"failed:{type(exc).__name__}"
                    )

        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {
            "status": SCAN_STATUS_COMPLETED,
            "failure_reason": None,
            "writeback_status": writeback_status,
        }
    except ProcessingError as exc:
        validation_result = fail_validation(exc.reason)
        _log_step(
            scan_id,
            "validation_failed",
            reason=validation_result["failure_reason"],
        )
        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()

        if exc.reason == FAILURE_REASON_WRITEBACK_FAILED:
            recovery = _recover_completed_scan_if_result_exists(scan_id)
            if recovery is True:
                _log_perf(
                    scan_id,
                    "directus_writeback_ms",
                    _elapsed_ms(writeback_started),
                )
                _log_perf(
                    scan_id,
                    "total_process_ms",
                    _elapsed_ms(total_started),
                )
                return {
                    "status": SCAN_STATUS_COMPLETED,
                    "failure_reason": None,
                    "writeback_status": {
                        "scan_result": "existing",
                        "wellness_scan": "recovered",
                    },
                }
            terminal_writeback_status = _mark_scan_failed_terminal(
                scan_id,
                validation_result["failure_reason"],
                validation_result["failure_message"],
            )
            validation_result["writeback_status"] = terminal_writeback_status
            if recovery is None:
                validation_result["writeback_status"]["recovery"] = "unavailable"

        else:
            terminal_writeback_status = _mark_scan_failed_terminal(
                scan_id,
                validation_result["failure_reason"],
                validation_result["failure_message"],
            )
            validation_result["writeback_status"] = terminal_writeback_status
        _log_perf(
            scan_id,
            "directus_writeback_ms",
            _elapsed_ms(writeback_started),
        )
        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {"status": SCAN_STATUS_FAILED, **validation_result}
    finally:
        if download_executor is not None:
            download_executor.shutdown(wait=False, cancel_futures=True)
        for path in temp_files:
            remove_temp_file(path)


def process_scan_background(scan_id: str) -> None:
    if not _claim_active_scan(scan_id):
        _log_step(scan_id, "background_process_skipped_duplicate")
        return

    try:
        result = _process_scan_sync(scan_id)
        _log_step(
            scan_id,
            "background_process_done",
            final_status=result.get("status"),
            failure_reason=result.get("failure_reason"),
        )
    except Exception as exc:
        logger.error("background_process_error scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        _log_step(scan_id, "directus_writeback_start")
        _mark_scan_failed_terminal(
            scan_id,
            FAILURE_REASON_ANALYSIS_EXCEPTION,
            failure_message(FAILURE_REASON_ANALYSIS_EXCEPTION),
        )
    finally:
        _release_active_scan(scan_id)


@app.get("/health")
def health():
    return {"ok": True, "status": "alive"}


@app.get("/ready")
def readiness():
    if not _directus_processing_ready():
        return _public_readiness_response(ready=False)

    runtime = get_analyzer_runtime()
    if not runtime.is_ready():
        logger.warning("readiness_analyzer_unavailable")
        return _public_readiness_response(ready=False)

    if ml_runtime.local_model_required() and not ml_runtime.is_loaded():
        logger.warning("readiness_required_model_unavailable")
        return _public_readiness_response(ready=False)

    if not _durable_processing_ready():
        return _public_readiness_response(ready=False)

    return _public_readiness_response(ready=True)


@app.get("/readiness")
def readiness_alias():
    return readiness()


def debug_scan(scan_id: str):
    scan_context = _resolve_scan_context(scan_id)
    media_row = scan_context.get("scan_media")
    baseline = _baseline_for_member(
        _relation_id(scan_context.get("member")),
        _relation_id(scan_context.get("business_profile")),
    )
    return {
        "scan_id": scan_id,
        "media_row_found": media_row is not None,
        "video_file_id": _relation_id((media_row or {}).get("video_file")),
        "audio_file_id": _relation_id((media_row or {}).get("audio_file")),
        "thumbnail_id": _relation_id((media_row or {}).get("thumbnail")),
        "member": _relation_id(scan_context.get("member")),
        "business_profile": _relation_id(scan_context.get("business_profile")),
        "user": _relation_id(scan_context.get("user")),
        "baseline_scan_count": int((baseline or {}).get("scan_count", 0)),
        "request_source": scan_context.get("request_source"),
        "status": scan_context.get("status"),
    }


if DEBUG_SCAN_ENDPOINT_ENABLED:
    app.get("/debug/scan/{scan_id}")(debug_scan)


@app.get("/process/status/{scan_id}")
def durable_process_status(
    scan_id: str,
    authorization: str | None = Header(default=None),
):
    if not DURABLE_PROCESSING_ENABLED:
        raise HTTPException(status_code=404, detail="not_found")
    normalized_scan_id = scan_id.strip()
    if not normalized_scan_id or len(normalized_scan_id) > 255:
        raise HTTPException(status_code=404, detail="process_status_not_found")
    authenticated_user_id = _authenticate_process_user(authorization, normalized_scan_id)
    _authorized_durable_scan(normalized_scan_id, authenticated_user_id)
    try:
        job = _durable_coordinator().public_status(
            scan_id=normalized_scan_id,
            processing_version=_durable_settings().processing_version,
        )
    except Exception as exc:
        logger.warning("durable_status_lookup_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=404, detail="process_status_not_found") from exc
    return {"scan_id": normalized_scan_id, "status": job.status.value}


@app.post(
    "/internal/processing/jobs/{job_id}/claim",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_processing_claim(job_id: UUID, req: DurableClaimRequest):
    try:
        state, claim, context = _durable_coordinator().claim(
            job_id=job_id,
            worker_id=req.worker_id,
            processing_version=req.processing_version,
            trace_id=req.trace_id,
        )
    except Exception as exc:
        logger.warning("internal_job_claim_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="processing_coordinator_unavailable") from exc
    if claim is None:
        return {"status": state, "job_id": str(job_id)}
    job_payload = _internal_job_payload(claim.job)
    job_payload.update(
        {
            "status": "processing",
            "lease_token": str(claim.attempt.lease_token),
            "attempt_number": claim.attempt.attempt_number,
            "analysis_context": {
                "task_metrics": context.get("task_metrics") if context else None,
                "expected_phrase": context.get("expected_phrase") if context else None,
                "baseline": context.get("baseline") if context else None,
            },
        }
    )
    return job_payload


@app.post(
    "/internal/processing/jobs/{job_id}/heartbeat",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_processing_heartbeat(job_id: UUID, req: DurableHeartbeatRequest):
    try:
        job = _durable_coordinator().heartbeat(
            job_id=job_id,
            worker_id=req.worker_id,
            lease_token=req.lease_token,
        )
    except Exception as exc:
        logger.warning("internal_job_heartbeat_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="processing_lease_lost") from exc
    return {"status": job.status.value, "job_id": str(job.id)}


@app.get(
    "/internal/processing/jobs/{job_id}/input/{asset_role}",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_processing_input(
    job_id: UUID,
    asset_role: str,
    x_ai_worker_id: str | None = Header(default=None),
    x_ai_lease_token: UUID | None = Header(default=None),
):
    if not x_ai_worker_id or x_ai_lease_token is None:
        raise HTTPException(status_code=401, detail="processing_lease_required")
    try:
        content, content_type = _durable_coordinator().read_asset(
            job_id=job_id,
            worker_id=x_ai_worker_id,
            lease_token=x_ai_lease_token,
            asset_role=asset_role,
        )
    except Exception as exc:
        logger.warning("internal_asset_read_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="asset_not_available") from exc
    return Response(content=content, media_type=content_type)


@app.post(
    "/internal/processing/jobs/{job_id}/commit",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_processing_commit(job_id: UUID, req: DurableCommitRequest):
    try:
        from processing_commit import ValidatedResultPayload

        result = ValidatedResultPayload(
            {
                "readiness_score": req.readiness_score,
                "confidence": req.confidence,
                "risk_level": req.risk_level,
                "explanation": req.explanation,
                "suggested_action": req.suggested_action,
            }
        )
        response = _durable_coordinator().commit(
            job_id=job_id,
            scan_id=req.scan_id,
            processing_version=req.processing_version,
            lease_token=req.lease_token,
            result_id=req.result_id,
            result=result,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_result") from exc
    except Exception as exc:
        logger.warning("internal_job_commit_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="processing_commit_rejected") from exc
    return {"status": response.status, "result_id": response.result_ref}


@app.post(
    "/internal/processing/jobs/{job_id}/fail",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_processing_fail(job_id: UUID, req: DurableFailRequest):
    try:
        job = _durable_coordinator().fail(
            job_id=job_id,
            worker_id=req.worker_id,
            lease_token=req.lease_token,
            failure_code=req.failure_code,
            error_type=req.error_type,
            retry_delay_seconds=req.retry_delay_seconds,
            terminal_failure=req.terminal_failure,
        )
    except Exception as exc:
        logger.warning("internal_job_fail_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="processing_failure_rejected") from exc
    return {"status": job.status.value, "job_id": str(job.id)}


@app.post(
    "/internal/processing/workers/register",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_worker_register(req: DurableWorkerRegisterRequest):
    try:
        worker = _durable_coordinator().register_worker(
            worker_id=req.worker_id,
            worker_version=req.worker_version,
            capacity=req.capacity,
        )
    except Exception as exc:
        logger.warning("internal_worker_register_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="worker_registration_unavailable") from exc
    return {"worker_id": worker.worker_id, "lease_token": str(worker.lease_token), "capacity": worker.capacity}


@app.post(
    "/internal/processing/workers/heartbeat",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_worker_heartbeat(req: DurableWorkerHeartbeatRequest):
    try:
        worker = _durable_coordinator().heartbeat_worker(
            worker_id=req.worker_id,
            lease_token=req.lease_token,
            active_slots=req.active_slots,
        )
    except Exception as exc:
        logger.warning("internal_worker_heartbeat_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="worker_liveness_lost") from exc
    return {"worker_id": worker.worker_id, "active_slots": worker.active_slots, "draining": worker.draining}


@app.post(
    "/internal/processing/workers/drain",
    dependencies=[Depends(_require_durable_internal_access)],
)
def internal_worker_drain(req: DurableWorkerDrainRequest):
    try:
        worker = _durable_coordinator().drain_worker(
            worker_id=req.worker_id,
            lease_token=req.lease_token,
        )
    except Exception as exc:
        logger.warning("internal_worker_drain_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=409, detail="worker_liveness_lost") from exc
    return {"worker_id": worker.worker_id, "draining": worker.draining}


@app.get("/baseline/status", response_model=BaselineStatusResponse)
def baseline_status(
    member_id: str = Query(..., min_length=1, max_length=255),
    business_profile_id: str = Query(..., min_length=1, max_length=255),
    authenticated_user_id: str = Depends(_authenticate_baseline_status_user),
):
    _authorize_baseline_status_access(
        authenticated_user_id,
        member_id,
        business_profile_id,
    )
    baseline = _authorized_baseline_for_status(member_id, business_profile_id)
    return baseline_status_payload(baseline)


@app.post("/baseline")
def set_baseline(
    req: BaselineRequest,
    authorization: str | None = Header(default=None),
):
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    if not req.scan_id:
        raise HTTPException(status_code=422, detail="scan_id_required")
    authenticated_user_id = _authenticate_process_user(authorization, req.scan_id.strip())
    scan_context = _resolve_scan_auth_context(req.scan_id.strip())
    _authorize_scan_access(scan_context, authenticated_user_id)
    scan_context = _resolve_scan_context(req.scan_id.strip())
    identifiers = _identifier_payload(scan_context)
    if not identifiers.get("member_id") or not identifiers.get("business_profile_id"):
        raise HTTPException(status_code=422, detail="scan_identifiers_missing")
    media = _merge_media(scan_context)
    baseline_rows = _baseline_rows_for_member(identifiers["member_id"], identifiers["business_profile_id"])
    if len(baseline_rows) > 1:
        logger.warning(
            "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
            identifiers["member_id"],
            identifiers["business_profile_id"],
            len(baseline_rows),
        )
        raise HTTPException(status_code=409, detail="duplicate_baseline_rows")
    baseline = baseline_rows[0] if baseline_rows else None
    if req.media:
        requested = _model_to_dict(req.media)
        trusted = _model_to_dict(media)
        for field_name in ["image", "audio", "video"]:
            candidate = requested.get(field_name)
            if candidate and str(candidate) != str(trusted.get(field_name)):
                raise HTTPException(status_code=422, detail="manual_baseline_requires_directus_media")
    expected_phrase = _expected_phrase(scan_context)
    signals, temp_files = _analyze_media("baseline", media)
    try:
        task = _merge_task(scan_context)
        quality_result = assess_quality(
            signals,
            task,
            speech_required=VALIDATION_POLICY.require_phrase_match or bool(expected_phrase),
        )
        if not quality_result["passed"]:
            raise HTTPException(status_code=422, detail=quality_result["failure_reason"] or FAILURE_REASON_LOW_QUALITY_MEDIA)
        transcript = None
        if expected_phrase:
            try:
                audio_path, audio_temp = _resolve_media_input(media.audio, ".bin", "audio", allow_url=False, allow_local_path=False)
                if audio_path and audio_temp:
                    temp_files.append(audio_path)
                if audio_path and _should_convert_audio(audio_path):
                    converted = _convert_audio_to_wav(audio_path)
                    temp_files.append(converted)
                    audio_path = converted
                transcript = _transcribe_audio_file(audio_path) if audio_path else None
            except Exception:
                transcript = None
        phrase_validation = validate_scan_inputs(
            policy=VALIDATION_POLICY,
            media=media,
            video_result=signals.get("video"),
            audio_result=signals.get("voice"),
            image_result=signals.get("camera"),
            expected_phrase=expected_phrase,
            transcript=transcript,
        )
        result = compute_result(
            signals=signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality_result,
            ml_result=None,
        )
        eligibility = evaluate_baseline_eligibility(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            signals=signals,
            expected_phrase=expected_phrase,
            task=task,
            manually_unreliable=req.manually_unreliable,
        )
        if not eligibility["eligible"]:
            raise HTTPException(status_code=422, detail={"reason": "baseline_ineligible", "reasons": eligibility["reasons"]})
        payload = baseline_signal_payload(
            baseline,
            signals=signals,
            scanned_at=_utc_now(),
        )
        payload["member"] = identifiers["member_id"]
        payload["business_profile"] = identifiers["business_profile_id"]
        updated = directus.upsert_employee_baseline(_relation_id((baseline or {}).get("id")), payload)
        return {
            "scan_id": req.scan_id,
            "member_id": identifiers["member_id"],
            "business_profile_id": identifiers["business_profile_id"],
            "baseline": updated,
            "baseline_status": baseline_status_payload(updated),
            "baseline_eligible": True,
            "model_version": MODEL_VERSION,
        }
    finally:
        for path in temp_files:
            remove_temp_file(path)


@app.post("/process")
def process_scan(
    req: ScanRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    scan_id = (req.scan_id or "").strip()
    logger.info("RECEIVED /process scan_id=%s", scan_id)

    if not scan_id:
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="invalid_scan_id",
            status_code=422,
        )

    try:
        authenticated_user_id = _authenticate_process_user(
            authorization,
            scan_id,
        )
        scan_context = _resolve_scan_auth_context(scan_id)
        _authorize_scan_access(scan_context, authenticated_user_id)
    except HTTPException as exc:
        error = "scan_not_found" if exc.status_code == 404 else "scan_context_failed"
        if exc.status_code == 401:
            error = "invalid_authorization"
        elif exc.status_code == 403:
            error = "active_membership_required"
        return _build_scan_result_response(ok=False, scan_id=scan_id, error=error, status_code=exc.status_code)

    if DURABLE_PROCESSING_ENABLED:
        return _process_scan_durable(scan_id, authenticated_user_id)

    raw_status = scan_context.get("status")
    current_status = raw_status.strip() if isinstance(raw_status, str) else ""
    if current_status == SCAN_STATUS_PENDING:
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="scan_not_ready",
            current_status=current_status,
            status_code=409,
        )
    if current_status in {SCAN_STATUS_PROCESSING, SCAN_STATUS_FAILED}:
        recovery = _recover_completed_scan_if_result_exists(scan_id)
        if recovery is True:
            return _build_scan_result_response(
                ok=True,
                scan_id=scan_id,
                status="recovered_completed",
                status_code=200,
            )
        if current_status == SCAN_STATUS_PROCESSING:
            return _build_scan_result_response(
                ok=True,
                scan_id=scan_id,
                status="already_processing",
                status_code=202,
            )
        return _build_scan_result_response(
            ok=True,
            scan_id=scan_id,
            status="already_failed",
            status_code=200,
        )
    if current_status == SCAN_STATUS_COMPLETED:
        return _build_scan_result_response(
            ok=True,
            scan_id=scan_id,
            status="already_completed",
            status_code=200,
        )
    if current_status != SCAN_STATUS_MEDIA_READY:
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="invalid_scan_status",
            current_status=current_status,
            status_code=409,
        )

    try:
        _ensure_scan_media_ready(scan_id)
    except HTTPException as exc:
        error = exc.detail if exc.detail == "scan_media_not_ready" else "invalid_scan_status"
        return _build_scan_result_response(ok=False, scan_id=scan_id, error=error, status_code=exc.status_code)

    runtime = get_analyzer_runtime()
    if not runtime.is_ready():
        try:
            runtime.start_background()
        except AnalysisRuntimeUnavailable:
            pass
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="analyzer_runtime_not_ready",
            status="analyzer_runtime_not_ready",
            status_code=503,
        )

    update_payload = _wellness_scan_update_payload(
        {
            "status": SCAN_STATUS_PROCESSING,
            "processing_started_at": _utc_now(),
            "failure_reason": None,
            "failure_message": None,
        }
    )
    if "processing_attempts" in scan_context:
        current_attempts = scan_context.get("processing_attempts")
        if type(current_attempts) is not int or current_attempts < 0:
            current_attempts = 0
        update_payload["processing_attempts"] = current_attempts + 1

    try:
        _retry_directus_call(
            "mark_scan_processing",
            lambda: directus.update_wellness_scan(
                scan_id,
                update_payload,
            ),
        )
    except Exception as exc:
        logger.exception("processing_state_update_failed scan_id=%s error=%s", scan_id, exc)
        return _build_scan_result_response(ok=False, scan_id=scan_id, error="processing_state_update_failed", status_code=500)

    background_tasks.add_task(process_scan_background, scan_id)
    return _build_scan_result_response(ok=True, scan_id=scan_id, status="accepted", status_code=202)
