from __future__ import annotations

import math
from statistics import median
from typing import Any

from config import (
    BASELINE_ACTIVE_AFTER,
    BASELINE_HIGH_CONFIDENCE_AFTER,
    BASELINE_PROVISIONAL_AFTER,
    BASELINE_USE_AFTER,
    BASELINE_MAX_STORED_SAMPLES,
    BASELINE_MIN_CAPTURE_QUALITY,
    BASELINE_MIN_CONFIDENCE,
)
from utils import clean_warning_codes, safe_number


BASELINE_SCHEMA_VERSION = 2
BASELINE_VERSION = "robust_v2"
BASELINE_SOURCE = "qualified_calibration_scans"
BASELINE_SAMPLE_LIMIT = BASELINE_MAX_STORED_SAMPLES
BASELINE_MAD_FLOOR = 0.02
BASELINE_STD_TO_MAD = 1.4826
FACE_FEATURES = ["open_eye_aperture", "left_right_eye_asymmetry"]
VOICE_FEATURES = ["normalized_voice_energy", "speech_rate"]
REACTION_FEATURES: list[str] = []
PERSONALIZATION_FEATURES = (
    ("face_avg", "open_eye_aperture"),
    ("face_avg", "left_right_eye_asymmetry"),
    ("voice_avg", "normalized_voice_energy"),
)

BASELINE_CALIBRATION_WARNING_BLOCKLIST = {
    "audio_missing",
    "audio_too_noisy",
    "audio_too_quiet",
    "audio_too_short",
    "face_not_visible",
    "image_blurry",
    "image_missing",
    "image_too_dark",
    "image_low_resolution",
    "insufficient_usable_frames",
    "low_quality_media",
    "missing_media",
    "phrase_mismatch",
    "sustained_eye_closure",
    "subject_not_visible",
    "transcription_failed",
    "unstable_camera",
    "unstable_video",
    "video_blurry",
    "video_missing",
    "video_low_resolution",
    "video_too_dark",
    "video_too_short",
    "speech_not_detected",
    "expected_phrase_missing",
    "fatigue_hard_gate",
}

def _normalized_status(details: dict[str, Any] | None) -> str | None:
    if not isinstance(details, dict):
        return None
    status = details.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    return normalized or None


def _is_finite_number(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) and math.isfinite(float(value))


def _strict_float(value: Any, *, min_value: float | None = None, max_value: float | None = None) -> float | None:
    if not _is_finite_number(value):
        return None
    numeric = float(value)
    if min_value is not None and numeric < min_value:
        return None
    if max_value is not None and numeric > max_value:
        return None
    return numeric


def _legacy_float(value: Any, *, min_value: float | None = None, max_value: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
    elif not _is_finite_number(value):
        return None
    return _strict_float(value, min_value=min_value, max_value=max_value)


def _strict_int(value: Any, *, min_value: int | None = None, max_value: int | None = None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if type(value) is int:
        numeric = value
    else:
        return None
    if min_value is not None and numeric < min_value:
        return None
    if max_value is not None and numeric > max_value:
        return None
    return numeric


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _append_unique(items: list[str], value: str | None) -> None:
    if isinstance(value, str):
        value = value.strip()
        if value and value not in items:
            items.append(value)


def _feature_valid(value: Any, *, non_negative: bool = True, normalized: bool = True) -> float | None:
    if normalized:
        return _strict_float(value, min_value=0.0, max_value=1.0)
    if non_negative:
        return _strict_float(value, min_value=0.0)
    return _strict_float(value)


def _baseline_count(baseline: dict | None) -> int:
    current = _safe_dict(baseline)
    eligible = _strict_int(current.get("eligible_scan_count"), min_value=0)
    if eligible is not None:
        return eligible
    scan_count = _strict_int(current.get("scan_count"), min_value=0)
    if scan_count is not None:
        return scan_count
    return 0


def _baseline_status_from_count(count: int) -> str:
    if count >= BASELINE_ACTIVE_AFTER:
        return "active"
    if count >= BASELINE_PROVISIONAL_AFTER:
        return "provisional"
    return "collecting"


def _baseline_confidence_from_count(count: int) -> float:
    if count <= 0:
        return 0.0
    if count >= BASELINE_HIGH_CONFIDENCE_AFTER:
        return 1.0
    if count < BASELINE_PROVISIONAL_AFTER:
        return round(min(count / float(BASELINE_PROVISIONAL_AFTER), 0.34), 4)
    if count < BASELINE_ACTIVE_AFTER:
        span = max(BASELINE_ACTIVE_AFTER - BASELINE_PROVISIONAL_AFTER, 1)
        progress = (count - BASELINE_PROVISIONAL_AFTER + 1) / float(span + 1)
        return round(0.4 + min(progress * 0.25, 0.25), 4)
    span = max(BASELINE_HIGH_CONFIDENCE_AFTER - BASELINE_ACTIVE_AFTER, 1)
    progress = min((count - BASELINE_ACTIVE_AFTER + 1) / float(span + 1), 1.0)
    return round(0.65 + (progress * 0.3), 4)


def _coerce_float(value: Any) -> float | None:
    return _strict_float(value)


def _clean_sample_list(values: Any) -> list[float]:
    cleaned: list[float] = []
    if not isinstance(values, (list, tuple)):
        return cleaned
    for raw in values:
        value = _strict_float(raw, min_value=0.0)
        if value is not None:
            cleaned.append(value)
    return cleaned[-BASELINE_SAMPLE_LIMIT:]


def _empty_robust_payload(feature_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "feature_stats": {
            name: {
                "median": 0.0,
                "mad": 0.0,
                "count": 0,
            }
            for name in feature_names
        },
    }


def _legacy_feature_seed(value: Any, count: int) -> list[float]:
    numeric = _legacy_float(value, min_value=0.0, max_value=1.0)
    valid_count = _strict_int(count, min_value=0)
    if numeric is None or valid_count is None or valid_count <= 0:
        return []
    sample_count = min(max(valid_count, 1), 5)
    return [float(numeric)] * sample_count


def _legacy_feature_stats(value: Any, std: Any) -> dict[str, float]:
    center = _legacy_float(value, min_value=0.0, max_value=1.0)
    spread = _legacy_float(std, min_value=0.0)
    if center is None or spread is None:
        return {}
    mad = spread / BASELINE_STD_TO_MAD if spread > 0 else 0.0
    return {
        "median": round(float(center), 4),
        "mad": round(max(float(mad), 0.0), 4),
        "count": 1,
    }


def _bucket_count_entry(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    count = _strict_int(value.get("count"), min_value=0)
    return count if count is not None else 0


def _existing_feature_samples(baseline: dict | None, field_name: str, feature_names: list[str]) -> dict[str, list[float]]:
    current = _safe_dict(baseline)
    metadata = _safe_dict(current.get("baseline_metadata"))
    sample_root = _safe_dict(metadata.get("samples"))
    field_samples = _safe_dict(sample_root.get(field_name))
    if not isinstance(field_samples, dict):
        field_samples = {}
    existing = {
        name: _clean_sample_list(field_samples.get(name))
        for name in feature_names
    }
    payload = _safe_dict(current).get(field_name)
    if isinstance(payload, dict) and payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        for name in feature_names:
            if existing[name]:
                continue
            existing[name] = _clean_sample_list((payload.get("feature_samples") or {}).get(name))
    count = _baseline_count(current)
    if isinstance(payload, dict) and payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        legacy_value = payload.get("avg")
        if field_name == "face_avg":
            if not existing["open_eye_aperture"]:
                existing["open_eye_aperture"] = _legacy_feature_seed(legacy_value, count)
            if not existing["left_right_eye_asymmetry"]:
                existing["left_right_eye_asymmetry"] = []
        elif field_name == "voice_avg":
            if not existing["normalized_voice_energy"]:
                existing["normalized_voice_energy"] = _legacy_feature_seed(legacy_value, count)
            if not existing.get("speech_rate"):
                existing["speech_rate"] = []
        elif field_name == "reaction_avg":
            existing = {}
        if field_name in {"face_avg", "voice_avg"}:
            for name in feature_names:
                existing[name] = existing.get(name, [])[-BASELINE_SAMPLE_LIMIT:]
    return existing


def _median_and_mad(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    med = float(median(samples))
    deviations = [abs(sample - med) for sample in samples]
    mad = float(median(deviations)) if deviations else 0.0
    return round(med, 4), round(max(mad, BASELINE_MAD_FLOOR if len(samples) > 1 else 0.0), 4)


def _build_feature_payload(
    feature_names: list[str],
    samples_by_feature: dict[str, list[float]],
) -> dict[str, Any]:
    payload = _empty_robust_payload(feature_names)
    feature_stats: dict[str, dict[str, float]] = {}
    feature_samples: dict[str, list[float]] = {}
    max_count = 0
    for name in feature_names:
        samples = _clean_sample_list(samples_by_feature.get(name))
        med, mad = _median_and_mad(samples)
        feature_stats[name] = {"median": med, "mad": mad, "count": len(samples)}
        if samples:
            feature_samples[name] = [round(value, 6) for value in samples]
            max_count = max(max_count, len(samples))
    payload["feature_stats"] = feature_stats
    if feature_samples:
        payload["feature_samples"] = feature_samples
    return payload


def _bucket_counts(baseline: dict | None) -> dict[str, int]:
    current = _safe_dict(baseline)
    metadata = _safe_dict(current.get("baseline_metadata"))
    counts = metadata.get("bucket_counts")
    if isinstance(counts, dict) and counts:
        return {
            "morning": _bucket_count_entry(counts.get("morning")),
            "midday": _bucket_count_entry(counts.get("midday")),
            "evening": _bucket_count_entry(counts.get("evening")),
        }
    face_avg = _safe_dict(current.get("face_avg"))
    buckets = _safe_dict(face_avg.get("buckets"))
    return {
        "morning": _bucket_count_entry(buckets.get("morning")),
        "midday": _bucket_count_entry(buckets.get("midday")),
        "evening": _bucket_count_entry(buckets.get("evening")),
    }


def _append_feature_sample(sample_map: dict[str, list[float]], feature_name: str, value: float | None) -> None:
    if value is None:
        return
    samples = list(sample_map.get(feature_name) or [])
    samples.append(float(value))
    sample_map[feature_name] = samples[-BASELINE_SAMPLE_LIMIT:]


def current_baseline_features(
    *,
    signals: dict | None,
    result: dict | None = None,
) -> dict[str, dict[str, float | None]]:
    _ = result
    signals = _safe_dict(signals)

    camera_wrapper = _safe_dict(signals.get("camera"))
    voice_wrapper = _safe_dict(signals.get("voice"))
    camera_details = _safe_dict(camera_wrapper.get("details"))
    voice_details = _safe_dict(voice_wrapper.get("details"))

    camera_status = _normalized_status(camera_details)
    voice_status = _normalized_status(voice_details)

    avg_eye = None
    asymmetry = None
    if camera_status == "ok":
        avg_eye = _feature_valid(camera_details.get("avg_ear"), normalized=True)
        left_eye = _feature_valid(camera_details.get("left_eye_aperture"), normalized=True)
        right_eye = _feature_valid(camera_details.get("right_eye_aperture"), normalized=True)
        explicit_asymmetry = _feature_valid(camera_details.get("left_right_eye_asymmetry"), normalized=True)
        if explicit_asymmetry is not None:
            asymmetry = explicit_asymmetry
        elif left_eye is not None and right_eye is not None:
            asymmetry = abs(left_eye - right_eye)

    voice_energy = None
    if voice_status == "ok":
        voice_energy = _feature_valid(voice_details.get("rms_energy"), normalized=True)

    return {
        "face_avg": {
            "open_eye_aperture": avg_eye,
            "left_right_eye_asymmetry": asymmetry,
        },
        "voice_avg": {
            "normalized_voice_energy": voice_energy,
            "speech_rate": None,
        },
        "reaction_avg": {},
    }


def _hard_gates_from_signals(signals: dict | None) -> list[str]:
    return []


def _feature_presence_requirements(signals: dict | None) -> tuple[bool, list[str], dict[str, dict[str, float | None]]]:
    features = current_baseline_features(signals=signals, result=None)
    missing_reasons: list[str] = []
    face = features.get("face_avg") or {}
    voice = features.get("voice_avg") or {}
    face_usable = face.get("open_eye_aperture") is not None
    voice_usable = voice.get("normalized_voice_energy") is not None
    if not face_usable:
        missing_reasons.append("missing_face_baseline_feature")
    if not voice_usable:
        missing_reasons.append("missing_voice_baseline_feature")
    return face_usable and voice_usable, missing_reasons, features


def _task_completion_status(
    *,
    validation_result: dict | None,
    quality_result: dict | None,
    expected_phrase: str | None,
    task: Any = None,
) -> str:
    validation_result = validation_result if isinstance(validation_result, dict) else {}
    quality_result = quality_result if isinstance(quality_result, dict) else {}
    warnings = set(clean_warning_codes(validation_result.get("warnings") or []))
    expected_phrase_value = expected_phrase.strip() if isinstance(expected_phrase, str) and expected_phrase.strip() else None
    if expected_phrase_value and warnings.intersection({"speech_not_detected", "transcription_failed", "expected_phrase_missing", "phrase_mismatch"}):
        return "incomplete_required_speech"
    task_quality = quality_result.get("task_quality")
    task_present = False
    if isinstance(task, dict):
        attempts = _strict_int(task.get("attempts"), min_value=0)
        errors = _strict_int(task.get("errors"), min_value=0)
        reaction_time = _strict_float(task.get("reaction_time"), min_value=0.0)
        if reaction_time is not None and reaction_time <= 0:
            reaction_time = None
        task_present = attempts is not None or errors is not None or reaction_time is not None
    elif task is not None:
        values = []
        for key in ["reaction_time", "errors", "attempts"]:
            try:
                values.append(getattr(task, key, None))
            except Exception:
                values.append(None)
        reaction_time = _strict_float(values[0], min_value=0.0)
        if reaction_time is not None and reaction_time <= 0:
            reaction_time = None
        errors = _strict_int(values[1], min_value=0)
        attempts = _strict_int(values[2], min_value=0)
        task_present = attempts is not None or errors is not None or reaction_time is not None
    if task_present and task_quality in {"weak", "failed"}:
        return "incomplete_required_task"
    if expected_phrase_value or task_present:
        return "completed"
    return "not_required"


def evaluate_baseline_eligibility(
    *,
    quality_result: dict | None,
    validation_result: dict | None,
    result: dict | None,
    signals: dict | None,
    expected_phrase: str | None = None,
    task: Any = None,
    manually_unreliable: bool = False,
) -> dict[str, Any]:
    quality_result = quality_result if isinstance(quality_result, dict) else {}
    validation_result = validation_result if isinstance(validation_result, dict) else {}
    result = result if isinstance(result, dict) else {}
    signals = signals if isinstance(signals, dict) else {}
    features_present, feature_reasons, _ = _feature_presence_requirements(signals)
    hard_gates_triggered = _hard_gates_from_signals(signals)
    task_completion_status = _task_completion_status(
        validation_result=validation_result,
        quality_result=quality_result,
        expected_phrase=expected_phrase,
        task=task,
    )
    reasons: list[str] = []

    quality_warnings = set(clean_warning_codes(quality_result.get("warnings") or []))
    validation_warnings = set(clean_warning_codes(validation_result.get("warnings") or []))
    combined_warnings = quality_warnings | validation_warnings
    capture_quality = _strict_float(_safe_dict(quality_result.get("media_quality")).get("aggregate_quality"), min_value=0.0, max_value=1.0)
    measurement_reliability = _strict_float(quality_result.get("confidence_multiplier"), min_value=0.0, max_value=1.0)
    baseline_confidence = _strict_float(result.get("confidence"), min_value=0.0, max_value=1.0)
    quality_status = quality_result.get("status")
    quality_passed_flag = quality_result.get("passed")
    quality_passed = quality_status == "passed" and (quality_passed_flag is None or quality_passed_flag is True)
    if "passed" in quality_result and not isinstance(quality_passed_flag, bool):
        quality_passed = False
    if quality_passed_flag is False:
        quality_passed = False
    weak_flag = quality_result.get("weak") is True
    retake_flag = quality_result.get("retake_required") is True
    result_retake_flag = result.get("retake_required") is True
    critical_errors = validation_result.get("critical_errors") or []
    if not isinstance(critical_errors, list):
        critical_errors = list(critical_errors) if isinstance(critical_errors, tuple) else [critical_errors]

    if not quality_passed or weak_flag:
        _append_unique(reasons, "insufficient_capture_quality")
    if retake_flag or result_retake_flag:
        _append_unique(reasons, "retake_required")
    if not features_present:
        for reason in feature_reasons:
            _append_unique(reasons, reason)
    if critical_errors:
        _append_unique(reasons, "validation_failure")
    if task_completion_status not in {"completed", "not_required"}:
        _append_unique(reasons, task_completion_status)
    if capture_quality is None or capture_quality < BASELINE_MIN_CAPTURE_QUALITY:
        _append_unique(reasons, "capture_quality_too_low")
    if measurement_reliability is None or measurement_reliability < BASELINE_MIN_CAPTURE_QUALITY:
        _append_unique(reasons, "measurement_reliability_too_low")
    if baseline_confidence is None or baseline_confidence < BASELINE_MIN_CONFIDENCE:
        _append_unique(reasons, "low_confidence")
    if combined_warnings & BASELINE_CALIBRATION_WARNING_BLOCKLIST:
        _append_unique(reasons, "measurement_warning")
    if hard_gates_triggered:
        _append_unique(reasons, "fatigue_hard_gate")
    if str(result.get("risk_level") or "").strip().lower() != "stable":
        _append_unique(reasons, "risk_not_stable")
    if manually_unreliable is True:
        _append_unique(reasons, "manually_unreliable")
    if validation_warnings & {"speech_not_detected", "expected_phrase_missing", "phrase_mismatch", "transcription_failed"}:
        _append_unique(reasons, "speech_required_not_completed")

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "hard_gates_triggered": hard_gates_triggered,
        "task_completion_status": task_completion_status,
        "capture_quality_score": safe_number(capture_quality, 4),
        "measurement_reliability_score": safe_number(measurement_reliability, 4),
    }


def is_scan_eligible_for_baseline(**kwargs: Any) -> bool:
    return bool(evaluate_baseline_eligibility(**kwargs)["eligible"])


def baseline_signal_payload(
    baseline: dict | None,
    *,
    signals: dict,
    scanned_at: str | None = None,
) -> dict:
    current = _safe_dict(baseline)
    next_count = _baseline_count(current) + 1
    features = current_baseline_features(signals=signals)

    face_samples = _existing_feature_samples(current, "face_avg", FACE_FEATURES)
    voice_samples = _existing_feature_samples(current, "voice_avg", VOICE_FEATURES)
    reaction_samples = _existing_feature_samples(current, "reaction_avg", REACTION_FEATURES)

    for name, value in (features.get("face_avg") or {}).items():
        _append_feature_sample(face_samples, name, value)
    for name, value in (features.get("voice_avg") or {}).items():
        _append_feature_sample(voice_samples, name, value)
    for name, value in (features.get("reaction_avg") or {}).items():
        _append_feature_sample(reaction_samples, name, value)

    baseline_status = _baseline_status_from_count(next_count)

    face_avg = _build_feature_payload(FACE_FEATURES, face_samples)
    voice_avg = _build_feature_payload(VOICE_FEATURES, voice_samples)
    reaction_avg = _build_feature_payload(REACTION_FEATURES, reaction_samples)
    return {
        "scan_count": next_count,
        "face_avg": face_avg,
        "voice_avg": voice_avg,
        "reaction_avg": reaction_avg,
        "is_active": baseline_status == "active",
    }


def baseline_status_payload(baseline: dict | None) -> dict:
    baseline = _safe_dict(baseline)
    count = _baseline_count(baseline)
    stored_is_active = baseline.get("is_active") is True
    count_status = _baseline_status_from_count(count)
    is_active = stored_is_active and count >= BASELINE_ACTIVE_AFTER and count_status == "active"
    status = "active" if is_active else ("provisional" if count >= BASELINE_PROVISIONAL_AFTER else "collecting")
    is_provisional = status == "provisional"
    buckets = _bucket_counts(baseline)
    scans_remaining = max(BASELINE_ACTIVE_AFTER - count, 0)
    stored_confidence = _strict_float(baseline.get("baseline_confidence"), min_value=0.0, max_value=1.0)

    if is_active:
        message = "Baseline active. Current scans are compared against the employee's own readiness pattern."
    elif count == 0:
        message = "No baseline yet. Capture qualified scans to start calibration."
    elif is_provisional:
        message = "Baseline is provisional. Keep collecting qualified scans until activation."
    else:
        message = "Calibration started. More qualified scans are needed before personalized scoring is active."

    return {
        "is_active": is_active,
        "scan_count": count,
        "eligible_scan_count": count,
        "scans_remaining": scans_remaining,
        "is_provisional": is_provisional,
        "baseline_status": status,
        "baseline_confidence": stored_confidence if stored_confidence is not None else _baseline_confidence_from_count(count),
        "needs_morning_scan": (buckets.get("morning", 0) == 0),
        "needs_evening_scan": (buckets.get("evening", 0) == 0),
        "message": message,
    }


def baseline_ready_for_scoring(baseline: dict | None) -> bool:
    baseline = baseline if isinstance(baseline, dict) else {}
    status = baseline_status_payload(baseline)
    return baseline.get("is_active") is True and bool(status["is_active"]) and status["scan_count"] >= BASELINE_USE_AFTER


def baseline_has_valid_personalization_references(baseline: dict | None) -> bool:
    baseline = _safe_dict(baseline)
    return any(
        baseline_feature_reference(baseline, field_name, feature_name) is not None
        for field_name, feature_name in PERSONALIZATION_FEATURES
    )


def baseline_ready_for_personalized_scoring(
    baseline: dict | None,
    *,
    quality_result: dict | None = None,
    validation_result: dict | None = None,
    result: dict | None = None,
    task: Any = None,
    expected_phrase: str | None = None,
    unique_row: bool = True,
) -> bool:
    if unique_row is not True:
        return False
    baseline = _safe_dict(baseline)
    if not baseline_ready_for_scoring(baseline):
        return False
    if not baseline_has_valid_personalization_references(baseline):
        return False

    quality_result = quality_result if isinstance(quality_result, dict) else {}
    validation_result = validation_result if isinstance(validation_result, dict) else {}
    result = result if isinstance(result, dict) else {}

    passed_flag = quality_result.get("passed")
    if quality_result.get("status") != "passed" or (passed_flag is not None and not isinstance(passed_flag, bool)) or passed_flag is False or quality_result.get("weak") is True or quality_result.get("retake_required") is True:
        return False
    if validation_result.get("critical_errors"):
        return False
    validation_passed_flag = validation_result.get("passed")
    if validation_passed_flag is not True:
        return False
    if _task_completion_status(
        validation_result=validation_result,
        quality_result=quality_result,
        expected_phrase=expected_phrase,
        task=task,
    ) not in {"completed", "not_required"}:
        return False

    quality_warnings = set(clean_warning_codes(quality_result.get("warnings") or []))
    validation_warnings = set(clean_warning_codes(validation_result.get("warnings") or []))
    if quality_warnings & BASELINE_CALIBRATION_WARNING_BLOCKLIST:
        return False
    if validation_warnings & {"speech_not_detected", "expected_phrase_missing", "phrase_mismatch", "transcription_failed"}:
        return False
    confidence_multiplier = _strict_float(quality_result.get("confidence_multiplier"), min_value=0.0, max_value=1.0)
    if confidence_multiplier is None:
        return False
    if confidence_multiplier < BASELINE_MIN_CAPTURE_QUALITY:
        return False
    aggregate_quality = _strict_float(_safe_dict(quality_result.get("media_quality")).get("aggregate_quality"), min_value=0.0, max_value=1.0)
    if aggregate_quality is None:
        return False
    if aggregate_quality < BASELINE_MIN_CAPTURE_QUALITY:
        return False
    result_confidence = _strict_float(result.get("confidence"), min_value=0.0, max_value=1.0)
    if result_confidence is None:
        return False
    if result_confidence < BASELINE_MIN_CONFIDENCE:
        return False
    return True


def legacy_baseline_stat(baseline: dict | None, key: str) -> dict[str, float] | None:
    payload = _safe_dict(baseline).get(key)
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        return None
    center = _legacy_float(payload.get("avg"), min_value=0.0, max_value=1.0)
    spread = payload.get("std")
    if center is None:
        return None
    if spread is None:
        spread_value = 0.0
    else:
        spread_value = _legacy_float(spread, min_value=0.0)
        if spread_value is None:
            return None
    return {
        "median": center,
        "mad": max((spread_value or 0.0) / BASELINE_STD_TO_MAD, BASELINE_MAD_FLOOR),
    }


def baseline_feature_reference(baseline: dict | None, field_name: str, feature_name: str) -> dict[str, float] | None:
    payload = _safe_dict(baseline).get(field_name)
    if isinstance(payload, dict) and payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        if (field_name, feature_name) not in PERSONALIZATION_FEATURES:
            return None
        stats = _safe_dict(payload.get("feature_stats")).get(feature_name)
        if isinstance(stats, dict):
            center = _strict_float(stats.get("median"), min_value=0.0, max_value=1.0)
            mad = _strict_float(stats.get("mad"), min_value=0.0)
            count = _strict_int(stats.get("count"), min_value=1)
            if center is not None and mad is not None and count is not None:
                return {
                    "median": round(center, 4),
                    "mad": round(max(mad, BASELINE_MAD_FLOOR), 4),
                }
    if field_name == "face_avg" and feature_name == "open_eye_aperture":
        return legacy_baseline_stat(baseline, field_name)
    if field_name == "voice_avg" and feature_name == "normalized_voice_energy":
        return legacy_baseline_stat(baseline, field_name)
    return None
