from __future__ import annotations

import math
from typing import Any

from utils import clamp01, clean_warning_codes, safe_number


FAILURE_REASON_LOW_QUALITY_MEDIA = "low_quality_media"
FAILURE_REASON_MISSING_MEDIA = "missing_media"

_READABLE_STATUSES = {"ok"}
_STRONG_THRESHOLD = 0.72
_EVIDENCE_ONLY_WARNINGS = {"sustained_eye_closure"}
_EVIDENCE_BLOCKING_WARNINGS = {
    "video": {
        "insufficient_usable_frames",
        "subject_not_visible",
        "face_not_visible",
        "landmark_detection_failed",
        "video_too_short",
        "low_quality_media",
    },
    "audio": {
        "speech_not_detected",
        "audio_too_short",
        "low_quality_media",
    },
    "image": {
        "face_not_visible",
        "subject_not_visible",
        "low_quality_media",
    },
}
_UNREADABLE_DIAGNOSTIC_WARNINGS = {
    "video_timeout",
    "audio_timeout",
    "image_timeout",
    "audio_decode_timeout",
}

_QUALITY_THRESHOLDS = {
    "video": 0.42,
    "audio": 0.40,
    "image": 0.38,
}

_VIDEO_DECISION_WARNINGS = {
    "video_too_dark",
    "video_blurry",
    "unstable_video",
    "unstable_camera",
    "insufficient_usable_frames",
    "subject_not_visible",
    "face_not_visible",
    "landmark_detection_failed",
    "video_too_short",
    "video_low_resolution",
    "low_quality_media",
}

_AUDIO_DECISION_WARNINGS = {
    "speech_not_detected",
    "audio_too_noisy",
    "audio_too_quiet",
    "too_much_silence",
    "audio_clipping",
    "audio_too_short",
    "low_quality_media",
}

_IMAGE_DECISION_WARNINGS = {
    "image_too_dark",
    "image_blurry",
    "face_not_visible",
    "subject_not_visible",
    "image_low_resolution",
    "low_quality_media",
}

_VIDEO_RETAKE_WARNINGS = {
    "video_blurry",
    "unstable_video",
    "insufficient_usable_frames",
    "subject_not_visible",
    "face_not_visible",
    "landmark_detection_failed",
}

_AUDIO_RETAKE_WARNINGS = {
    "audio_too_quiet",
    "too_much_silence",
    "audio_clipping",
    "audio_too_noisy",
    "speech_not_detected",
}


def _task_value(task: Any, key: str) -> Any:
    if task is None:
        return None
    try:
        if isinstance(task, dict):
            return task.get(key)
        return getattr(task, key, None)
    except Exception:
        return None


def _task_fields(task: Any) -> tuple[Any, Any, Any]:
    if task is None:
        return None, None, None
    if isinstance(task, dict):
        return task.get("attempts"), task.get("reaction_time"), task.get("errors")
    return (
        _task_value(task, "attempts"),
        _task_value(task, "reaction_time"),
        _task_value(task, "errors"),
    )


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _capture_quality_score(value: Any) -> float | None:
    numeric = _finite_number(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _non_negative_number(value: Any) -> float | None:
    numeric = _finite_number(value)
    if numeric is None or numeric < 0.0:
        return None
    return numeric


def _positive_number(value: Any) -> float | None:
    numeric = _finite_number(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _non_negative_int(value: Any) -> int | None:
    numeric = _finite_number(value)
    if numeric is None or numeric < 0.0 or not numeric.is_integer():
        return None
    return int(numeric)


def _normalized_status(details: dict[str, Any] | None) -> str | None:
    if not isinstance(details, dict):
        return None
    status = details.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().casefold()
    return normalized or None


def _signal_payload(signals: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not isinstance(signals, dict):
        return None
    value = signals.get(key)
    return value if isinstance(value, dict) else None


def _signal_details(signal: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(signal, dict):
        return {}
    details = signal.get("details")
    return details if isinstance(details, dict) else {}


def _details_warnings(details: dict[str, Any], key: str) -> list[str]:
    return clean_warning_codes(details.get(key) or [])


def _decision_warning_set(name: str, *, speech_required: bool, quiet_but_usable: bool) -> set[str]:
    if name == "video":
        return _VIDEO_DECISION_WARNINGS
    if name == "audio":
        allowed = set(_AUDIO_DECISION_WARNINGS)
        if not speech_required:
            allowed.discard("speech_not_detected")
        if quiet_but_usable:
            allowed.discard("audio_too_quiet")
        return allowed
    if name == "image":
        return _IMAGE_DECISION_WARNINGS
    return set()


def _public_warning_suppressions(name: str, *, speech_required: bool, quiet_but_usable: bool) -> set[str]:
    suppressed: set[str] = set()
    if name == "audio":
        if not speech_required:
            suppressed.add("speech_not_detected")
        if quiet_but_usable:
            suppressed.add("audio_too_quiet")
    return suppressed


def _retake_warning_set(name: str, *, speech_required: bool) -> set[str]:
    if name == "video":
        return _VIDEO_RETAKE_WARNINGS
    if name == "audio":
        allowed = set(_AUDIO_RETAKE_WARNINGS)
        if not speech_required:
            allowed.discard("speech_not_detected")
        return allowed
    return set()


def _modality_summary(
    name: str,
    signal: dict[str, Any] | None,
    *,
    threshold: float,
    quality_key: str,
    warning_key: str,
    required_duration_keys: tuple[str, ...] = (),
    speech_required: bool = False,
) -> dict[str, Any]:
    details = _signal_details(signal)
    status = _normalized_status(details)
    score = _capture_quality_score(details.get(quality_key))
    warnings = _details_warnings(details, warning_key)

    public_warnings = list(warnings)
    quiet_but_usable = details.get("quiet_but_usable") is True
    public_warning_suppressions = _public_warning_suppressions(
        name,
        speech_required=speech_required,
        quiet_but_usable=quiet_but_usable,
    )
    if public_warning_suppressions:
        public_warnings = [warning for warning in public_warnings if warning not in public_warning_suppressions]
    decision_warning_set = _decision_warning_set(
        name,
        speech_required=speech_required,
        quiet_but_usable=quiet_but_usable,
    )
    decision_warnings = [warning for warning in public_warnings if warning in decision_warning_set and warning not in _EVIDENCE_ONLY_WARNINGS]
    evidence_blocking_warnings = [
        warning for warning in decision_warnings
        if warning in _EVIDENCE_BLOCKING_WARNINGS.get(name, set())
    ]
    public_warnings = clean_warning_codes(public_warnings)

    duration = None
    for key in required_duration_keys:
        duration = _non_negative_number(details.get(key))
        if duration is not None:
            break

    readable = status in _READABLE_STATUSES and score is not None
    if required_duration_keys and duration is None:
        readable = False

    if not readable:
        missing_warning = f"{name}_missing"
        # Failed/unreadable analyzer payloads must not leak stale evidence or
        # arbitrary warning strings. Preserve only timeout diagnostics that
        # downstream required-modality handling uses to distinguish a timeout
        # from an ordinary absent/unreadable recording.
        diagnostic_warnings = [
            warning
            for warning in public_warnings
            if warning in _UNREADABLE_DIAGNOSTIC_WARNINGS
        ]
        public_warnings = clean_warning_codes(
            [*diagnostic_warnings, missing_warning]
        )
        decision_warnings = []
        score = None
        if not required_duration_keys:
            duration = None

    usable = bool(readable and score is not None and score >= threshold and not evidence_blocking_warnings)
    weak = bool(readable and not usable)
    strong = bool(usable and score is not None and score >= _STRONG_THRESHOLD)

    return {
        "name": name,
        "present": readable,
        "usable": usable,
        "weak": weak,
        "strong": strong,
        "score": score,
        "warnings": public_warnings,
        "decision_warnings": decision_warnings,
        "evidence_blocking_warnings": evidence_blocking_warnings,
        "details": details,
    }


def _task_quality(task: Any) -> str:
    attempts_value, reaction_time_value, errors_value = _task_fields(task)
    values_present = any(value is not None for value in (attempts_value, reaction_time_value, errors_value))
    if not values_present:
        return "missing"

    attempts = _non_negative_int(attempts_value)
    reaction_time = _positive_number(reaction_time_value)
    errors = _non_negative_int(errors_value)
    if attempts is None or reaction_time is None or errors is None:
        return "failed"
    if attempts <= 0:
        return "failed"
    if attempts >= 3:
        return "good"
    return "weak"


def _confidence_multiplier_from_formula(formula: float | None) -> float:
    if formula is None or not math.isfinite(formula):
        return 0.15
    return max(0.15, min(1.0, formula))


def _combined_quality_warnings(*modality_summaries: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for summary in modality_summaries:
        for warning in summary["warnings"]:
            if warning not in ordered:
                ordered.append(warning)
    return ordered


def _decision_warnings(*modality_summaries: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for summary in modality_summaries:
        for warning in summary["decision_warnings"]:
            if warning not in ordered:
                ordered.append(warning)
    return ordered


def assess_quality(signals: dict | None, task: Any = None, *, speech_required: bool = False) -> dict:
    signals = signals if isinstance(signals, dict) else {}

    video = _modality_summary(
        "video",
        _signal_payload(signals, "video"),
        threshold=_QUALITY_THRESHOLDS["video"],
        quality_key="visual_quality_score",
        warning_key="visual_warnings",
        required_duration_keys=("duration_seconds",),
    )
    audio = _modality_summary(
        "audio",
        _signal_payload(signals, "voice"),
        threshold=_QUALITY_THRESHOLDS["audio"],
        quality_key="audio_quality_score",
        warning_key="audio_warnings",
        required_duration_keys=("duration_seconds", "duration_sec"),
        speech_required=speech_required,
    )
    image = _modality_summary(
        "image",
        _signal_payload(signals, "camera"),
        threshold=_QUALITY_THRESHOLDS["image"],
        quality_key="image_quality_score",
        warning_key="image_warnings",
    )

    quality_warnings = _combined_quality_warnings(video, audio, image)
    decision_warnings = _decision_warnings(video, audio, image)

    readable_modalities = [signal for signal in (video, audio, image) if signal["present"]]
    usable_modalities = [signal for signal in readable_modalities if signal["usable"]]
    weak_modalities = [signal["name"] for signal in readable_modalities if signal["weak"]]
    missing_modalities = [signal["name"] for signal in (video, audio, image) if not signal["present"]]

    present_modalities = len(readable_modalities)
    usable_count = len(usable_modalities)
    strong_count = sum(1 for signal in usable_modalities if signal["strong"])

    quality_scores = [signal["score"] for signal in readable_modalities if signal["score"] is not None]
    aggregate_quality = safe_number(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0
    aggregate_quality = clamp01(aggregate_quality, 0.0) or 0.0

    warning_factor = 1.0 - min(len(decision_warnings) / 8.0, 1.0)
    formula = (
        (0.5 * aggregate_quality)
        + (0.2 * (usable_count / 3.0))
        + (0.15 * (strong_count / 3.0))
        + (0.15 * warning_factor)
    )
    confidence_multiplier = _confidence_multiplier_from_formula(formula)
    confidence_multiplier = safe_number(confidence_multiplier) or 0.15

    task_quality = _task_quality(task)

    failure_reason = None
    retake_required = False
    status = "passed"

    if present_modalities == 0:
        failure_reason = FAILURE_REASON_MISSING_MEDIA
        status = "weak"
        retake_required = True
    elif usable_count == 0:
        failure_reason = FAILURE_REASON_LOW_QUALITY_MEDIA
        status = "weak"
        retake_required = True
    elif present_modalities == 1 and usable_count == 1 and strong_count == 0:
        failure_reason = FAILURE_REASON_LOW_QUALITY_MEDIA
        status = "weak"
        retake_required = True
    elif any(set(signal["decision_warnings"]) & _retake_warning_set(signal["name"], speech_required=speech_required) for signal in readable_modalities):
        failure_reason = FAILURE_REASON_LOW_QUALITY_MEDIA
        status = "weak"
        retake_required = True
    elif weak_modalities:
        status = "weak"

    if retake_required:
        suggested_action = "rescan_recommended"
    elif status == "weak":
        suggested_action = "review_required"
    else:
        suggested_action = "continue_normal_activity"

    result_warnings = quality_warnings
    weak_reasons = decision_warnings if status == "weak" else []

    return {
        "status": status,
        "passed": True,
        "weak": status == "weak",
        "failure_reason": failure_reason,
        "reasons": [failure_reason] if failure_reason else [],
        "weak_reasons": weak_reasons,
        "retake_required": retake_required,
        "suggested_action": suggested_action,
        "usable_modalities": usable_count,
        "weak_modalities": weak_modalities,
        "missing_modalities": missing_modalities,
        "present_modalities": present_modalities,
        "confidence_multiplier": confidence_multiplier,
        "task_quality": task_quality,
        "warnings": result_warnings,
        "media_quality": {
            "aggregate_quality": aggregate_quality,
            "video": {
                "present": video["present"],
                "usable": video["usable"],
                "weak": video["weak"],
                "score": safe_number(video["score"]),
                "warnings": video["warnings"],
                "decision_warnings": video["decision_warnings"],
                "evidence_blocking_warnings": video["evidence_blocking_warnings"],
            },
            "audio": {
                "present": audio["present"],
                "usable": audio["usable"],
                "weak": audio["weak"],
                "score": safe_number(audio["score"]),
                "warnings": audio["warnings"],
                "decision_warnings": audio["decision_warnings"],
                "evidence_blocking_warnings": audio["evidence_blocking_warnings"],
            },
            "image": {
                "present": image["present"],
                "usable": image["usable"],
                "weak": image["weak"],
                "score": safe_number(image["score"]),
                "warnings": image["warnings"],
                "decision_warnings": image["decision_warnings"],
                "evidence_blocking_warnings": image["evidence_blocking_warnings"],
            },
        },
    }
