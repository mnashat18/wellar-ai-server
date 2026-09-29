"""The durable worker's analysis adapter.

This module calls the same runtime, modality analyzers, validation, feature
fusion, and scoring modules used by ``main.py``.  It contains no Directus or
database access; the coordinator supplies authorized context and local media
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

from analysis_runtime import AnalysisRuntimeUnavailable, get_runtime
from baseline import baseline_ready_for_personalized_scoring, baseline_status_payload, evaluate_baseline_eligibility
from config import MODEL_VERSION
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from quality import assess_quality
from scoring import compute_result
from validation import ValidationPolicy, validate_scan_inputs


@dataclass(frozen=True)
class AnalysisInput:
    scan_id: str
    video_path: str | None
    audio_path: str | None
    image_path: str | None
    task_metrics: Mapping[str, Any] | None = None
    baseline: Mapping[str, Any] | None = None
    expected_phrase: str | None = None


@dataclass(frozen=True)
class AnalysisOutcome:
    ok: bool
    result: Mapping[str, Any] | None = None
    failure_code: str | None = None
    error_type: str | None = None
    model_version: str = MODEL_VERSION


def _task(value: Mapping[str, Any] | None) -> Any:
    if not isinstance(value, Mapping):
        return None
    return SimpleNamespace(
        reaction_time=value.get("reaction_time"),
        errors=value.get("errors"),
        attempts=value.get("attempts"),
    )


def _positive_timeout() -> float:
    raw = os.getenv("MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS", "8.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 8.0
    return value if math.isfinite(value) and value > 0 else 8.0


def _personal_deviation(result: Mapping[str, Any]) -> float | None:
    drifts: list[float] = []
    for field_name in ("face_metrics", "voice_metrics", "reaction_metrics"):
        metrics = result.get(field_name)
        if not isinstance(metrics, Mapping):
            continue
        for drift_payload in (metrics.get("baseline_drifts") or {}).values():
            if not isinstance(drift_payload, Mapping):
                continue
            drift = drift_payload.get("drift")
            if isinstance(drift, bool) or type(drift) not in {int, float}:
                continue
            if math.isfinite(float(drift)):
                drifts.append(abs(float(drift)))
    return round(sum(drifts) / len(drifts), 4) if drifts else None


def _optional_transcript(path: str | None) -> tuple[str | None, str]:
    """Use the same bounded, isolated Whisper invocation as the legacy path."""
    if not path:
        return None, "audio_missing"
    try:
        timeout = float(os.getenv("AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS", "1.5"))
    except (TypeError, ValueError):
        timeout = 1.5
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = 1.5
    command = [
        sys.executable,
        "-c",
        "import sys; from audio import transcribe_audio; print(transcribe_audio(sys.argv[1]))",
        path,
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=timeout,
        )
        transcript = (completed.stdout or "").strip()
        return (transcript, "completed") if transcript else (None, "error")
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception:
        return None, "error"


def _required_failure(
    policy: ValidationPolicy,
    quality: Mapping[str, Any],
    worker_states: Mapping[str, Any],
) -> str | None:
    media_quality = quality.get("media_quality") if isinstance(quality, Mapping) else None
    media_quality = media_quality if isinstance(media_quality, Mapping) else {}
    for name, required, missing_reason in (
        ("video", policy.require_video, "video_missing"),
        ("audio", policy.require_audio, "audio_missing"),
        ("image", policy.require_image, "image_missing"),
    ):
        if not required:
            continue
        summary = media_quality.get(name)
        if not isinstance(summary, Mapping) or not summary.get("present"):
            return missing_reason
        if not summary.get("usable"):
            return "low_quality_media"
        state = worker_states.get(name)
        if not isinstance(state, Mapping) or state.get("result_received") is not True:
            return "analysis_exception"
    return None


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and type(value) in {int, float}
        and math.isfinite(float(value))
    )


def _raw_analysis_has_valid_evidence(
    modality: str,
    result: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(result, Mapping) or not isinstance(state, Mapping):
        return False
    if (
        state.get("result_received") is not True
        or state.get("timed_out") is True
        or state.get("analyzer_error") is True
        or state.get("final_alive") is True
    ):
        return False
    details = result.get("details")
    if not isinstance(details, Mapping) or details.get("status") != "ok":
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
    if not _finite_number(details.get(confidence_key)) or not _finite_number(details.get(quality_key)):
        return False
    if modality in {"video", "audio"}:
        duration = details.get("duration_seconds")
        if duration is None and modality == "audio":
            duration = details.get("duration_sec")
        if not _finite_number(duration) or float(duration) < 0.0:
            return False
    return True


def _face_eye_evidence_unreliable(warnings: list[str], video_details: Mapping[str, Any], image_details: Mapping[str, Any]) -> bool:
    if set(warnings) & {"face_not_visible", "subject_not_visible", "landmark_detection_failed", "insufficient_usable_frames"}:
        return True
    face_frames = video_details.get("face_frames")
    valid_face_frames = face_frames if type(face_frames) is int and face_frames >= 0 else 0
    if video_details.get("reliable_eye_landmarks") is False and valid_face_frames <= 0:
        return True
    return bool(image_details and image_details.get("face_detected") is False and image_details.get("avg_ear") is None)


def _full_evidence_failure(
    *,
    policy: ValidationPolicy,
    result: Mapping[str, Any],
    worker_states: Mapping[str, Any],
    valid_modalities: list[str],
) -> str | None:
    if not (policy.require_video and policy.require_audio and policy.require_image):
        return None
    required = {"video", "audio", "image"}
    valid = set(valid_modalities)
    if not required.issubset(valid):
        if "audio" not in valid:
            audio_state = worker_states.get("audio") or {}
            if audio_state.get("timed_out"):
                return "audio_validation_timeout"
            return "audio_missing"
        if "video" not in valid:
            return "video_missing"
        if "image" not in valid:
            return "image_missing"
        return "missing_media"
    if not _finite_number(result.get("voice_confidence")):
        return "audio_missing"
    return None


def _result_has_valid_evidence(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    profiles = ((result.get("fusion_details") or {}).get("signal_profiles") or {})
    if isinstance(profiles, Mapping):
        for profile in profiles.values():
            if isinstance(profile, Mapping) and profile.get("present") is True and _finite_number(profile.get("score")):
                return True
    modality_scores = result.get("modality_scores")
    if isinstance(modality_scores, Mapping) and any(_finite_number(value) for value in modality_scores.values()):
        return True
    for field_name, score_names in {
        "face_metrics": ("face_score", "image_score", "video_score"),
        "voice_metrics": ("voice_score",),
        "reaction_metrics": ("reaction_score",),
    }.items():
        metrics = result.get(field_name)
        if isinstance(metrics, Mapping) and any(_finite_number(metrics.get(name)) for name in score_names):
            return True
    return False


def analyze_scan(
    request: AnalysisInput,
    *,
    runtime: Any | None = None,
    ml_runtime: MLRuntime | None = None,
    policy: ValidationPolicy | None = None,
) -> AnalysisOutcome:
    """Run the canonical multimodal pipeline without any persistence side effect."""
    try:
        active_policy = policy or ValidationPolicy.from_env()
        active_runtime = runtime or get_runtime()
        if not active_runtime.is_ready():
            raise AnalysisRuntimeUnavailable("analyzer_runtime_not_ready")
        media = SimpleNamespace(
            video=request.video_path,
            audio=request.audio_path,
            image=request.image_path,
        )
        analysis_results, worker_states = active_runtime.run_scan(
            request.scan_id,
            media,
            deadline_seconds=_positive_timeout(),
        )
        video_result = analysis_results.get("video")
        audio_result = analysis_results.get("audio")
        image_result = analysis_results.get("image")
        raw_signals = {
            "camera": image_result or {"score": None, "details": {"status": "missing"}},
            "video": video_result,
            "voice": audio_result,
        }
        task = _task(request.task_metrics)
        quality = assess_quality(
            raw_signals,
            task,
            speech_required=active_policy.require_phrase_match or bool(request.expected_phrase),
        )
        transcript = None
        phrase_status = "not_required"
        if request.expected_phrase and active_policy.require_phrase_match:
            transcript, phrase_status = _optional_transcript(request.audio_path)
        phrase_expected = request.expected_phrase if (active_policy.require_phrase_match or transcript) else None
        validation = validate_scan_inputs(
            policy=active_policy,
            media=media,
            video_result=video_result,
            audio_result=audio_result,
            image_result=image_result,
            expected_phrase=phrase_expected,
            transcript=transcript,
        )
        combined_warnings = list(quality.get("warnings") or [])
        combined_warnings.extend(validation.get("warnings") or [])
        combined_warnings.extend(validation.get("critical_errors") or [])
        quality["warnings"] = list(dict.fromkeys(item for item in combined_warnings if item))
        if validation.get("warnings") or validation.get("critical_errors"):
            quality["weak"] = True
            quality["status"] = "weak"
        if validation.get("failure_reason") in {"missing_media", "unreadable_media"}:
            quality["failure_reason"] = "missing_media"
            quality["retake_required"] = True
            quality["suggested_action"] = "rescan_recommended"

        critical = set(validation.get("critical_errors") or [])
        if critical and not critical.issubset({"missing_media", "unreadable_media"}):
            return AnalysisOutcome(
                ok=False,
                failure_code=validation.get("failure_reason") or "analysis_validation_failed",
                error_type="ValidationError",
            )

        timed_out_modalities = [
            modality
            for modality, state in worker_states.items()
            if isinstance(state, Mapping) and state.get("timed_out") is True
        ]
        analyzer_error_modalities = [
            modality
            for modality, state in worker_states.items()
            if isinstance(state, Mapping) and state.get("analyzer_error") is True
        ]
        running_modalities = [
            modality
            for modality, state in worker_states.items()
            if isinstance(state, Mapping) and state.get("final_alive") is True
        ]
        raw_valid_modalities = [
            modality
            for modality, result_value in {
                "video": video_result,
                "audio": audio_result,
                "image": image_result,
            }.items()
            if _raw_analysis_has_valid_evidence(modality, result_value, worker_states.get(modality) or {})
        ]
        terminal_failure: str | None = None
        if worker_states and len(timed_out_modalities) == len(worker_states):
            terminal_failure = "validation_timeout"
        elif worker_states and len(analyzer_error_modalities) == len(worker_states) and not raw_valid_modalities:
            terminal_failure = "analysis_exception"
        required_failure = _required_failure(active_policy, quality, worker_states)
        if terminal_failure is None and required_failure is not None:
            terminal_failure = required_failure
        if terminal_failure is None and not raw_valid_modalities:
            terminal_failure = quality.get("failure_reason") or "analysis_no_result"
        if terminal_failure is None and running_modalities:
            terminal_failure = "analysis_exception"
        if terminal_failure is not None:
            return AnalysisOutcome(ok=False, failure_code=terminal_failure, error_type="ValidationError")

        if _face_eye_evidence_unreliable(
            quality.get("warnings") or [],
            (video_result.get("details") or {}) if isinstance(video_result, Mapping) else {},
            (image_result.get("details") or {}) if isinstance(image_result, Mapping) else {},
        ):
            quality["failure_reason"] = quality.get("failure_reason") or "low_quality_media"
            quality["retake_required"] = True
            quality["suggested_action"] = "rescan_recommended"
            quality["status"] = "weak"
            quality["weak"] = True

        features, _ = features_from_signals(raw_signals, task=task)
        vector = vector_from_features(features)
        model = ml_runtime or MLRuntime()
        if model.local_model_required() and not model.is_loaded():
            model.load()
        ml_result = model.predict(vector)
        baseline = dict(request.baseline) if isinstance(request.baseline, Mapping) else None
        preview = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality,
            ml_result=ml_result,
        )
        baseline_used = baseline_ready_for_personalized_scoring(
            baseline,
            quality_result=quality,
            validation_result=validation,
            result=preview,
            task=task,
            expected_phrase=request.expected_phrase,
            unique_row=baseline is not None,
        )
        result = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=baseline_used,
            quality=quality,
            ml_result=ml_result,
        )
        result.update(
            {
                "spoken_transcript": transcript,
                "expected_phrase": request.expected_phrase,
                "phrase_match_score": (validation.get("quality_scores") or {}).get("phrase_match"),
                "audio_quality_score": (validation.get("quality_scores") or {}).get("audio"),
                "video_quality_score": (validation.get("quality_scores") or {}).get("video"),
                "image_quality_score": (validation.get("quality_scores") or {}).get("image"),
                "validation_warnings": quality.get("warnings"),
            }
        )
        baseline_eligibility = evaluate_baseline_eligibility(
            quality_result=quality,
            validation_result=validation,
            result=result,
            signals=raw_signals,
            expected_phrase=request.expected_phrase,
            task=task,
            manually_unreliable=False,
        )
        result.update(
            {
                "capture_quality_score": baseline_eligibility.get("capture_quality_score"),
                "measurement_reliability_score": baseline_eligibility.get("measurement_reliability_score"),
                "personal_deviation_score": _personal_deviation(result),
                "task_completion_status": baseline_eligibility.get("task_completion_status"),
                "baseline_status_at_inference": baseline_status_payload(baseline).get("baseline_status"),
                "baseline_confidence": baseline_status_payload(baseline).get("baseline_confidence"),
                "baseline_eligible": baseline_eligibility.get("eligible"),
                "phrase_status": phrase_status,
            }
        )
        full_evidence_failure = _full_evidence_failure(
            policy=active_policy,
            result=result,
            worker_states=worker_states,
            valid_modalities=raw_valid_modalities,
        )
        if full_evidence_failure is not None:
            return AnalysisOutcome(ok=False, failure_code=full_evidence_failure, error_type="ValidationError")
        if not _result_has_valid_evidence(result):
            return AnalysisOutcome(ok=False, failure_code="analysis_no_result", error_type="ValidationError")
        if (
            type(result.get("readiness_score")) is not int
            or not 0 <= result.get("readiness_score") <= 100
            or not _finite_number(result.get("confidence"))
            or not 0 <= float(result.get("confidence")) <= 1
        ):
            return AnalysisOutcome(ok=False, failure_code="analysis_no_result", error_type="ValidationError")
        return AnalysisOutcome(ok=True, result=result)
    except AnalysisRuntimeUnavailable as exc:
        return AnalysisOutcome(ok=False, failure_code="analyzer_unavailable", error_type=type(exc).__name__)
    except Exception as exc:
        return AnalysisOutcome(ok=False, failure_code="analysis_exception", error_type=type(exc).__name__)
