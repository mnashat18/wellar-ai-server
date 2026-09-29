"""Dedicated unit tests for scoring.py.

These tests use the standard-library ``unittest`` framework ONLY. They construct
plain dictionaries as analyzer signals and never touch pytest, FastAPI, Directus,
the network, media files, or any live model inference. Every case exercises the
public ``compute_result`` contract or a pure scoring helper directly.
"""

from __future__ import annotations

import copy
import json
import math
import unittest

import scoring
from scoring import (
    BASE_WEIGHTS,
    VALID_ACTIONS,
    VALID_RISK_LEVELS,
    _audio_usable_speech,
    _baseline_drift,
    _bool_flag,
    _confirmed_sustained_eye_closure,
    _finite_number,
    _signal_score,
    _status_ok,
    _unit_interval,
    clamp_confidence,
    compute_result,
    compute_task_score,
)
from config import MODEL_VERSION


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _signal(score, details):
    return {"score": score, "details": details}


def _ok_camera(score=0.76, **extra):
    details = {
        "status": "ok",
        "image_confidence": score,
        "image_quality_score": round(score - 0.03, 4),
        "image_warnings": [],
    }
    details.update(extra)
    return _signal(score, details)


def _ok_video(score=0.82, **extra):
    details = {
        "status": "ok",
        "visual_confidence": score,
        "visual_quality_score": round(score - 0.02, 4),
        "visual_warnings": [],
    }
    details.update(extra)
    return _signal(score, details)


def _ok_voice(score=0.78, **extra):
    details = {
        "status": "ok",
        "audio_confidence": score,
        "audio_quality_score": round(score - 0.02, 4),
        "audio_warnings": [],
    }
    details.update(extra)
    return _signal(score, details)


def _clean_signals():
    return {
        "camera": _ok_camera(),
        "video": _ok_video(),
        "voice": _ok_voice(),
    }


def _missing_signals():
    return {
        "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
        "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
        "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
    }


def _active_baseline(open_eye=0.30, asym=0.01, energy=0.03):
    return {
        "scan_count": 4,
        "is_active": True,
        "face_avg": {
            "schema_version": 2,
            "feature_stats": {
                "open_eye_aperture": {"median": open_eye, "mad": 0.02, "count": 4},
                "left_right_eye_asymmetry": {"median": asym, "mad": 0.02, "count": 4},
            },
        },
        "voice_avg": {
            "schema_version": 2,
            "feature_stats": {
                "normalized_voice_energy": {"median": energy, "mad": 0.02, "count": 4},
            },
        },
        "reaction_avg": {"schema_version": 2, "feature_stats": {}},
    }


def _assert_json_finite(testcase, obj, path="root"):
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        testcase.assertTrue(math.isfinite(obj), f"non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            testcase.assertIsInstance(key, str, f"non-str key at {path}")
            _assert_json_finite(testcase, value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _assert_json_finite(testcase, value, f"{path}[{index}]")
    elif obj is not None and not isinstance(obj, (str, int)):
        testcase.fail(f"unexpected type {type(obj)!r} at {path}")


# ---------------------------------------------------------------------------
# 1. strict public-input safety
# ---------------------------------------------------------------------------
class StrictInputSafetyTests(unittest.TestCase):
    def test_finite_number_rejects_none(self):
        self.assertIsNone(_finite_number(None))

    def test_finite_number_rejects_bool_true(self):
        self.assertIsNone(_finite_number(True))

    def test_finite_number_rejects_bool_false(self):
        self.assertIsNone(_finite_number(False))

    def test_finite_number_rejects_nan(self):
        self.assertIsNone(_finite_number(float("nan")))

    def test_finite_number_rejects_inf(self):
        self.assertIsNone(_finite_number(float("inf")))
        self.assertIsNone(_finite_number(float("-inf")))

    def test_finite_number_rejects_numeric_string(self):
        self.assertIsNone(_finite_number("0.5"))

    def test_finite_number_accepts_int_and_float(self):
        self.assertEqual(_finite_number(3), 3.0)
        self.assertEqual(_finite_number(0.25), 0.25)

    def test_unit_interval_accepts_exact_zero(self):
        self.assertEqual(_unit_interval(0.0), 0.0)

    def test_unit_interval_accepts_exact_one(self):
        self.assertEqual(_unit_interval(1.0), 1.0)

    def test_unit_interval_rejects_above_one_not_clamped(self):
        self.assertIsNone(_unit_interval(1.5))

    def test_unit_interval_rejects_below_zero_not_clamped(self):
        self.assertIsNone(_unit_interval(-0.01))

    def test_bool_flag_only_true_is_true(self):
        self.assertTrue(_bool_flag(True))
        self.assertFalse(_bool_flag(1))
        self.assertFalse(_bool_flag("true"))
        self.assertFalse(_bool_flag(None))

    def test_compute_result_does_not_mutate_signals(self):
        signals = _clean_signals()
        before = copy.deepcopy(signals)
        compute_result(signals=signals)
        self.assertEqual(signals, before)

    def test_compute_result_does_not_mutate_quality(self):
        signals = _clean_signals()
        quality = {"status": "passed", "warnings": ["audio_too_noisy"], "weak": False}
        before = copy.deepcopy(quality)
        compute_result(signals=signals, quality=quality)
        self.assertEqual(quality, before)

    def test_malformed_signals_do_not_crash(self):
        bad = {
            "camera": {"score": float("nan"), "details": {"status": "ok", "image_confidence": float("inf"), "image_quality_score": "x", "image_warnings": None}},
            "video": {"score": "x", "details": {"status": "ok", "visual_confidence": 2.0, "visual_warnings": ["weird_code"]}},
            "voice": "not-a-dict",
        }
        result = compute_result(signals=bad, quality={"warnings": None})
        _assert_json_finite(self, result)

    def test_non_dict_signals_are_tolerated(self):
        result = compute_result(signals="nonsense")  # type: ignore[arg-type]
        self.assertIn(result["risk_level"], VALID_RISK_LEVELS)

    def test_keyboard_interrupt_is_not_swallowed(self):
        class Boom:
            @property
            def reaction_time(self):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            compute_result(signals={}, task=Boom())

    def test_system_exit_is_not_swallowed(self):
        class Boom:
            @property
            def attempts(self):
                raise SystemExit()

        with self.assertRaises(SystemExit):
            compute_result(signals={}, task=Boom())


# ---------------------------------------------------------------------------
# 2. analyzer-status precedence
# ---------------------------------------------------------------------------
class AnalyzerStatusPrecedenceTests(unittest.TestCase):
    def test_status_ok_is_case_insensitive(self):
        self.assertTrue(_status_ok({"details": {"status": "OK"}}))
        self.assertTrue(_status_ok({"details": {"status": " ok "}}))

    def test_non_ok_status_is_not_readable(self):
        self.assertFalse(_status_ok({"details": {"status": "error"}}))
        self.assertFalse(_status_ok({"details": {}}))
        self.assertFalse(_status_ok("nope"))

    def test_top_level_score_cannot_rescue_failed_status(self):
        analysis = {"score": 0.9, "details": {"status": "error"}}
        self.assertIsNone(_signal_score(analysis, "visual_confidence"))

    def test_failed_status_modality_is_not_present(self):
        signals = {
            "camera": _signal(0.9, {"status": "error", "image_confidence": 0.9}),
            "video": _ok_video(),
            "voice": _ok_voice(),
        }
        result = compute_result(signals=signals)
        self.assertIsNone(result["modality_scores"]["image"])

    def test_stale_warnings_ignored_under_bad_status(self):
        analysis = {"score": 0.5, "details": {"status": "error", "visual_warnings": ["video_blurry"]}}
        self.assertEqual(scoring._signal_warnings(analysis, "visual_warnings"), [])

    def test_ok_status_exposes_warnings(self):
        analysis = _ok_video(visual_warnings=["video_blurry"])
        self.assertIn("video_blurry", scoring._signal_warnings(analysis, "visual_warnings"))


# ---------------------------------------------------------------------------
# 3. dedicated score/quality fields
# ---------------------------------------------------------------------------
class DedicatedFieldTests(unittest.TestCase):
    def test_dedicated_field_is_authoritative(self):
        analysis = {"score": 0.2, "details": {"status": "ok", "visual_confidence": 0.8}}
        self.assertEqual(_signal_score(analysis, "visual_confidence"), 0.8)

    def test_top_level_score_is_only_fallback_alias(self):
        analysis = {"score": 0.7, "details": {"status": "ok"}}
        self.assertEqual(_signal_score(analysis, "visual_confidence"), 0.7)

    def test_out_of_range_dedicated_field_is_invalid(self):
        analysis = {"score": 0.7, "details": {"status": "ok", "visual_confidence": 1.4}}
        self.assertIsNone(_signal_score(analysis, "visual_confidence"))

    def test_exact_zero_dedicated_field_is_valid(self):
        analysis = {"score": 0.7, "details": {"status": "ok", "visual_confidence": 0.0}}
        self.assertEqual(_signal_score(analysis, "visual_confidence"), 0.0)

    def test_present_dedicated_field_is_not_overridden_by_score(self):
        analysis = {"score": 0.9, "details": {"status": "ok", "visual_confidence": 0.0}}
        self.assertEqual(_signal_score(analysis, "visual_confidence"), 0.0)


# ---------------------------------------------------------------------------
# 4 & 5. capture-quality vs fatigue evidence; duplicate penalties
# ---------------------------------------------------------------------------
class CaptureVsFatigueTests(unittest.TestCase):
    def test_blur_lowers_confidence_but_not_fatigue(self):
        clean = compute_result(signals=_clean_signals())
        blurry_signals = _clean_signals()
        blurry_signals["video"] = _ok_video(visual_warnings=["video_blurry"])
        blurry = compute_result(signals=blurry_signals)
        self.assertLess(blurry["confidence"], clean["confidence"])
        self.assertEqual(blurry["observed_fatigue_score"], clean["observed_fatigue_score"])

    def test_capture_warnings_do_not_raise_observed_fatigue(self):
        signals = _clean_signals()
        signals["voice"] = _ok_voice(audio_warnings=["audio_too_noisy"])
        signals["video"] = _ok_video(visual_warnings=["video_blurry", "video_too_dark"])
        result = compute_result(signals=signals)
        self.assertEqual(result["observed_fatigue_score"], 0)

    def test_unknown_warning_has_no_numeric_penalty(self):
        base = compute_result(signals=_clean_signals())
        signals = _clean_signals()
        signals["video"] = _ok_video(visual_warnings=["totally_unknown_code"])
        unknown = compute_result(signals=signals)
        self.assertEqual(unknown["modality_scores"]["video"], base["modality_scores"]["video"])

    def test_blurry_video_reduces_video_modality_score(self):
        clean = compute_result(signals=_clean_signals())
        signals = _clean_signals()
        signals["video"] = _ok_video(visual_warnings=["video_blurry"])
        blurry = compute_result(signals=signals)
        self.assertLess(blurry["modality_scores"]["video"], clean["modality_scores"]["video"])

    def test_degrading_video_warning_still_reduces_scored_quality(self):
        clean = compute_result(signals=_clean_signals())
        degraded_signals = _clean_signals()
        degraded_signals["video"] = _ok_video(visual_warnings=["video_too_dark"])
        degraded = compute_result(signals=degraded_signals)

        self.assertLess(degraded["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertLess(degraded["confidence"], clean["confidence"])


# ---------------------------------------------------------------------------
# 6. speech_rate retirement
# ---------------------------------------------------------------------------
class SpeechRateRetirementTests(unittest.TestCase):
    def test_voice_metrics_speech_rate_is_none(self):
        signals = _clean_signals()
        signals["voice"] = _ok_voice(speech_rate=1.2, rms_energy=0.03)
        result = compute_result(signals=signals)
        self.assertIsNone(result["voice_metrics"]["speech_rate"])

    def test_voice_baseline_drift_speech_rate_is_none(self):
        signals = _clean_signals()
        signals["voice"] = _ok_voice(speech_rate=1.2, rms_energy=0.03)
        result = compute_result(signals=signals, baseline=_active_baseline(), baseline_used=True)
        self.assertIsNone(result["voice_metrics"]["baseline_drifts"]["speech_rate"])

    def test_raw_observations_never_expose_speech_rate_value(self):
        observations = scoring._raw_baseline_observations(
            {"voice": _ok_voice(speech_rate=9.9, rms_energy=0.03)}
        )
        self.assertIsNone(observations["speech_rate"])


# ---------------------------------------------------------------------------
# 7 & 8. confidence monotonicity and semantics
# ---------------------------------------------------------------------------
class ConfidenceTests(unittest.TestCase):
    def test_noisy_audio_reduces_confidence(self):
        clean = compute_result(signals=_clean_signals())
        noisy_signals = _clean_signals()
        noisy_signals["voice"] = _ok_voice(audio_warnings=["audio_too_noisy"])
        noisy = compute_result(signals=noisy_signals)
        self.assertLess(noisy["confidence"], clean["confidence"])

    def test_more_noise_reduces_confidence_further(self):
        single = _clean_signals()
        single["voice"] = _ok_voice(audio_warnings=["audio_too_noisy"])
        double = _clean_signals()
        double["voice"] = _ok_voice(audio_warnings=["audio_too_noisy", "too_much_silence"])
        self.assertLessEqual(
            compute_result(signals=double)["confidence"],
            compute_result(signals=single)["confidence"],
        )

    def test_invalid_scan_confidence_below_reliable_floor(self):
        quality = {"status": "failed", "retake_required": True, "failure_reason": "low_quality_media", "warnings": []}
        result = compute_result(signals=_clean_signals(), quality=quality)
        self.assertLessEqual(result["confidence"], scoring.INVALID_SCAN_CONFIDENCE_CAP)

    def test_invalid_scan_preserves_relative_ordering(self):
        quality = {"status": "failed", "retake_required": True, "failure_reason": "low_quality_media", "warnings": []}
        cleaner = {
            "camera": _ok_camera(0.5),
            "video": _ok_video(0.5),
            "voice": _ok_voice(0.5),
        }
        noisier = dict(cleaner)
        noisier["voice"] = _ok_voice(0.3, audio_warnings=["audio_too_noisy", "too_much_silence"])
        c = compute_result(signals=cleaner, quality=dict(quality))
        n = compute_result(signals=noisier, quality=dict(quality))
        self.assertLess(n["confidence"], c["confidence"])

    def test_confidence_is_finite_and_unit_interval(self):
        for signals in (_clean_signals(), _missing_signals(), {}):
            result = compute_result(signals=signals)
            self.assertIsInstance(result["confidence"], float)
            self.assertTrue(math.isfinite(result["confidence"]))
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_previous_confidence_accepts_finite_number(self):
        result = compute_result(signals=_clean_signals(), previous_confidence=0.5)
        self.assertIsInstance(result["confidence_drift"], float)
        self.assertTrue(math.isfinite(result["confidence_drift"]))

    def test_previous_confidence_rejects_nan(self):
        result = compute_result(signals=_clean_signals(), previous_confidence=float("nan"))
        self.assertEqual(result["confidence_drift"], 0.0)

    def test_previous_confidence_rejects_string(self):
        result = compute_result(signals=_clean_signals(), previous_confidence="0.9")
        self.assertEqual(result["confidence_drift"], 0.0)

    def test_clamp_confidence_behaviour(self):
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(-0.2), 0.0)
        self.assertIsNone(clamp_confidence(None))


# ---------------------------------------------------------------------------
# 9, 10, 11. sustained eye closure
# ---------------------------------------------------------------------------
class EyeClosureTests(unittest.TestCase):
    def _closure_video(self, **extra):
        details = {
            "status": "ok",
            "visual_confidence": 0.8,
            "visual_quality_score": 0.8,
            "visual_warnings": [],
            "reliable_eye_landmarks": True,
            "sustained_eye_closure": True,
            "motion_stability_score": 0.8,
            "eye_closure_sample_count": 8,
            "closed_eye_ratio": 0.75,
            "longest_eye_closure_streak": 4,
            "eye_closure_window_ms": 600,
            "eye_closure_window_seconds": 0.6,
            "avg_eye_aperture": 0.16,
            "eye_aperture_std": 0.02,
        }
        details.update(extra)
        return _signal(0.8, details)

    def test_warning_string_alone_is_insufficient(self):
        analysis = {"video": _ok_video(visual_warnings=["sustained_eye_closure"])}
        self.assertFalse(_confirmed_sustained_eye_closure(analysis))

    def test_booleans_are_source_of_truth(self):
        self.assertTrue(_confirmed_sustained_eye_closure({"video": self._closure_video()}))

    def test_requires_reliable_landmarks(self):
        video = self._closure_video(reliable_eye_landmarks=False)
        self.assertFalse(_confirmed_sustained_eye_closure({"video": video}))

    def test_requires_ok_status(self):
        video = self._closure_video(status="error")
        self.assertFalse(_confirmed_sustained_eye_closure({"video": video}))

    def test_non_finite_temporal_field_vetoes_closure(self):
        video = self._closure_video(longest_eye_closure_streak="5")
        self.assertFalse(_confirmed_sustained_eye_closure({"video": video}))

    def test_closure_cannot_return_stable(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": self._closure_video(visual_warnings=["sustained_eye_closure"]),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=signals)
        self.assertNotEqual(result["risk_level"], "stable")

    def test_closure_reaches_elevated_but_not_high_alone(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": self._closure_video(),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=signals)
        self.assertEqual(result["risk_level"], "elevated_fatigue")

    def test_closure_does_not_clear_capture_retake(self):
        quality = {"status": "failed", "retake_required": True, "failure_reason": "low_quality_media", "warnings": []}
        result = compute_result(signals={"video": self._closure_video()}, quality=quality)
        self.assertTrue(result["retake_required"])

    def test_valid_capture_closure_is_not_a_retake(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": self._closure_video(),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=signals)
        self.assertFalse(result["retake_required"])

    def test_single_blink_does_not_reduce_valid_scan(self):
        clean = {
            "camera": _ok_camera(0.8),
            "video": _ok_video(0.84, reliable_eye_landmarks=True, sustained_eye_closure=False),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=clean)
        self.assertEqual(result["risk_level"], "stable")
        self.assertGreaterEqual(result["confidence"], 0.45)


# ---------------------------------------------------------------------------
# 12. voice evidence honesty
# ---------------------------------------------------------------------------
class VoiceEvidenceTests(unittest.TestCase):
    def test_noisy_audio_is_not_usable_speech(self):
        self.assertFalse(_audio_usable_speech({"usable_speech_detected": True}, ["audio_too_noisy"]))

    def test_silence_is_not_usable_speech(self):
        self.assertFalse(_audio_usable_speech({"usable_speech_detected": True}, ["too_much_silence"]))

    def test_usable_speech_flag_enables_evidence(self):
        self.assertTrue(
            _audio_usable_speech(
                {
                    "usable_speech_detected": True,
                    "speech_state": "usable_speech",
                    "quiet_but_usable": False,
                },
                [],
            )
        )

    def test_quiet_but_usable_enables_evidence(self):
        self.assertTrue(
            _audio_usable_speech(
                {
                    "usable_speech_detected": True,
                    "speech_state": "quiet_usable_speech",
                    "quiet_but_usable": True,
                },
                [],
            )
        )

    def test_noise_produces_no_audio_fatigue_signal(self):
        signals = {
            "voice": _ok_voice(
                0.72,
                audio_warnings=["audio_too_noisy"],
                speech_presence_score=0.2,
                rms_energy=0.005,
                silence_ratio=0.7,
            ),
        }
        signal = scoring._audio_fatigue_signal(signals=signals, baseline_flags=[], voice_energy_drift=None)
        self.assertEqual(signal, 0.0)


# ---------------------------------------------------------------------------
# 13 & 14. baseline personalization and drift
# ---------------------------------------------------------------------------
class BaselineTests(unittest.TestCase):
    def test_baseline_used_must_be_true_bool(self):
        result = compute_result(signals=_clean_signals(), baseline=_active_baseline(), baseline_used=1)
        self.assertFalse(result["baseline_used"])

    def test_inactive_baseline_does_not_change_confidence(self):
        signals = {
            "camera": _ok_camera(0.76, avg_ear=0.18, left_right_eye_asymmetry=0.01),
            "video": _ok_video(0.82),
            "voice": _ok_voice(0.78, rms_energy=0.014),
        }
        without = compute_result(signals=signals, baseline=None, baseline_used=False)
        ignored = compute_result(signals=signals, baseline=_active_baseline(), baseline_used=False)
        self.assertEqual(ignored["confidence"], without["confidence"])
        self.assertFalse(ignored["baseline_used"])

    def test_baseline_does_not_mutate_payload(self):
        baseline = _active_baseline()
        before = copy.deepcopy(baseline)
        compute_result(signals=_clean_signals(), baseline=baseline, baseline_used=True)
        self.assertEqual(baseline, before)

    def test_baseline_cannot_rescue_confirmed_closure(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": _ok_video(
                0.8,
                reliable_eye_landmarks=True,
                sustained_eye_closure=True,
                motion_stability_score=0.8,
                eye_closure_sample_count=8,
                closed_eye_ratio=0.75,
                longest_eye_closure_streak=4,
                eye_closure_window_ms=600,
                eye_closure_window_seconds=0.6,
                avg_eye_aperture=0.16,
                eye_aperture_std=0.02,
            ),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=signals, baseline=_active_baseline(open_eye=0.22), baseline_used=True)
        self.assertNotEqual(result["risk_level"], "stable")

    def test_baseline_drift_none_without_stat(self):
        self.assertIsNone(_baseline_drift(0.2, None))

    def test_baseline_drift_none_without_current(self):
        self.assertIsNone(_baseline_drift(None, {"median": 0.3, "mad": 0.02}))

    def test_baseline_drift_zero_mad_is_floored(self):
        drift = _baseline_drift(0.3, {"median": 0.3, "mad": 0.0})
        self.assertIsNotNone(drift)
        self.assertGreaterEqual(drift["baseline_mad"], scoring.BASELINE_MAD_FLOOR)

    def test_baseline_drift_flags_below_threshold(self):
        drift = _baseline_drift(0.10, {"median": 0.30, "mad": 0.02})
        self.assertTrue(drift["below_threshold"])
        self.assertLess(drift["z_score"], 0.0)

    def test_baseline_drift_above_baseline_not_flagged(self):
        drift = _baseline_drift(0.35, {"median": 0.30, "mad": 0.02})
        self.assertFalse(drift["below_threshold"])

    def test_active_baseline_can_raise_confidence(self):
        signals = {
            "camera": _ok_camera(0.78, avg_ear=0.18, left_right_eye_asymmetry=0.01),
            "video": _ok_video(0.8),
            "voice": _ok_voice(0.77, rms_energy=0.014),
        }
        baseline_free = compute_result(signals=signals, baseline=None, baseline_used=False)
        personalized = compute_result(
            signals=signals, baseline=_active_baseline(open_eye=0.18, energy=0.014), baseline_used=True
        )
        self.assertGreater(personalized["confidence"], baseline_free["confidence"])
        self.assertTrue(personalized["baseline_used"])

    def test_personal_deviation_alone_cannot_be_high_risk(self):
        signals = {
            "camera": _ok_camera(0.82, avg_ear=0.17, left_right_eye_asymmetry=0.01),
            "video": _ok_video(0.84),
            "voice": _ok_voice(0.81, rms_energy=0.03),
        }
        result = compute_result(signals=signals, baseline=_active_baseline(open_eye=0.32), baseline_used=True)
        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertIn("open_eye_aperture", result["fusion_details"]["baseline_flags"])


# ---------------------------------------------------------------------------
# 15. task handling
# ---------------------------------------------------------------------------
class TaskHandlingTests(unittest.TestCase):
    def test_missing_task_is_absent_not_zero(self):
        self.assertIsNone(compute_task_score(None))

    def test_bool_task_values_are_not_numeric(self):
        self.assertIsNone(compute_task_score({"reaction_time": True, "attempts": True, "errors": False}))

    def test_numeric_string_task_values_rejected(self):
        self.assertIsNone(compute_task_score({"reaction_time": "0.5", "attempts": "3", "errors": "0"}))

    def test_zero_attempts_is_invalid(self):
        self.assertIsNone(compute_task_score({"attempts": 0}))

    def test_negative_reaction_time_ignored(self):
        self.assertIsNone(compute_task_score({"reaction_time": -1.0}))

    def test_valid_task_scores(self):
        score = compute_task_score({"reaction_time": 0.5, "attempts": 3, "errors": 0})
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_missing_task_performance_score_is_none(self):
        result = compute_result(signals=_clean_signals(), task=None)
        self.assertIsNone(result["task_performance_score"])

    def test_task_performance_score_scales_to_percent(self):
        result = compute_result(signals=_clean_signals(), task={"reaction_time": 0.5, "attempts": 3, "errors": 0})
        self.assertIsInstance(result["task_performance_score"], int)
        self.assertGreaterEqual(result["task_performance_score"], 0)
        self.assertLessEqual(result["task_performance_score"], 100)


# ---------------------------------------------------------------------------
# 16. adaptive fusion
# ---------------------------------------------------------------------------
class AdaptiveFusionTests(unittest.TestCase):
    def test_only_readable_modalities_get_weight(self):
        signals = {
            "camera": _signal(None, {"status": "missing"}),
            "video": _ok_video(0.8),
            "voice": _signal(None, {"status": "missing"}),
        }
        result = compute_result(signals=signals)
        weights = result["fusion_details"]["adaptive_weights"]
        self.assertNotIn("audio", weights)
        self.assertNotIn("image", weights)
        self.assertIn("video", weights)

    def test_weights_sum_to_about_one(self):
        result = compute_result(signals=_clean_signals())
        weights = result["fusion_details"]["adaptive_weights"]
        total = sum(v for v in weights.values() if v is not None)
        # Individual weights are rounded for display, so the reported sum is ~1.0.
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_base_weights_constants_preserved(self):
        self.assertEqual(BASE_WEIGHTS["video"], 0.45)
        self.assertEqual(BASE_WEIGHTS["audio"], 0.35)
        self.assertEqual(BASE_WEIGHTS["image"], 0.15)
        self.assertEqual(BASE_WEIGHTS["task"], 0.05)

    def test_all_modalities_fusion_is_confident(self):
        result = compute_result(signals=_clean_signals())
        self.assertGreater(result["confidence"], 0.5)
        self.assertIn(result["risk_level"], VALID_RISK_LEVELS)


# ---------------------------------------------------------------------------
# 17. ML blend safety
# ---------------------------------------------------------------------------
class MLBlendTests(unittest.TestCase):
    def test_non_dict_ml_result_ignored(self):
        base = compute_result(signals=_clean_signals())
        blended = compute_result(signals=_clean_signals(), ml_result="not-a-dict")  # type: ignore[arg-type]
        self.assertEqual(blended["confidence"], base["confidence"])

    def test_ml_confidence_out_of_range_ignored(self):
        base = compute_result(signals=_clean_signals())
        blended = compute_result(signals=_clean_signals(), ml_result={"confidence": 5.0})
        self.assertEqual(blended["readiness_score"], base["readiness_score"])

    def test_ml_nan_confidence_does_not_corrupt_output(self):
        result = compute_result(signals=_clean_signals(), ml_result={"confidence": float("nan")})
        _assert_json_finite(self, result)

    def test_ml_confidence_influences_blend(self):
        low = compute_result(signals=_clean_signals(), ml_result={"confidence": 0.1})
        high = compute_result(signals=_clean_signals(), ml_result={"confidence": 0.9})
        self.assertLessEqual(low["readiness_score"], high["readiness_score"])


# ---------------------------------------------------------------------------
# 18 & 19. risk-level and retake/failure invariants
# ---------------------------------------------------------------------------
class RiskAndRetakeInvariantTests(unittest.TestCase):
    def test_risk_level_constants_exact(self):
        self.assertEqual(VALID_RISK_LEVELS, {"stable", "low_focus", "elevated_fatigue", "high_risk"})

    def test_action_constants_exact(self):
        self.assertEqual(
            VALID_ACTIONS,
            {
                "continue_normal_activity",
                "review_required",
                "rescan_recommended",
                "rest_advised",
                "manager_review",
            },
        )

    def test_risk_level_always_valid_and_never_none(self):
        for signals in (_clean_signals(), _missing_signals(), {}):
            result = compute_result(signals=signals)
            self.assertIn(result["risk_level"], VALID_RISK_LEVELS)
            self.assertIsNotNone(result["risk_level"])

    def test_suggested_action_always_valid(self):
        for signals in (_clean_signals(), _missing_signals(), {}):
            result = compute_result(signals=signals)
            self.assertIn(result["suggested_action"], VALID_ACTIONS)

    def test_result_is_deterministic(self):
        signals = _clean_signals()
        first = compute_result(signals=signals)
        second = compute_result(signals=copy.deepcopy(signals))
        self.assertEqual(first["risk_level"], second["risk_level"])
        self.assertEqual(first["confidence"], second["confidence"])
        self.assertEqual(first["readiness_score"], second["readiness_score"])

    def test_failure_reason_none_when_no_retake(self):
        result = compute_result(signals=_clean_signals())
        self.assertFalse(result["retake_required"])
        self.assertIsNone(result["failure_reason"])

    def test_failure_reason_is_from_allowed_set(self):
        quality = {"status": "failed", "retake_required": True, "failure_reason": "weird_reason", "warnings": []}
        result = compute_result(signals=_clean_signals(), quality=quality)
        self.assertTrue(result["retake_required"])
        self.assertIn(result["failure_reason"], {"low_quality_media", "missing_media"})

    def test_missing_media_requires_retake(self):
        result = compute_result(signals=_missing_signals())
        self.assertTrue(result["retake_required"])


# ---------------------------------------------------------------------------
# 20. observed fatigue score
# ---------------------------------------------------------------------------
class ObservedFatigueTests(unittest.TestCase):
    def test_observed_fatigue_is_int_in_range(self):
        for signals in (_clean_signals(), _missing_signals()):
            result = compute_result(signals=signals)
            self.assertIsInstance(result["observed_fatigue_score"], int)
            self.assertGreaterEqual(result["observed_fatigue_score"], 0)
            self.assertLessEqual(result["observed_fatigue_score"], 100)

    def test_clean_scan_has_zero_observed_fatigue(self):
        self.assertEqual(compute_result(signals=_clean_signals())["observed_fatigue_score"], 0)

    def test_capture_penalties_do_not_inflate_observed_fatigue(self):
        signals = _clean_signals()
        signals["video"] = _ok_video(visual_warnings=["video_blurry", "video_too_dark"])
        signals["voice"] = _ok_voice(audio_warnings=["audio_too_noisy"])
        self.assertEqual(compute_result(signals=signals)["observed_fatigue_score"], 0)

    def test_fatigue_biometrics_raise_observed_fatigue(self):
        signals = {
            "camera": _ok_camera(0.84, face_detected=True),
            "video": _ok_video(
                0.78,
                reliable_eye_landmarks=True,
                sustained_eye_closure=False,
                closed_eye_ratio=0.58,
                avg_eye_aperture=0.16,
                longest_eye_closure_streak=5,
                eye_closure_window_seconds=1.0,
                motion_stability_score=0.74,
            ),
            "voice": _ok_voice(
                0.72,
                speech_presence_score=0.46,
                rms_energy=0.011,
                silence_ratio=0.61,
                quiet_but_usable=True,
            ),
        }
        result = compute_result(signals=signals)
        self.assertGreaterEqual(result["observed_fatigue_score"], 55)
        self.assertIn(result["risk_level"], {"elevated_fatigue", "high_risk"})


# ---------------------------------------------------------------------------
# 21. explanation correctness
# ---------------------------------------------------------------------------
class ExplanationTests(unittest.TestCase):
    def test_explanation_within_length_limit(self):
        for signals in (_clean_signals(), _missing_signals(), {}):
            result = compute_result(signals=signals)
            self.assertLessEqual(len(result["explanation"]), 500)

    def test_explanation_has_no_filesystem_paths(self):
        signals = _clean_signals()
        result = compute_result(signals=signals)
        self.assertNotIn(":\\", result["explanation"])
        self.assertNotIn("/", result["explanation"])

    def test_explanation_never_says_you_look_tired(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": _ok_video(
                0.8,
                reliable_eye_landmarks=True,
                sustained_eye_closure=True,
                motion_stability_score=0.8,
                eye_closure_sample_count=8,
                closed_eye_ratio=0.75,
                longest_eye_closure_streak=4,
                eye_closure_window_ms=600,
                eye_closure_window_seconds=0.6,
                avg_eye_aperture=0.16,
                eye_aperture_std=0.02,
            ),
            "voice": _ok_voice(0.8),
        }
        result = compute_result(signals=signals)
        self.assertNotIn("you look tired", result["explanation"].lower())

    def test_invalid_scan_does_not_claim_real_fatigue(self):
        quality = {"status": "failed", "retake_required": True, "failure_reason": "low_quality_media", "warnings": []}
        result = compute_result(signals=_missing_signals(), quality=quality)
        self.assertNotIn("real fatigue", result["explanation"].lower())

    def test_blur_wording(self):
        signals = _clean_signals()
        signals["video"] = _ok_video(visual_warnings=["video_blurry"])
        self.assertIn("blur", compute_result(signals=signals)["explanation"].lower())

    def test_noise_wording(self):
        signals = _clean_signals()
        signals["voice"] = _ok_voice(audio_warnings=["audio_too_noisy"])
        self.assertIn("noise", compute_result(signals=signals)["explanation"].lower())

    def test_dark_wording(self):
        signals = _clean_signals()
        signals["video"] = _ok_video(visual_warnings=["video_too_dark"])
        explanation = compute_result(signals=signals)["explanation"].lower()
        self.assertTrue("dark" in explanation or "lighting" in explanation)

    def test_face_visibility_wording(self):
        quality = dict(status="weak", weak=True, retake_required=False, warnings=["face_not_visible"])
        result = compute_result(signals=_clean_signals(), quality=quality)
        self.assertIn("face visibility", result["explanation"].lower())

    def test_eye_closure_wording(self):
        signals = {
            "camera": _ok_camera(0.82),
            "video": _ok_video(
                0.8,
                reliable_eye_landmarks=True,
                sustained_eye_closure=True,
                motion_stability_score=0.8,
                eye_closure_sample_count=8,
                closed_eye_ratio=0.75,
                longest_eye_closure_streak=4,
                eye_closure_window_ms=600,
                eye_closure_window_seconds=0.6,
                avg_eye_aperture=0.16,
                eye_aperture_std=0.02,
            ),
            "voice": _ok_voice(0.8),
        }
        self.assertIn("eye closure", compute_result(signals=signals)["explanation"].lower())


# ---------------------------------------------------------------------------
# 22. serialization safety and schema
# ---------------------------------------------------------------------------
class SerializationAndSchemaTests(unittest.TestCase):
    PUBLIC_KEYS = {
        "status",
        "retake_required",
        "failure_reason",
        "readiness_score",
        "observed_fatigue_score",
        "risk_level",
        "confidence",
        "camera_confidence",
        "voice_confidence",
        "task_performance_score",
        "baseline_used",
        "confidence_drift",
        "fatigue_evidence_score",
        "face_metrics",
        "voice_metrics",
        "reaction_metrics",
        "explanation",
        "suggested_action",
        "ai_model_version",
        "modality_scores",
        "fusion_details",
    }

    def test_all_public_keys_present(self):
        result = compute_result(signals=_clean_signals())
        self.assertEqual(set(result.keys()), self.PUBLIC_KEYS)

    def test_result_is_strict_json_serializable(self):
        for signals in (_clean_signals(), _missing_signals(), {}):
            result = compute_result(signals=signals)
            json.dumps(result, allow_nan=False)

    def test_result_has_no_non_finite_floats(self):
        result = compute_result(
            signals=_clean_signals(),
            ml_result={"confidence": float("inf")},
            previous_confidence=float("nan"),
        )
        _assert_json_finite(self, result)

    def test_numeric_public_fields_are_not_bools(self):
        result = compute_result(signals=_clean_signals())
        for field in ("readiness_score", "observed_fatigue_score"):
            self.assertNotIsInstance(result[field], bool)
            self.assertIsInstance(result[field], int)

    def test_baseline_used_is_a_bool(self):
        result = compute_result(signals=_clean_signals())
        self.assertIsInstance(result["baseline_used"], bool)

    def test_model_version_is_reported(self):
        result = compute_result(signals=_clean_signals())
        self.assertEqual(result["ai_model_version"], MODEL_VERSION)


# ---------------------------------------------------------------------------
# 24. focused compatibility mirrors
# ---------------------------------------------------------------------------
class CompatibilityTests(unittest.TestCase):
    def test_fusion_with_only_audio_is_capped(self):
        signals = {
            "camera": _signal(None, {"status": "missing"}),
            "video": _signal(None, {"status": "missing"}),
            "voice": _ok_voice(0.74),
        }
        self.assertLessEqual(compute_result(signals=signals)["confidence"], 0.78)

    def test_fusion_with_only_video_is_capped(self):
        signals = {
            "camera": _signal(None, {"status": "missing"}),
            "video": _ok_video(0.81),
            "voice": _signal(None, {"status": "missing"}),
        }
        self.assertLessEqual(compute_result(signals=signals)["confidence"], 0.78)

    def test_weak_face_reduces_video_score(self):
        clean = compute_result(signals=_clean_signals())
        quality = dict(status="weak", weak=True, retake_required=False, warnings=["face_not_visible"])
        weak = compute_result(signals=_clean_signals(), quality=quality)
        self.assertLess(weak["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertLess(weak["confidence"], clean["confidence"])

    def test_low_quality_reduces_confidence_and_readiness(self):
        good = compute_result(signals=_clean_signals())
        weak_signals = {
            "camera": _ok_camera(0.22, image_warnings=["image_blurry"]),
            "video": _ok_video(0.25, visual_warnings=["video_blurry"]),
            "voice": _ok_voice(0.24, audio_warnings=["audio_too_noisy"]),
        }
        weak = compute_result(signals=weak_signals)
        self.assertLess(weak["confidence"], good["confidence"])
        self.assertLess(weak["readiness_score"], good["readiness_score"])

    def test_empty_signals_are_low_focus_retake(self):
        result = compute_result(signals={})
        self.assertEqual(result["risk_level"], "low_focus")
        self.assertTrue(result["retake_required"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
