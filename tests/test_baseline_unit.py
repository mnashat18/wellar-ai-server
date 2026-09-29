from __future__ import annotations

import copy
import math
from decimal import Decimal
from unittest import TestCase

import baseline


def _signal(score: float = 0.9, details: dict | None = None) -> dict:
    return {"score": score, "details": details or {}}


def _valid_signals() -> dict:
    return {
        "camera": _signal(
            0.8,
            {
                "status": "ok",
                "avg_ear": 0.31,
                "left_eye_aperture": 0.30,
                "right_eye_aperture": 0.32,
                "left_right_eye_asymmetry": 0.02,
            },
        ),
        "voice": _signal(
            0.8,
            {
                "status": "ok",
                "rms_energy": 0.03,
                "speech_rate": 2.0,
            },
        ),
    }


def _baseline_row(*, is_active: bool = True, count: int = 3, confidence: float = 0.8) -> dict:
    return {
        "scan_count": count,
        "eligible_scan_count": count,
        "is_active": is_active,
        "baseline_confidence": confidence,
        "face_avg": {
            "schema_version": baseline.BASELINE_SCHEMA_VERSION,
            "feature_stats": {
                "open_eye_aperture": {"median": 0.31, "mad": 0.03, "count": 3},
                "left_right_eye_asymmetry": {"median": 0.02, "mad": 0.02, "count": 3},
            },
        },
        "voice_avg": {
            "schema_version": baseline.BASELINE_SCHEMA_VERSION,
            "feature_stats": {
                "normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 3},
                "speech_rate": {"median": 2.0, "mad": 0.02, "count": 3},
            },
        },
        "baseline_metadata": {
            "bucket_counts": {"morning": 1, "midday": 1, "evening": 1},
            "samples": {
                "face_avg": {"open_eye_aperture": [0.3, 0.31, 0.32]},
                "voice_avg": {"normalized_voice_energy": [0.03, 0.031, 0.032], "speech_rate": [1.9, 2.0, 2.1]},
            },
        },
    }


class BaselineUnitTests(TestCase):
    def test_strict_numeric_contracts(self):
        class FloatLike:
            def __float__(self):
                return 0.25

        cases = [
            (1, 1.0),
            (0, 0.0),
            (0.5, 0.5),
            (True, None),
            (None, None),
            ("1", None),
            ("", None),
            ("abc", None),
            (Decimal("0.5"), None),
            (FloatLike(), None),
            (float("nan"), None),
            (float("inf"), None),
            (-float("inf"), None),
            (object(), None),
        ]
        for value, expected in cases:
            with self.subTest(value=type(value).__name__):
                self.assertEqual(baseline._coerce_float(value), expected)

    def test_strict_integer_counts(self):
        cases = [
            (3, 3),
            (0, 0),
            (True, None),
            (None, None),
            ("3", None),
            ("", None),
            ("abc", None),
            (3.0, None),
            (3.5, None),
            (float("nan"), None),
            (float("inf"), None),
            (-1, None),
            (object(), None),
        ]
        for value, expected in cases:
            with self.subTest(value=repr(value)):
                self.assertEqual(baseline._strict_int(value, min_value=0), expected)

    def test_clean_sample_list_accepts_lists_and_tuples_only(self):
        original = [0.1, 0.2, None, True, -1.0, float("nan"), float("inf"), 0.3]
        copied = copy.deepcopy(original)
        cases = [
            (original, [0.1, 0.2, 0.3]),
            ((0.4, 0.5, "x"), [0.4, 0.5]),
            ("abc", []),
            ({"a": 1}, []),
            (1.0, []),
            (None, []),
            ((value for value in [0.1, 0.2]), []),
        ]
        for values, expected in cases:
            with self.subTest(values=type(values).__name__):
                self.assertEqual(baseline._clean_sample_list(values), expected)
        self.assertEqual(original, copied)

    def test_clean_sample_list_enforces_limit(self):
        values = [float(index) / 100.0 for index in range(baseline.BASELINE_SAMPLE_LIMIT + 3)]
        cleaned = baseline._clean_sample_list(values)
        self.assertEqual(len(cleaned), baseline.BASELINE_SAMPLE_LIMIT)
        self.assertEqual(cleaned[0], values[-baseline.BASELINE_SAMPLE_LIMIT])

    def test_baseline_count_prefers_eligible_when_valid(self):
        self.assertEqual(baseline._baseline_count({"eligible_scan_count": 4, "scan_count": 1}), 4)

    def test_baseline_count_falls_back_to_scan_count_when_eligible_invalid(self):
        self.assertEqual(baseline._baseline_count({"eligible_scan_count": "bad", "scan_count": 2}), 2)

    def test_baseline_count_resolves_malformed_values_to_zero(self):
        payloads = [
            {"eligible_scan_count": float("nan"), "scan_count": float("inf")},
            {"eligible_scan_count": True, "scan_count": False},
            {"eligible_scan_count": -1, "scan_count": -2},
            {"eligible_scan_count": 2.5, "scan_count": 1.5},
            {},
            None,
            [],
        ]
        for payload in payloads:
            with self.subTest(payload=type(payload).__name__):
                self.assertEqual(baseline._baseline_count(payload), 0)

    def test_bucket_counts_handle_nested_malformed_metadata(self):
        payloads = [
            {"baseline_metadata": None},
            {"baseline_metadata": {"bucket_counts": None}},
            {"baseline_metadata": {"bucket_counts": 1}},
            {"baseline_metadata": {"bucket_counts": {"morning": 1, "midday": "bad", "evening": None}}},
            {"baseline_metadata": {"bucket_counts": {"morning": [1], "midday": {"count": "bad"}, "evening": {"count": -1}}}},
            {"face_avg": None},
            {"face_avg": {"buckets": None}},
            {"face_avg": {"buckets": 1}},
            {"face_avg": {"buckets": {"morning": 1, "midday": "bad", "evening": None}}},
            {"face_avg": {"buckets": {"morning": [1], "midday": {"count": "bad"}, "evening": {"count": -1}}}},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(baseline._bucket_counts(payload), {"morning": 0, "midday": 0, "evening": 0})

    def test_median_and_mad_semantics(self):
        self.assertEqual(baseline._median_and_mad([]), (0.0, 0.0))
        self.assertEqual(baseline._median_and_mad([0.3]), (0.3, 0.0))
        med, mad = baseline._median_and_mad([0.1, 0.1, 0.1, 0.9])
        self.assertEqual(med, 0.1)
        self.assertGreaterEqual(mad, baseline.BASELINE_MAD_FLOOR)

    def test_median_and_mad_remove_nonfinite_outliers(self):
        samples = baseline._clean_sample_list([0.1, 0.1, 0.1, 0.2, float("nan"), float("inf")])
        self.assertEqual(samples, [0.1, 0.1, 0.1, 0.2])
        med, mad = baseline._median_and_mad(samples)
        self.assertTrue(math.isfinite(med))
        self.assertTrue(math.isfinite(mad))

    def test_legacy_baseline_stat_handles_valid_zero_and_malformed_values(self):
        self.assertEqual(
            baseline.legacy_baseline_stat({"face_avg": {"avg": 0.0, "std": 0.0}}, "face_avg"),
            {"median": 0.0, "mad": baseline.BASELINE_MAD_FLOOR},
        )
        self.assertIsNone(baseline.legacy_baseline_stat({"face_avg": {"avg": None, "std": 0.1}}, "face_avg"))
        self.assertIsNone(baseline.legacy_baseline_stat({"face_avg": {"avg": float("nan"), "std": 0.1}}, "face_avg"))
        self.assertIsNone(baseline.legacy_baseline_stat({"face_avg": {"avg": 0.2, "std": -0.1}}, "face_avg"))
        self.assertIsNone(baseline.legacy_baseline_stat({"face_avg": {"avg": 0.2, "std": float("inf")}}, "face_avg"))

    def test_schema_v2_feature_reference_contract(self):
        baseline_row = _baseline_row()
        self.assertEqual(baseline.baseline_feature_reference(baseline_row, "face_avg", "open_eye_aperture"), {"median": 0.31, "mad": 0.03})
        self.assertEqual(baseline.baseline_feature_reference(baseline_row, "voice_avg", "normalized_voice_energy"), {"median": 0.03, "mad": baseline.BASELINE_MAD_FLOOR})
        self.assertIsNone(baseline.baseline_feature_reference(baseline_row, "voice_avg", "speech_rate"))
        for payload in [
            {"face_avg": None},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": "bad"}},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": {"open_eye_aperture": {"median": float("nan"), "mad": 0.1, "count": 1}}}},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": {"open_eye_aperture": {"median": 0.3, "mad": -0.1, "count": 1}}}},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": {"open_eye_aperture": {"median": 0.3, "mad": 0.1, "count": 0}}}},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": {"open_eye_aperture": {"median": True, "mad": 0.1, "count": 1}}}},
            {"face_avg": {"schema_version": baseline.BASELINE_SCHEMA_VERSION, "feature_stats": {"open_eye_aperture": {"median": 0.3, "mad": Decimal("0.1"), "count": 1}}}},
        ]:
            with self.subTest(payload=payload):
                self.assertIsNone(baseline.baseline_feature_reference(payload, "face_avg", "open_eye_aperture"))

    def test_current_baseline_features_fail_closed_on_status_and_stale_details(self):
        valid = baseline.current_baseline_features(signals=_valid_signals(), result={"confidence": 0.9})
        self.assertEqual(valid["face_avg"]["open_eye_aperture"], 0.31)
        self.assertEqual(valid["voice_avg"]["normalized_voice_energy"], 0.03)
        self.assertIsNone(valid["voice_avg"]["speech_rate"])

        cases = [
            (
                {"camera": _signal(0.99, {"status": "missing", "avg_ear": 0.99}), "voice": _signal(0.99, {"status": "ok", "rms_energy": 0.03})},
                None,
                0.03,
            ),
            (
                {"camera": _signal(0.99, {"status": "failed", "avg_ear": 0.99}), "voice": _signal(0.99, {"status": "processing", "rms_energy": 0.99})},
                None,
                None,
            ),
            (
                {"camera": _signal(0.99, {"status": "ok", "avg_ear": float("nan")}), "voice": _signal(0.99, {"status": "ok", "rms_energy": float("inf")})},
                None,
                None,
            ),
            (
                {"camera": _signal(0.99, {"status": "ok", "avg_ear": -1.0}), "voice": _signal(0.99, {"status": "ok", "rms_energy": -0.1})},
                None,
                None,
            ),
            (
                {"camera": _signal(0.99, {"status": "ok", "avg_ear": 1.5}), "voice": _signal(0.99, {"status": "ok", "rms_energy": 1.5})},
                None,
                None,
            ),
        ]
        for signals, expected_face, expected_voice in cases:
            with self.subTest(signals=signals):
                features = baseline.current_baseline_features(signals=signals, result=None)
                self.assertEqual(features["face_avg"]["open_eye_aperture"], expected_face)
                self.assertEqual(features["voice_avg"]["normalized_voice_energy"], expected_voice)

    def test_feature_presence_requirements_are_deterministic(self):
        valid = _valid_signals()
        self.assertEqual(baseline._feature_presence_requirements(valid)[0], True)
        cases = [
            ({}, False, ["missing_face_baseline_feature", "missing_voice_baseline_feature"]),
            ({"camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31})}, False, ["missing_voice_baseline_feature"]),
            ({"voice": _signal(0.8, {"status": "ok", "rms_energy": 0.03})}, False, ["missing_face_baseline_feature"]),
            ({"camera": _signal(0.8, {"status": "ok", "left_right_eye_asymmetry": 0.02}), "voice": _signal(0.8, {"status": "ok", "rms_energy": 0.03})}, False, ["missing_face_baseline_feature"]),
            ({"camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31}), "voice": _signal(0.8, {"status": "ok", "speech_rate": 2.0})}, False, ["missing_voice_baseline_feature"]),
        ]
        for signals, expected, reasons in cases:
            with self.subTest(signals=signals):
                present, got_reasons, _ = baseline._feature_presence_requirements(signals)
                self.assertEqual(present, expected)
                self.assertEqual(got_reasons, reasons)

    def test_task_completion_status_handles_malformed_objects(self):
        class BadTask:
            @property
            def reaction_time(self):
                raise RuntimeError("boom")

        self.assertEqual(
            baseline._task_completion_status(validation_result={"warnings": []}, quality_result={"task_quality": "good"}, expected_phrase=None, task=None),
            "not_required",
        )
        self.assertEqual(
            baseline._task_completion_status(validation_result={"warnings": ["speech_not_detected"]}, quality_result={"task_quality": "good"}, expected_phrase="  phrase  ", task=None),
            "incomplete_required_speech",
        )
        self.assertEqual(
            baseline._task_completion_status(validation_result={"warnings": []}, quality_result={"task_quality": "failed"}, expected_phrase=None, task={"attempts": 1}),
            "incomplete_required_task",
        )
        self.assertEqual(
            baseline._task_completion_status(validation_result={"warnings": []}, quality_result={"task_quality": "good"}, expected_phrase="maybe", task={"attempts": 0, "errors": 0, "reaction_time": 0}),
            "completed",
        )
        self.assertEqual(
            baseline._task_completion_status(validation_result={"warnings": []}, quality_result={"task_quality": "good"}, expected_phrase=None, task=BadTask()),
            "not_required",
        )

    def test_evaluate_baseline_eligibility_fails_closed_and_is_deterministic(self):
        valid = baseline.evaluate_baseline_eligibility(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "passed": True, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": []},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals=_valid_signals(),
            expected_phrase=None,
            manually_unreliable=False,
        )
        self.assertTrue(valid["eligible"])
        self.assertEqual(valid["reasons"], [])
        self.assertIsInstance(valid["hard_gates_triggered"], list)

        malformed_variants = [
            {"quality_result": None, "validation_result": None, "result": None, "signals": None},
            {"quality_result": {"status": "weak", "weak": True, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.1}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.1}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.1, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": ["boom"], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": ["audio_too_noisy"]}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": ["video_low_resolution"]}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "unstable"}, "signals": _valid_signals()},
            {"quality_result": {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9}, "validation_result": {"critical_errors": [], "warnings": []}, "result": {"confidence": 0.8, "risk_level": "stable"}, "signals": {}, "manually_unreliable": True},
        ]
        for kwargs in malformed_variants:
            with self.subTest(kwargs=kwargs):
                eligible = baseline.evaluate_baseline_eligibility(**kwargs)
                self.assertFalse(eligible["eligible"])
                self.assertEqual(eligible["reasons"], list(dict.fromkeys(eligible["reasons"])))

    def test_malformed_baseline_types_are_safe_across_public_helpers(self):
        malformed_values = [None, "", [], (), 0, True, object()]
        for malformed in malformed_values:
            with self.subTest(malformed=type(malformed).__name__):
                self.assertEqual(baseline._baseline_count(malformed), 0)
                self.assertEqual(baseline._bucket_counts(malformed), {"morning": 0, "midday": 0, "evening": 0})
                self.assertEqual(baseline._existing_feature_samples(malformed, "face_avg", baseline.FACE_FEATURES), {"open_eye_aperture": [], "left_right_eye_asymmetry": []})
                self.assertEqual(baseline._existing_feature_samples(malformed, "voice_avg", baseline.VOICE_FEATURES), {"normalized_voice_energy": [], "speech_rate": []})
                self.assertEqual(baseline._existing_feature_samples(malformed, "reaction_avg", baseline.REACTION_FEATURES), {})
                self.assertEqual(set(baseline.baseline_signal_payload(malformed, signals=_valid_signals(), scanned_at=None).keys()), {"scan_count", "face_avg", "voice_avg", "reaction_avg", "is_active"})
                status = baseline.baseline_status_payload(malformed)
                self.assertIsInstance(status, dict)
                self.assertFalse(status["is_active"])
                self.assertEqual(status["scan_count"], 0)
                self.assertEqual(status["eligible_scan_count"], 0)
                self.assertGreaterEqual(status["scans_remaining"], 0)
                self.assertFalse(baseline.baseline_ready_for_scoring(malformed))
                self.assertFalse(baseline.baseline_has_valid_personalization_references(malformed))
                self.assertFalse(baseline.baseline_ready_for_personalized_scoring(malformed, quality_result={}, validation_result={}, result={}, unique_row=True))
                self.assertIsNone(baseline.legacy_baseline_stat(malformed, "face_avg"))
                self.assertIsNone(baseline.baseline_feature_reference(malformed, "face_avg", "open_eye_aperture"))

    def test_baseline_signal_payload_is_exact_and_immutable(self):
        current = _baseline_row()
        original = copy.deepcopy(current)
        payload = baseline.baseline_signal_payload(current, signals=_valid_signals(), scanned_at="2026-07-01T08:00:00Z")
        self.assertEqual(set(payload.keys()), {"scan_count", "face_avg", "voice_avg", "reaction_avg", "is_active"})
        self.assertEqual(current, original)
        self.assertEqual(payload["scan_count"], 4)
        self.assertIsInstance(payload["is_active"], bool)
        self.assertIn("feature_stats", payload["face_avg"])
        self.assertIn("feature_stats", payload["voice_avg"])
        self.assertIn("feature_stats", payload["reaction_avg"])

    def test_baseline_signal_payload_uses_only_valid_samples(self):
        baseline_row = {
            "scan_count": 1,
            "is_active": False,
            "face_avg": {
                "schema_version": baseline.BASELINE_SCHEMA_VERSION,
                "feature_stats": {"open_eye_aperture": {"median": 0.3, "mad": 0.02, "count": 1}},
                "feature_samples": {"open_eye_aperture": [0.3, float("nan"), 0.31]},
            },
            "voice_avg": {
                "schema_version": baseline.BASELINE_SCHEMA_VERSION,
                "feature_stats": {"normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 1}},
                "feature_samples": {"normalized_voice_energy": [0.03, 0.031, float("inf")]},
            },
        }
        payload = baseline.baseline_signal_payload(baseline_row, signals=_valid_signals(), scanned_at=None)
        self.assertEqual(payload["face_avg"]["feature_stats"]["open_eye_aperture"]["count"], 3)
        self.assertEqual(payload["voice_avg"]["feature_stats"]["normalized_voice_energy"]["count"], 3)
        self.assertNotIn(float("nan"), payload["face_avg"].get("feature_samples", {}).get("open_eye_aperture", []))
        self.assertNotIn(float("inf"), payload["voice_avg"].get("feature_samples", {}).get("normalized_voice_energy", []))

    def test_baseline_signal_payload_rejects_invalid_current_evidence(self):
        payload = baseline.baseline_signal_payload(None, signals={"camera": _signal(0.9, {"status": "missing", "avg_ear": 0.99}), "voice": _signal(0.9, {"status": "failed", "rms_energy": 0.99})}, scanned_at="bad")
        self.assertEqual(payload["face_avg"]["feature_stats"]["open_eye_aperture"]["count"], 0)
        self.assertEqual(payload["voice_avg"]["feature_stats"]["normalized_voice_energy"]["count"], 0)

    def test_baseline_status_payload_is_safe_and_truthful(self):
        row = _baseline_row()
        status = baseline.baseline_status_payload(row)
        self.assertTrue(status["is_active"])
        self.assertEqual(status["scan_count"], 3)
        self.assertEqual(status["eligible_scan_count"], 3)
        self.assertEqual(status["baseline_confidence"], 0.8)
        self.assertIsInstance(status["needs_morning_scan"], bool)
        self.assertIsInstance(status["needs_evening_scan"], bool)

        cases = [
            {"scan_count": 9, "eligible_scan_count": 9, "is_active": False, "baseline_confidence": 0.0},
            {"scan_count": 9, "eligible_scan_count": 9, "baseline_confidence": 0.75},
            {"scan_count": 9, "eligible_scan_count": 9, "is_active": "true", "baseline_confidence": "bad"},
            {"scan_count": 9, "eligible_scan_count": 9, "is_active": 1},
            {"scan_count": 9, "eligible_scan_count": 9},
            {"scan_count": "bad", "eligible_scan_count": float("nan"), "baseline_confidence": "0.0", "is_active": "true", "baseline_metadata": {"bucket_counts": {"morning": "bad", "midday": float("inf"), "evening": -1}}},
            {"scan_count": 9, "eligible_scan_count": 9, "baseline_metadata": {"bucket_counts": {"morning": {"count": "bad"}, "midday": [1], "evening": None}}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status = baseline.baseline_status_payload(payload)
                self.assertIsInstance(status, dict)
                self.assertFalse(status["is_active"])
                self.assertGreaterEqual(status["scans_remaining"], 0)
                self.assertIsInstance(status["needs_morning_scan"], bool)
                self.assertIsInstance(status["needs_evening_scan"], bool)
                if payload.get("baseline_confidence") == 0.0:
                    self.assertEqual(status["baseline_confidence"], 0.0)

    def test_baseline_ready_for_scoring_is_fail_closed(self):
        active = _baseline_row(is_active=True, count=baseline.BASELINE_USE_AFTER)
        inactive = _baseline_row(is_active=False, count=baseline.BASELINE_USE_AFTER)
        malformed = {"scan_count": baseline.BASELINE_USE_AFTER, "eligible_scan_count": baseline.BASELINE_USE_AFTER, "is_active": "true"}
        self.assertTrue(baseline.baseline_ready_for_scoring(active))
        self.assertFalse(baseline.baseline_ready_for_scoring(inactive))
        self.assertFalse(baseline.baseline_ready_for_scoring(malformed))

    def test_baseline_has_valid_personalization_references_supports_valid_reference_and_ignores_speech_rate_only(self):
        row = _baseline_row()
        self.assertTrue(baseline.baseline_has_valid_personalization_references(row))

        speech_rate_only = _baseline_row()
        speech_rate_only["face_avg"]["feature_stats"] = {}
        speech_rate_only["voice_avg"]["feature_stats"] = {"speech_rate": {"median": 2.0, "mad": 0.02, "count": 3}}
        self.assertFalse(baseline.baseline_has_valid_personalization_references(speech_rate_only))

        valid_reference_with_speech_rate = _baseline_row()
        valid_reference_with_speech_rate["voice_avg"]["feature_stats"]["speech_rate"] = {"median": 2.0, "mad": 0.02, "count": 3}
        valid_reference_with_speech_rate["face_avg"]["feature_stats"] = {}
        self.assertTrue(baseline.baseline_has_valid_personalization_references(valid_reference_with_speech_rate))

    def test_baseline_ready_for_personalized_scoring_requires_validation_passed(self):
        baseline_row = _baseline_row(is_active=True, count=baseline.BASELINE_USE_AFTER)
        valid_kwargs = dict(
            baseline=baseline_row,
            quality_result={"status": "passed", "passed": True, "weak": False, "retake_required": False, "warnings": [], "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": [], "passed": True},
            result={"confidence": 0.8},
            task=None,
            expected_phrase=None,
            unique_row=True,
        )
        self.assertTrue(baseline.baseline_ready_for_personalized_scoring(**valid_kwargs))

        for passed_value in [None, False, "true", 1]:
            kwargs = copy.deepcopy(valid_kwargs)
            kwargs["validation_result"]["passed"] = passed_value
            with self.subTest(passed_value=passed_value):
                self.assertFalse(baseline.baseline_ready_for_personalized_scoring(**kwargs))

    def test_baseline_status_payload_and_signal_payload_accept_timestamp_variants(self):
        current = None
        for scanned_at in ["2026-07-01T08:00:00Z", "2026-07-01T08:00:00+02:00", None, "not-a-timestamp"]:
            with self.subTest(scanned_at=scanned_at):
                payload = baseline.baseline_signal_payload(current, signals=_valid_signals(), scanned_at=scanned_at)
                self.assertEqual(set(payload.keys()), {"scan_count", "face_avg", "voice_avg", "reaction_avg", "is_active"})
                current = payload

    def test_old_speech_rate_samples_are_preserved_but_ignored(self):
        row = {
            "scan_count": 3,
            "is_active": True,
            "voice_avg": {
                "schema_version": baseline.BASELINE_SCHEMA_VERSION,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 3},
                    "speech_rate": {"median": 2.0, "mad": 0.02, "count": 3},
                },
                "feature_samples": {
                    "normalized_voice_energy": [0.03, 0.04, 0.05],
                    "speech_rate": [1.9, 2.0, 2.1],
                },
            },
            "face_avg": {
                "schema_version": baseline.BASELINE_SCHEMA_VERSION,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.31, "mad": 0.03, "count": 3},
                    "left_right_eye_asymmetry": {"median": 0.02, "mad": 0.02, "count": 3},
                },
            },
        }
        payload = baseline.baseline_signal_payload(row, signals=_valid_signals(), scanned_at="2026-07-01T08:00:00Z")
        self.assertIn("speech_rate", payload["voice_avg"]["feature_stats"])
        self.assertIn("speech_rate", payload["voice_avg"].get("feature_samples", {}))
        row["face_avg"]["feature_stats"] = {}
        row["voice_avg"]["feature_stats"] = {"speech_rate": {"median": 2.0, "mad": 0.02, "count": 3}}
        self.assertFalse(baseline.baseline_has_valid_personalization_references(row))
