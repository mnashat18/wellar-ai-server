from __future__ import annotations

import math
import unittest

import quality


def _signal(kind: str, *, score: float | None, details_overrides: dict[str, object] | None = None, top_score: float | None = None):
    details_overrides = details_overrides or {}
    if kind == "video":
        details = {
            "status": "ok",
            "duration_seconds": 5.0,
            "visual_quality_score": score,
            "visual_warnings": [],
        }
    elif kind == "audio":
        details = {
            "status": "ok",
            "duration_seconds": 4.0,
            "audio_quality_score": score,
            "audio_warnings": [],
        }
    elif kind == "image":
        details = {
            "status": "ok",
            "image_quality_score": score,
            "image_warnings": [],
        }
    else:
        raise ValueError(kind)
    details.update(details_overrides)
    return {"score": top_score if top_score is not None else 0.11, "details": details}


class QualityUnitTests(unittest.TestCase):
    def _assert_schema(self, result: dict):
        expected_keys = {
            "status",
            "passed",
            "weak",
            "failure_reason",
            "reasons",
            "weak_reasons",
            "retake_required",
            "suggested_action",
            "usable_modalities",
            "weak_modalities",
            "missing_modalities",
            "present_modalities",
            "confidence_multiplier",
            "task_quality",
            "warnings",
            "media_quality",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["passed"], bool)
        self.assertTrue(result["passed"])
        self.assertIn(result["status"], {"passed", "weak"})
        self.assertEqual(result["weak"], result["status"] == "weak")
        self.assertIn(result["suggested_action"], {"continue_normal_activity", "review_required", "rescan_recommended"})
        self.assertIn(result["task_quality"], {"missing", "good", "weak", "failed"})
        self.assertIsInstance(result["usable_modalities"], int)
        self.assertIsInstance(result["present_modalities"], int)
        self.assertIsInstance(result["weak_modalities"], list)
        self.assertIsInstance(result["missing_modalities"], list)
        self.assertIsInstance(result["warnings"], list)
        self.assertIsInstance(result["reasons"], list)
        self.assertIsInstance(result["weak_reasons"], list)
        self.assertLessEqual(result["usable_modalities"], result["present_modalities"])
        self.assertEqual(result["present_modalities"], result["usable_modalities"] + len(result["weak_modalities"]))
        self.assertEqual(result["reasons"], list(dict.fromkeys(result["reasons"])))
        self.assertEqual(result["warnings"], list(dict.fromkeys(result["warnings"])))
        self.assertEqual(result["weak_reasons"], list(dict.fromkeys(result["weak_reasons"])))
        self.assertIsInstance(result["media_quality"], dict)
        self.assertEqual(set(result["media_quality"].keys()), {"aggregate_quality", "video", "audio", "image"})
        self.assertTrue(math.isfinite(result["confidence_multiplier"]))
        self.assertGreaterEqual(result["confidence_multiplier"], 0.0)
        self.assertLessEqual(result["confidence_multiplier"], 1.0)
        self.assertTrue(math.isfinite(result["media_quality"]["aggregate_quality"]))
        self.assertGreaterEqual(result["media_quality"]["aggregate_quality"], 0.0)
        self.assertLessEqual(result["media_quality"]["aggregate_quality"], 1.0)
        for name in ("video", "audio", "image"):
            modality = result["media_quality"][name]
            self.assertEqual(set(modality.keys()), {"present", "usable", "weak", "score", "warnings"})
            self.assertIsInstance(modality["present"], bool)
            self.assertIsInstance(modality["usable"], bool)
            self.assertIsInstance(modality["weak"], bool)
            self.assertIsInstance(modality["warnings"], list)
            self.assertEqual(modality["usable"], modality["present"] and not modality["weak"])

    def test_empty_signals_produce_missing_media_and_retake(self):
        result = quality.assess_quality({})
        self._assert_schema(result)
        self.assertEqual(result["status"], "weak")
        self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)
        self.assertTrue(result["retake_required"])
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertEqual(result["present_modalities"], 0)
        self.assertEqual(result["usable_modalities"], 0)
        self.assertEqual(result["missing_modalities"], ["video", "audio", "image"])

    def test_signals_none_does_not_crash(self):
        result = quality.assess_quality(None)
        self._assert_schema(result)
        self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)

    def test_non_dict_signals_do_not_crash(self):
        result = quality.assess_quality("not-a-dict")
        self._assert_schema(result)
        self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)

    def test_only_normalized_ok_status_is_readable(self):
        cases = [
            ("failed", False),
            ("corrupt", False),
            ("processing", False),
            ("unreadable", False),
            ("arbitrary_unknown_value", False),
            (" OK ", True),
        ]
        for status, expected_present in cases:
            with self.subTest(status=status):
                result = quality.assess_quality({
                    "video": _signal("video", score=0.9, details_overrides={"status": status}),
                    "voice": _signal("audio", score=0.9, details_overrides={"status": status}),
                    "camera": _signal("image", score=0.9, details_overrides={"status": status}),
                })
                self._assert_schema(result)
                if expected_present:
                    self.assertEqual(result["present_modalities"], 3)
                    self.assertEqual(result["failure_reason"], None)
                else:
                    self.assertEqual(result["present_modalities"], 0)
                    self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)

    def test_missing_status_with_stale_score_remains_missing(self):
        result = quality.assess_quality({
            "video": _signal("video", score=None, top_score=0.99, details_overrides={"status": "missing", "visual_quality_score": None}),
            "voice": _signal("audio", score=None, top_score=0.99, details_overrides={"status": "missing", "audio_quality_score": None, "duration_seconds": None, "duration_sec": None}),
            "camera": _signal("image", score=None, top_score=0.99, details_overrides={"status": "missing", "image_quality_score": None}),
        })
        self._assert_schema(result)
        self.assertEqual(result["present_modalities"], 0)
        self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)

    def test_video_uses_visual_quality_score_not_top_level_score(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.20, top_score=0.99),
            "voice": _signal("audio", score=0.90),
            "camera": _signal("image", score=0.90),
        })
        self._assert_schema(result)
        self.assertAlmostEqual(result["media_quality"]["video"]["score"], 0.20)
        self.assertEqual(result["media_quality"]["video"]["usable"], False)
        self.assertEqual(result["status"], "weak")

    def test_audio_uses_audio_quality_score_not_top_level_score(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.90),
            "voice": _signal("audio", score=0.20, top_score=0.99),
            "camera": _signal("image", score=0.90),
        })
        self._assert_schema(result)
        self.assertAlmostEqual(result["media_quality"]["audio"]["score"], 0.20)
        self.assertFalse(result["media_quality"]["audio"]["usable"])

    def test_image_uses_image_quality_score_not_top_level_score(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.90),
            "voice": _signal("audio", score=0.90),
            "camera": _signal("image", score=0.20, top_score=0.99),
        })
        self._assert_schema(result)
        self.assertAlmostEqual(result["media_quality"]["image"]["score"], 0.20)
        self.assertFalse(result["media_quality"]["image"]["usable"])

    def test_low_top_level_score_can_still_be_usable_with_real_quality(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.90, top_score=0.10),
            "voice": _signal("audio", score=0.90, top_score=0.10),
            "camera": _signal("image", score=0.90, top_score=0.10),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["media_quality"]["video"]["usable"])
        self.assertTrue(result["media_quality"]["audio"]["usable"])
        self.assertTrue(result["media_quality"]["image"]["usable"])

    def test_degrading_warnings_preserve_usable_evidence_and_reach_quality_result(self):
        cases = [
            ("video", "visual_warnings", "video_too_dark", False),
            ("video", "visual_warnings", "video_blurry", False),
            ("video", "visual_warnings", "unstable_video", False),
            ("video", "visual_warnings", "unstable_camera", False),
            ("video", "visual_warnings", "video_low_resolution", False),
            ("image", "image_warnings", "image_too_dark", False),
            ("image", "image_warnings", "image_blurry", False),
            ("image", "image_warnings", "image_low_resolution", False),
            ("audio", "audio_warnings", "audio_too_noisy", False),
            ("audio", "audio_warnings", "audio_too_quiet", True),
            ("audio", "audio_warnings", "too_much_silence", True),
            ("audio", "audio_warnings", "audio_clipping", True),
        ]
        for modality, warning_key, warning, speech_required in cases:
            with self.subTest(modality=modality, warning=warning):
                signals = {
                    "video": _signal("video", score=0.65),
                    "voice": _signal("audio", score=0.92),
                    "camera": _signal("image", score=0.86),
                }
                signal_key = {"video": "video", "audio": "voice", "image": "camera"}[modality]
                signals[signal_key]["details"][warning_key] = [warning]

                result = quality.assess_quality(signals, speech_required=speech_required)

                media = result["media_quality"][modality]
                self.assertTrue(media["present"])
                self.assertTrue(media["usable"])
                self.assertFalse(media["weak"])
                self.assertIn(warning, media["warnings"])
                self.assertIn(warning, result["warnings"])

    def test_evidence_blocking_warnings_remain_unusable(self):
        cases = [
            ("video", "visual_warnings", "face_not_visible", False),
            ("video", "visual_warnings", "insufficient_usable_frames", False),
            ("audio", "audio_warnings", "speech_not_detected", True),
            ("video", "visual_warnings", "video_too_short", False),
            ("audio", "audio_warnings", "audio_too_short", False),
        ]
        for modality, warning_key, warning, speech_required in cases:
            with self.subTest(modality=modality, warning=warning):
                signals = {
                    "video": _signal("video", score=0.65),
                    "voice": _signal("audio", score=0.92),
                    "camera": _signal("image", score=0.86),
                }
                signal_key = {"video": "video", "audio": "voice", "image": "camera"}[modality]
                signals[signal_key]["details"][warning_key] = [warning]

                result = quality.assess_quality(signals, speech_required=speech_required)

                self.assertFalse(result["media_quality"][modality]["usable"])
                self.assertTrue(result["media_quality"][modality]["present"])

    def test_unreadable_required_analyzer_payload_stays_missing(self):
        for status in ("corrupt", "error", "unreadable"):
            with self.subTest(status=status):
                result = quality.assess_quality({
                    "video": _signal("video", score=0.9, details_overrides={"status": status}),
                    "voice": _signal("audio", score=0.9),
                    "camera": _signal("image", score=0.9),
                })
                self.assertFalse(result["media_quality"]["video"]["present"])
                self.assertFalse(result["media_quality"]["video"]["usable"])

    def test_missing_quality_score_and_nonfinite_values_are_unreadable(self):
        cases = [
            ("missing", None),
            ("nan", float("nan")),
            ("infinity", float("inf")),
        ]
        for label, value in cases:
            with self.subTest(label=label):
                payload = {
                    "video": _signal("video", score=value, details_overrides={"visual_quality_score": value, "duration_seconds": None}),
                    "voice": _signal("audio", score=value, details_overrides={"audio_quality_score": value, "duration_seconds": None, "duration_sec": None}),
                    "camera": _signal("image", score=value, details_overrides={"image_quality_score": value}),
                }
                result = quality.assess_quality(payload)
                self._assert_schema(result)
                self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)

    def test_video_and_audio_missing_duration_are_unreadable(self):
        video_result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"duration_seconds": None}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        audio_result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9, details_overrides={"duration_seconds": None, "duration_sec": None}),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(video_result)
        self._assert_schema(audio_result)
        self.assertFalse(video_result["media_quality"]["video"]["present"])
        self.assertFalse(audio_result["media_quality"]["audio"]["present"])
        self.assertIsNone(video_result["failure_reason"])
        self.assertIsNone(audio_result["failure_reason"])

    def test_negative_duration_is_unreadable(self):
        video_result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"duration_seconds": -1.0}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        audio_result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9, details_overrides={"duration_seconds": -1.0}),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(video_result)
        self._assert_schema(audio_result)
        self.assertFalse(video_result["media_quality"]["video"]["present"])
        self.assertFalse(audio_result["media_quality"]["audio"]["present"])
        self.assertIsNone(video_result["failure_reason"])
        self.assertIsNone(audio_result["failure_reason"])

    def test_valid_image_minimum_evidence_is_readable(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.38),
        })
        self._assert_schema(result)
        self.assertTrue(result["media_quality"]["image"]["present"])
        self.assertTrue(result["media_quality"]["image"]["usable"])

    def test_quality_below_threshold_is_weak_and_at_threshold_is_usable(self):
        below = quality.assess_quality({
            "video": _signal("video", score=0.41),
            "voice": _signal("audio", score=0.39),
            "camera": _signal("image", score=0.37),
        })
        exact = quality.assess_quality({
            "video": _signal("video", score=0.42),
            "voice": _signal("audio", score=0.40),
            "camera": _signal("image", score=0.38),
        })
        self._assert_schema(below)
        self._assert_schema(exact)
        self.assertEqual(below["status"], "weak")
        self.assertTrue(exact["media_quality"]["video"]["usable"])
        self.assertTrue(exact["media_quality"]["audio"]["usable"])
        self.assertTrue(exact["media_quality"]["image"]["usable"])

    def test_degrading_warning_preserves_usability_but_recommends_retake(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["video_blurry"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "weak")
        self.assertIn("video_blurry", result["warnings"])
        self.assertIn("video_blurry", result["weak_reasons"])
        self.assertTrue(result["media_quality"]["video"]["usable"])
        self.assertFalse(result["media_quality"]["video"]["weak"])
        self.assertTrue(result["retake_required"])

    def test_strong_modality_is_also_usable(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["media_quality"]["video"]["usable"])
        self.assertTrue(result["media_quality"]["audio"]["usable"])
        self.assertTrue(result["media_quality"]["image"]["usable"])

    def test_sustained_eye_closure_stays_in_public_warnings_only(self):
        with_eye = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        clean = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(with_eye)
        self._assert_schema(clean)
        self.assertIn("sustained_eye_closure", with_eye["warnings"])
        self.assertNotIn("sustained_eye_closure", with_eye["weak_reasons"])
        self.assertFalse(with_eye["retake_required"])
        self.assertEqual(with_eye["confidence_multiplier"], clean["confidence_multiplier"])

    def test_sustained_eye_closure_alone_does_not_make_quality_weak(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["weak"])
        self.assertFalse(result["retake_required"])

    def test_sustained_eye_closure_does_not_reduce_confidence(self):
        clean = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        with_eye = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(clean)
        self._assert_schema(with_eye)
        self.assertEqual(clean["confidence_multiplier"], with_eye["confidence_multiplier"])

    def test_sustained_eye_closure_does_not_require_retake(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertFalse(result["retake_required"])

    def test_speech_required_false_suppresses_speech_not_detected(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9, details_overrides={"audio_warnings": ["speech_not_detected"]}),
            "camera": _signal("image", score=0.9),
        }, speech_required=False)
        self._assert_schema(result)
        self.assertNotIn("speech_not_detected", result["warnings"])
        self.assertNotIn("speech_not_detected", result["weak_reasons"])
        self.assertTrue(result["media_quality"]["audio"]["usable"])
        self.assertFalse(result["retake_required"])

    def test_speech_required_true_retains_speech_not_detected(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9, details_overrides={"audio_warnings": ["speech_not_detected"]}),
            "camera": _signal("image", score=0.9),
        }, speech_required=True)
        self._assert_schema(result)
        self.assertIn("speech_not_detected", result["warnings"])
        self.assertIn("speech_not_detected", result["weak_reasons"])
        self.assertTrue(result["weak"])
        self.assertTrue(result["retake_required"])

    def test_quiet_but_usable_true_suppresses_audio_too_quiet(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.82, details_overrides={"audio_warnings": ["audio_too_quiet"], "quiet_but_usable": True}),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertNotIn("audio_too_quiet", result["warnings"])
        self.assertNotIn("audio_too_quiet", result["weak_reasons"])
        self.assertTrue(result["media_quality"]["audio"]["usable"])

    def test_truthy_non_bool_quiet_but_usable_does_not_suppress(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.82, details_overrides={"audio_warnings": ["audio_too_quiet"], "quiet_but_usable": 1}),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertIn("audio_too_quiet", result["warnings"])
        self.assertIn("audio_too_quiet", result["weak_reasons"])

    def test_genuine_audio_noise_clipping_and_silence_warnings_remain_effective(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal(
                "audio",
                score=0.9,
                details_overrides={"audio_warnings": ["audio_too_noisy", "audio_clipping", "too_much_silence"]},
            ),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "weak")
        self.assertTrue(result["retake_required"])
        self.assertTrue({"audio_too_noisy", "audio_clipping", "too_much_silence"} <= set(result["warnings"]))

    def test_warning_order_is_preserved_and_duplicates_removed(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.2, details_overrides={"visual_warnings": ["video_blurry", "video_blurry", "sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.2, details_overrides={"audio_warnings": ["audio_too_noisy", "audio_too_noisy", "audio_clipping"]}),
            "camera": _signal("image", score=0.2, details_overrides={"image_warnings": ["image_blurry", "image_blurry"]}),
        })
        self._assert_schema(result)
        self.assertEqual(
            result["warnings"],
            ["video_blurry", "sustained_eye_closure", "audio_too_noisy", "audio_clipping", "image_blurry"],
        )

    def test_modality_states_are_exclusive(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.2),
            "camera": {"score": None, "details": {"status": "missing"}},
        })
        self._assert_schema(result)
        self.assertIn("audio", result["weak_modalities"])
        self.assertIn("image", result["missing_modalities"])
        self.assertNotIn("audio", result["missing_modalities"])
        self.assertNotIn("image", result["weak_modalities"])

    def test_counts_satisfy_state_invariant(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.2),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertEqual(result["present_modalities"], result["usable_modalities"] + len(result["weak_modalities"]))

    def test_aggregate_quality_uses_only_readable_quality_scores(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9, top_score=0.1),
            "voice": _signal("audio", score=None, top_score=0.99, details_overrides={"audio_quality_score": None, "duration_seconds": None, "duration_sec": None, "status": "missing"}),
            "camera": _signal("image", score=0.1, top_score=0.99),
        })
        self._assert_schema(result)
        self.assertAlmostEqual(result["media_quality"]["aggregate_quality"], 0.5)
        self.assertAlmostEqual(result["media_quality"]["video"]["score"], 0.9)
        self.assertAlmostEqual(result["media_quality"]["image"]["score"], 0.1)
        self.assertEqual(result["media_quality"]["audio"]["score"], None)

    def test_confidence_multiplier_is_finite_and_bounded(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertTrue(math.isfinite(result["confidence_multiplier"]))
        self.assertGreaterEqual(result["confidence_multiplier"], 0.0)
        self.assertLessEqual(result["confidence_multiplier"], 1.0)

    def test_malformed_task_values_do_not_crash(self):
        cases = [
            {"attempts": "3", "reaction_time": "0.8", "errors": "0"},
            {"attempts": True, "reaction_time": 0.8, "errors": 0},
            {"attempts": -1, "reaction_time": 0.8, "errors": 0},
            {"attempts": 3, "reaction_time": 0.0, "errors": 0},
            {"attempts": 3, "reaction_time": float("nan"), "errors": 0},
            {"attempts": 3, "reaction_time": 0.8, "errors": -1},
            {"attempts": 3, "reaction_time": 0.8, "errors": float("inf")},
            {},
        ]
        for task in cases:
            with self.subTest(task=task):
                result = quality.assess_quality({
                    "video": _signal("video", score=0.9),
                    "voice": _signal("audio", score=0.9),
                    "camera": _signal("image", score=0.9),
                }, task=task)
                self._assert_schema(result)

    def test_valid_task_with_three_attempts_is_good(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        }, task={"attempts": 3, "reaction_time": 0.8, "errors": 0})
        self._assert_schema(result)
        self.assertEqual(result["task_quality"], "good")

    def test_valid_task_with_fewer_than_three_attempts_is_weak(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        }, task={"attempts": 2, "reaction_time": 0.8, "errors": 0})
        self._assert_schema(result)
        self.assertEqual(result["task_quality"], "weak")

    def test_invalid_task_is_failed(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        }, task={"attempts": 2, "reaction_time": -1, "errors": 0})
        self._assert_schema(result)
        self.assertEqual(result["task_quality"], "failed")

    def test_task_does_not_alter_media_counts(self):
        result_without_task = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.2),
            "camera": _signal("image", score=0.9),
        })
        result_with_task = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.2),
            "camera": _signal("image", score=0.9),
        }, task={"attempts": 3, "reaction_time": 0.8, "errors": 0})
        self._assert_schema(result_without_task)
        self._assert_schema(result_with_task)
        self.assertEqual(result_without_task["usable_modalities"], result_with_task["usable_modalities"])
        self.assertEqual(result_without_task["present_modalities"], result_with_task["present_modalities"])

    def test_failure_reason_and_action_invariants(self):
        missing = quality.assess_quality({})
        weak = quality.assess_quality({
            "video": _signal("video", score=0.2),
            "voice": _signal("audio", score=0.2),
            "camera": _signal("image", score=0.2),
        })
        clean = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(missing)
        self._assert_schema(weak)
        self._assert_schema(clean)
        self.assertEqual(missing["failure_reason"], quality.FAILURE_REASON_MISSING_MEDIA)
        self.assertTrue(missing["retake_required"])
        self.assertEqual(missing["suggested_action"], "rescan_recommended")
        self.assertEqual(weak["failure_reason"], quality.FAILURE_REASON_LOW_QUALITY_MEDIA)
        self.assertTrue(weak["retake_required"])
        self.assertEqual(weak["suggested_action"], "rescan_recommended")
        self.assertIsNone(clean["failure_reason"])
        self.assertFalse(clean["retake_required"])
        self.assertEqual(clean["suggested_action"], "continue_normal_activity")

    def test_complete_output_schema_is_always_returned(self):
        result = quality.assess_quality({
            "video": {"score": None, "details": {"status": "failed"}},
            "voice": None,
            "camera": "bad",
        })
        self._assert_schema(result)

    def test_confidence_multiplier_floor_and_ceiling_are_enforced(self):
        floor_zero = quality.assess_quality({
            "video": _signal(
                "video",
                score=0.0,
                details_overrides={
                    "visual_warnings": [
                        "video_too_dark",
                        "video_blurry",
                        "unstable_video",
                        "unstable_camera",
                        "insufficient_usable_frames",
                        "subject_not_visible",
                        "face_not_visible",
                        "landmark_detection_failed",
                    ],
                },
            ),
            "voice": None,
            "camera": None,
        })
        floor_positive = quality.assess_quality({
            "video": _signal(
                "video",
                score=0.1,
                details_overrides={
                    "visual_warnings": [
                        "video_too_dark",
                        "video_blurry",
                        "unstable_video",
                        "unstable_camera",
                        "insufficient_usable_frames",
                        "subject_not_visible",
                        "face_not_visible",
                        "landmark_detection_failed",
                    ],
                },
            ),
            "voice": None,
            "camera": None,
        })
        normal = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(floor_zero)
        self._assert_schema(floor_positive)
        self._assert_schema(normal)
        self.assertEqual(floor_zero["confidence_multiplier"], 0.15)
        self.assertEqual(floor_positive["confidence_multiplier"], 0.15)
        self.assertGreater(normal["confidence_multiplier"], 0.15)
        self.assertLessEqual(normal["confidence_multiplier"], 1.0)

    def test_unknown_warning_is_public_but_not_decision_relevant(self):
        with_unknown = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["future_observation_code"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        clean = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(with_unknown)
        self._assert_schema(clean)
        self.assertEqual(with_unknown["status"], "passed")
        self.assertIn("future_observation_code", with_unknown["warnings"])
        self.assertNotIn("future_observation_code", with_unknown["weak_reasons"])
        self.assertFalse(with_unknown["retake_required"])
        self.assertEqual(with_unknown["confidence_multiplier"], clean["confidence_multiplier"])

    def test_weak_modalities_do_not_force_retake_without_explicit_warning(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.2),
        })
        self._assert_schema(result)
        self.assertEqual(result["status"], "weak")
        self.assertEqual(result["suggested_action"], "review_required")
        self.assertIsNone(result["failure_reason"])
        self.assertFalse(result["retake_required"])
        self.assertEqual(result["reasons"], [])

    def test_explicit_video_and_audio_retake_warnings_request_retake(self):
        cases = [
            ("video", {"visual_warnings": ["video_blurry"]}),
            ("audio", {"audio_warnings": ["audio_too_noisy"]}),
        ]
        for modality, overrides in cases:
            with self.subTest(modality=modality):
                result = quality.assess_quality({
                    "video": _signal("video", score=0.9, details_overrides=overrides if modality == "video" else None),
                    "voice": _signal("audio", score=0.9, details_overrides=overrides if modality == "audio" else None),
                    "camera": _signal("image", score=0.9),
                })
                self._assert_schema(result)
                self.assertTrue(result["retake_required"])
                self.assertEqual(result["failure_reason"], quality.FAILURE_REASON_LOW_QUALITY_MEDIA)
                self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_sustained_eye_closure_never_requests_retake(self):
        result = quality.assess_quality({
            "video": _signal("video", score=0.9, details_overrides={"visual_warnings": ["sustained_eye_closure"]}),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        })
        self._assert_schema(result)
        self.assertIn("sustained_eye_closure", result["warnings"])
        self.assertNotIn("sustained_eye_closure", result["weak_reasons"])
        self.assertFalse(result["retake_required"])
        self.assertIsNone(result["failure_reason"])

    def test_unreadable_payloads_discard_stale_evidence_and_scores(self):
        cases = [
            (
                "video",
                _signal(
                    "video",
                    score=0.99,
                    details_overrides={
                        "status": "missing",
                        "visual_warnings": ["sustained_eye_closure", "future_observation_code"],
                    },
                ),
                "video_missing",
            ),
            (
                "audio",
                _signal(
                    "audio",
                    score=0.99,
                    details_overrides={
                        "status": "failed",
                        "audio_warnings": ["audio_too_noisy", "future_observation_code"],
                    },
                ),
                "audio_missing",
            ),
            (
                "image",
                _signal(
                    "image",
                    score=0.99,
                    details_overrides={
                        "status": "processing",
                        "image_warnings": ["image_blurry", "future_observation_code"],
                    },
                ),
                "image_missing",
            ),
        ]
        for modality, payload, expected_warning in cases:
            with self.subTest(modality=modality):
                signals = {
                    "video": _signal("video", score=0.9) if modality != "video" else payload,
                    "voice": _signal("audio", score=0.9) if modality != "audio" else payload,
                    "camera": _signal("image", score=0.9) if modality != "image" else payload,
                }
                result = quality.assess_quality(signals)
                self._assert_schema(result)
                self.assertFalse(result["media_quality"][modality]["present"])
                self.assertIsNone(result["media_quality"][modality]["score"])
                self.assertEqual(result["media_quality"][modality]["warnings"], [expected_warning])
                self.assertNotIn("sustained_eye_closure", result["warnings"])
                self.assertNotIn("future_observation_code", result["warnings"])

    def test_capture_quality_scores_must_stay_in_range(self):
        cases = [
            ("video", "visual_quality_score", ("duration_seconds",), "visual_warnings"),
            ("audio", "audio_quality_score", ("duration_seconds", "duration_sec"), "audio_warnings"),
            ("image", "image_quality_score", (), "image_warnings"),
        ]
        for modality, score_key, duration_keys, warning_key in cases:
            with self.subTest(modality=modality):
                base_details = {"status": "ok", warning_key: []}
                if duration_keys:
                    base_details[duration_keys[0]] = 5.0
                valid_zero = quality.assess_quality({
                    "video": _signal("video", score=0.9),
                    "voice": _signal("audio", score=0.9),
                    "camera": _signal("image", score=0.9),
                })
                details_zero = dict(base_details)
                details_zero[score_key] = 0.0
                details_one = dict(base_details)
                details_one[score_key] = 1.0
                details_negative = dict(base_details)
                details_negative[score_key] = -0.1
                details_above_one = dict(base_details)
                details_above_one[score_key] = 1.1
                if modality == "video":
                    valid_zero = quality.assess_quality({
                        "video": {"score": 0.9, "details": details_zero},
                        "voice": _signal("audio", score=0.9),
                        "camera": _signal("image", score=0.9),
                    })
                    valid_one = quality.assess_quality({
                        "video": {"score": 0.9, "details": details_one},
                        "voice": _signal("audio", score=0.9),
                        "camera": _signal("image", score=0.9),
                    })
                    invalid_negative = quality.assess_quality({
                        "video": {"score": 0.9, "details": details_negative},
                        "voice": _signal("audio", score=0.9),
                        "camera": _signal("image", score=0.9),
                    })
                    invalid_above_one = quality.assess_quality({
                        "video": {"score": 0.9, "details": details_above_one},
                        "voice": _signal("audio", score=0.9),
                        "camera": _signal("image", score=0.9),
                    })
                elif modality == "audio":
                    valid_zero = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": {"score": 0.9, "details": details_zero},
                        "camera": _signal("image", score=0.9),
                    })
                    valid_one = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": {"score": 0.9, "details": details_one},
                        "camera": _signal("image", score=0.9),
                    })
                    invalid_negative = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": {"score": 0.9, "details": details_negative},
                        "camera": _signal("image", score=0.9),
                    })
                    invalid_above_one = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": {"score": 0.9, "details": details_above_one},
                        "camera": _signal("image", score=0.9),
                    })
                else:
                    valid_zero = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": _signal("audio", score=0.9),
                        "camera": {"score": 0.9, "details": details_zero},
                    })
                    valid_one = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": _signal("audio", score=0.9),
                        "camera": {"score": 0.9, "details": details_one},
                    })
                    invalid_negative = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": _signal("audio", score=0.9),
                        "camera": {"score": 0.9, "details": details_negative},
                    })
                    invalid_above_one = quality.assess_quality({
                        "video": _signal("video", score=0.9),
                        "voice": _signal("audio", score=0.9),
                        "camera": {"score": 0.9, "details": details_above_one},
                    })
                self._assert_schema(valid_zero)
                self._assert_schema(valid_one)
                self._assert_schema(invalid_negative)
                self._assert_schema(invalid_above_one)
                self.assertTrue(valid_zero["media_quality"][modality]["present"])
                self.assertTrue(valid_one["media_quality"][modality]["present"])
                self.assertFalse(invalid_negative["media_quality"][modality]["present"])
                self.assertFalse(invalid_above_one["media_quality"][modality]["present"])
                self.assertEqual(valid_zero["media_quality"][modality]["score"], 0.0)
                self.assertEqual(valid_one["media_quality"][modality]["score"], 1.0)

    def test_task_attempts_thresholds_and_property_failure(self):
        class RaisingTask:
            @property
            def attempts(self):
                raise RuntimeError("attempts boom")

            @property
            def reaction_time(self):
                raise RuntimeError("reaction_time boom")

            @property
            def errors(self):
                raise RuntimeError("errors boom")

        cases = [
            ({"attempts": 0, "reaction_time": 0.8, "errors": 0}, "failed"),
            ({"attempts": 1, "reaction_time": 0.8, "errors": 0}, "weak"),
            ({"attempts": 2, "reaction_time": 0.8, "errors": 0}, "weak"),
            ({"attempts": 3, "reaction_time": 0.8, "errors": 0}, "good"),
        ]
        for task, expected in cases:
            with self.subTest(task=task):
                result = quality.assess_quality({
                    "video": _signal("video", score=0.9),
                    "voice": _signal("audio", score=0.9),
                    "camera": _signal("image", score=0.9),
                }, task=task)
                self._assert_schema(result)
                self.assertEqual(result["task_quality"], expected)

        result = quality.assess_quality({
            "video": _signal("video", score=0.9),
            "voice": _signal("audio", score=0.9),
            "camera": _signal("image", score=0.9),
        }, task=RaisingTask())
        self._assert_schema(result)
        self.assertEqual(result["task_quality"], "missing")
