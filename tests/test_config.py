import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


CONFIG_ENV_KEYS = {
    "REQUIRE_LOCAL_MODEL",
    "MAX_DOWNLOAD_BYTES",
    "ML_MODEL_PATH",
    "BASELINE_PATH",
    "APP_BUILD_SHA",
    "GIT_COMMIT_SHA",
    "RENDER_GIT_COMMIT",
    "SOURCE_COMMIT",
    "APP_BUILD_VERSION",
}


def _reload_config():
    import config

    return importlib.reload(config)


@contextmanager
def config_env(**env):
    original = {name: os.environ.get(name) for name in CONFIG_ENV_KEYS}
    try:
        for name in CONFIG_ENV_KEYS:
            if name in env:
                value = env[name]
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            elif original[name] is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original[name]
        yield _reload_config()
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _reload_config()


class ConfigTests(unittest.TestCase):
    def test_server_build_sha_uses_configured_deployment_value(self):
        configured_sha = "a" * 40
        with config_env(APP_BUILD_SHA=configured_sha) as config:
            self.assertEqual(config.server_build_sha(), configured_sha)

    def test_server_build_sha_returns_unknown_when_unset(self):
        with config_env(
            APP_BUILD_SHA=None,
            GIT_COMMIT_SHA=None,
            RENDER_GIT_COMMIT=None,
            SOURCE_COMMIT=None,
        ) as config:
            self.assertEqual(config.server_build_sha(), "unknown")

    def test_require_local_model_accepts_true_values(self):
        for value in ["1", "true", "TRUE", "t", "yes", "y", "on", " On "]:
            with self.subTest(value=value):
                with config_env(REQUIRE_LOCAL_MODEL=value) as config:
                    self.assertTrue(config.REQUIRE_LOCAL_MODEL)

    def test_require_local_model_accepts_false_values(self):
        for value in ["0", "false", "FALSE", "f", "no", "n", "off", " Off "]:
            with self.subTest(value=value):
                with config_env(REQUIRE_LOCAL_MODEL=value) as config:
                    self.assertFalse(config.REQUIRE_LOCAL_MODEL)

    def test_require_local_model_rejects_invalid_boolean_values(self):
        with self.assertRaisesRegex(ValueError, "REQUIRE_LOCAL_MODEL must be a boolean value"):
            with config_env(REQUIRE_LOCAL_MODEL="maybe"):
                pass

    def test_max_download_bytes_rejects_invalid_numeric_values(self):
        for value in ["abc", "20.5", ""]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "MAX_DOWNLOAD_BYTES must be an integer"):
                    with config_env(MAX_DOWNLOAD_BYTES=value):
                        pass

    def test_max_download_bytes_rejects_out_of_range_values(self):
        for value in ["0", "-1"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "MAX_DOWNLOAD_BYTES must be at least 1"):
                    with config_env(MAX_DOWNLOAD_BYTES=value):
                        pass

    def test_max_download_bytes_accepts_positive_integer(self):
        with config_env(MAX_DOWNLOAD_BYTES="123456") as config:
            self.assertEqual(config.MAX_DOWNLOAD_BYTES, 123456)

    def test_baseline_thresholds_are_non_negative(self):
        with config_env() as config:
            self.assertGreaterEqual(config.BASELINE_PROVISIONAL_AFTER, 0)
            self.assertGreaterEqual(config.BASELINE_ACTIVE_AFTER, 0)
            self.assertGreaterEqual(config.BASELINE_USE_AFTER, 0)
            self.assertGreaterEqual(config.BASELINE_HIGH_CONFIDENCE_AFTER, 0)

    def test_non_negative_scan_counts_accept_zero(self):
        with config_env() as config:
            config._validate_non_negative_int("BASELINE_PROVISIONAL_AFTER", 0)
            config._validate_non_negative_int("BASELINE_ACTIVE_AFTER", 0)
            config._validate_non_negative_int("BASELINE_USE_AFTER", 0)
            config._validate_non_negative_int("BASELINE_HIGH_CONFIDENCE_AFTER", 0)
            with patch.object(config, "BASELINE_PROVISIONAL_AFTER", 0), patch.object(
                config, "BASELINE_ACTIVE_AFTER", 0
            ), patch.object(config, "BASELINE_USE_AFTER", 0), patch.object(
                config, "BASELINE_HIGH_CONFIDENCE_AFTER", 0
            ):
                config._validate_config()

    def test_non_negative_scan_counts_reject_invalid_types(self):
        with config_env() as config:
            for value in [2.5, True, "3"]:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "must be a non-negative integer"):
                        config._validate_non_negative_int("BASELINE_ACTIVE_AFTER", value)

    def test_positive_integer_constants_reject_zero(self):
        with config_env() as config:
            for name in [
                "BASELINE_MAX_STORED_SAMPLES",
                "MIN_TASK_ATTEMPTS",
                "MIN_VIDEO_FACE_FRAMES",
                "MIN_VIDEO_SAMPLED_FRAMES",
                "VIDEO_MAX_FRAMES",
                "VIDEO_FRAME_STRIDE",
            ]:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                        config._validate_positive_int(name, 0)

    def test_labels_and_scores_are_consistent(self):
        with config_env() as config:
            self.assertEqual(len(config.LABELS), len(config.LABEL_SCORES))
            self.assertEqual(len(config.LABELS), len(set(config.LABELS)))
            self.assertTrue(all(0.0 <= score <= 1.0 for score in config.LABEL_SCORES))

    def test_label_readiness_alias_is_retained(self):
        with config_env() as config:
            self.assertIs(config.LABEL_READINESS_VALUES, config.LABEL_SCORES)

    def test_legacy_compatibility_constants_are_retained(self):
        with config_env() as config:
            self.assertTrue(hasattr(config, "WEIGHT_CAMERA"))
            self.assertTrue(hasattr(config, "READINESS_FACE_WEIGHT"))
            self.assertTrue(hasattr(config, "ML_WEIGHT"))

    def test_optional_local_model_missing_path_has_no_runtime_error(self):
        expected_model_path = "models/definitely-missing-test-model.pt"
        with config_env(
            REQUIRE_LOCAL_MODEL="false",
            ML_MODEL_PATH=expected_model_path,
        ) as config:
            self.assertEqual(config.ML_MODEL_PATH, expected_model_path)
            with patch.dict(sys.modules, {"numpy": types.SimpleNamespace()}, clear=False):
                import ml.runtime

                runtime_module = importlib.reload(ml.runtime)
                runtime = runtime_module.MLRuntime()

        self.assertEqual(runtime.model_path, expected_model_path)
        self.assertFalse(runtime.local_model_required())
        self.assertFalse(runtime.load())
        self.assertIsNone(runtime.error)


if __name__ == "__main__":
    unittest.main()
