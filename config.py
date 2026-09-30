import os
import string
import random
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SEED = 42


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if parsed < min_value:
        raise ValueError(f"{name} must be at least {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}")
    return parsed


def _seed_random_generators(seed: int) -> None:
    # Importing config has historically made local inference and tests reproducible.
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


MODEL_VERSION = "cie_v1_2"


def server_build_sha() -> str:
    """Return deployment-provided build identity without inspecting the host."""
    for name in ("APP_BUILD_SHA", "GIT_COMMIT_SHA", "RENDER_GIT_COMMIT", "SOURCE_COMMIT"):
        value = os.getenv(name, "").strip()
        if 7 <= len(value) <= 64 and all(character in string.hexdigits for character in value):
            return value
    return "unknown"


def server_build_version() -> str:
    value = os.getenv("APP_BUILD_VERSION", "").strip()
    allowed = string.ascii_letters + string.digits + "._+-"
    if value and len(value) <= 64 and all(character in allowed for character in value):
        return value
    return "unknown"
REQUIRE_LOCAL_MODEL = _env_bool("REQUIRE_LOCAL_MODEL", False)

LABELS = ["High Risk", "Elevated Fatigue", "Low Focus", "Stable"]
# Expected readiness value per label. Probabilities are weighted by these
# values to calculate expected readiness, and the result is not model confidence.
LABEL_SCORES = [0.2, 0.45, 0.6, 0.85]
LABEL_READINESS_VALUES = LABEL_SCORES

# Legacy compatibility configuration retained for older imports.
WEIGHT_CAMERA = 0.3
WEIGHT_VIDEO = 0.4
WEIGHT_VOICE = 0.3
WEIGHT_TASK = 0.15
ML_WEIGHT = 0.7
MISSING_MEDIA_PENALTY = 0.1

READINESS_FACE_WEIGHT = 0.45
READINESS_VOICE_WEIGHT = 0.25
READINESS_REACTION_WEIGHT = 0.30
READINESS_ML_BLEND = 0.25

BASELINE_PROVISIONAL_AFTER = 2
BASELINE_ACTIVE_AFTER = 3
BASELINE_USE_AFTER = 3
BASELINE_HIGH_CONFIDENCE_AFTER = 5
BASELINE_MIN_CAPTURE_QUALITY = 0.65
BASELINE_MIN_CONFIDENCE = 0.60
BASELINE_MAX_STORED_SAMPLES = 9
BASELINE_MIN_DRIFT = 0.12
BASELINE_STD_MULTIPLIER = 1.5
BASELINE_MORNING_END_HOUR = 12
BASELINE_EVENING_START_HOUR = 17

BASELINE_ALPHA = 0.25
BASELINE_DRIFT_THRESHOLD = 0.12
BASELINE_DRIFT_PENALTY = 0.05
BASELINE_PATH = os.getenv("BASELINE_PATH", "data/baselines.json")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/latest.pt")
MAX_DOWNLOAD_BYTES = _env_int("MAX_DOWNLOAD_BYTES", 20_000_000, min_value=1)

MIN_AUDIO_DURATION_SEC = 1.5
MIN_TASK_ATTEMPTS = 3
MIN_VIDEO_FACE_FRAMES = 5
MIN_VIDEO_SAMPLED_FRAMES = 8
LOW_CONFIDENCE_THRESHOLD = 0.45

CAMERA_GOOD = 0.7
CAMERA_MED  = 0.5

VIDEO_GOOD = 0.65
VIDEO_MED  = 0.45

VOICE_GOOD = 0.6
VOICE_MED  = 0.4

TASK_RT_GOOD = 0.6
TASK_RT_MED = 1.0
TASK_ERR_GOOD = 0
TASK_ERR_MED = 2

MIN_AUDIO_ENERGY = 0.01
MIN_SWAY_STD = 0.01
MIN_EAR = 0.2
VIDEO_MAX_FRAMES = 300
VIDEO_FRAME_STRIDE = 5


def _validate_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")


def _validate_non_negative_number(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ValueError(f"{name} must be a non-negative number")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: int) -> None:
    _validate_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_config() -> None:
    if len(LABELS) != len(LABEL_SCORES):
        raise ValueError("LABELS and LABEL_SCORES must have the same length")
    if len(set(LABELS)) != len(LABELS):
        raise ValueError("LABELS must be unique")
    for index, score in enumerate(LABEL_SCORES):
        _validate_probability(f"LABEL_SCORES[{index}]", score)

    for name, value in {
        "BASELINE_PROVISIONAL_AFTER": BASELINE_PROVISIONAL_AFTER,
        "BASELINE_ACTIVE_AFTER": BASELINE_ACTIVE_AFTER,
        "BASELINE_USE_AFTER": BASELINE_USE_AFTER,
        "BASELINE_HIGH_CONFIDENCE_AFTER": BASELINE_HIGH_CONFIDENCE_AFTER,
    }.items():
        _validate_non_negative_int(name, value)
    for name, value in {
        "BASELINE_MIN_CAPTURE_QUALITY": BASELINE_MIN_CAPTURE_QUALITY,
        "BASELINE_MIN_CONFIDENCE": BASELINE_MIN_CONFIDENCE,
    }.items():
        _validate_probability(name, value)
    _validate_positive_int("BASELINE_MAX_STORED_SAMPLES", BASELINE_MAX_STORED_SAMPLES)
    if not 0 <= BASELINE_MORNING_END_HOUR <= 23:
        raise ValueError("BASELINE_MORNING_END_HOUR must be between 0 and 23")
    if not 0 <= BASELINE_EVENING_START_HOUR <= 23:
        raise ValueError("BASELINE_EVENING_START_HOUR must be between 0 and 23")
    if BASELINE_MORNING_END_HOUR >= BASELINE_EVENING_START_HOUR:
        raise ValueError("baseline time buckets require morning end before evening start")

    _validate_non_negative_number("MIN_AUDIO_DURATION_SEC", MIN_AUDIO_DURATION_SEC)

    for name, value in {
        "MIN_TASK_ATTEMPTS": MIN_TASK_ATTEMPTS,
        "MIN_VIDEO_FACE_FRAMES": MIN_VIDEO_FACE_FRAMES,
        "MIN_VIDEO_SAMPLED_FRAMES": MIN_VIDEO_SAMPLED_FRAMES,
        "VIDEO_MAX_FRAMES": VIDEO_MAX_FRAMES,
        "VIDEO_FRAME_STRIDE": VIDEO_FRAME_STRIDE,
    }.items():
        _validate_positive_int(name, value)


_seed_random_generators(SEED)
_validate_config()
