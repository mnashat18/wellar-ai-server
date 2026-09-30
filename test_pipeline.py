import unittest
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
import types
from contextlib import ExitStack
from concurrent.futures import InvalidStateError
from unittest.mock import MagicMock, patch

from baseline import (
    baseline_ready_for_personalized_scoring,
    baseline_signal_payload,
    baseline_status_payload,
    is_scan_eligible_for_baseline,
)
from directus_client import DirectusClient
from quality import assess_quality
from scoring import clamp_confidence, compute_result
from utils import sanitize_payload
import requests
import analysis_worker
import analysis_runtime
import audio
import scoring
import video
import main


def _signal(score, details):
    return {"score": score, "details": details}


def _production_multimodal_results():
    return {
        "video": _signal(
            0.7666,
            {
                "status": "ok",
                "visual_confidence": 0.7666,
                "visual_quality_score": 0.8,
                "duration_seconds": 5.0,
                "visual_warnings": [],
                "face_frames": 10,
                "sampled_frames": 12,
                "face_or_subject_visibility": 0.8,
            },
        ),
        "audio": _signal(
            0.9158,
            {
                "status": "ok",
                "audio_confidence": 0.9158,
                "audio_quality_score": 0.82,
                "duration_seconds": 4.0,
                "audio_warnings": [],
                "rms_energy": 0.03,
                "zero_crossing_rate": 0.08,
                "spectral_centroid": 1200.0,
                "silent": False,
            },
        ),
        "image": _signal(
            0.8969,
            {
                "status": "ok",
                "image_confidence": 0.8969,
                "image_quality_score": 0.83,
                "image_warnings": [],
                "face_detected": True,
                "avg_ear": 0.3,
            },
        ),
    }


def _successful_worker_states():
    return {
        modality: {
            "result_received": True,
            "timed_out": False,
            "analyzer_error": False,
            "final_alive": False,
            "process_exitcode": 0,
            "worker_generation": 1,
        }
        for modality in ("video", "audio", "image")
    }


class _FakePipeEndpoint:
    NO_RESPONSE = object()

    def __init__(self, shared_state, on_send=None):
        self._shared_state = shared_state
        self._on_send = on_send
        self.closed = False

    def send(self, payload):
        if self._on_send is not None:
            callback_result = self._on_send(payload, self._shared_state)
            if callback_result is self.NO_RESPONSE:
                return
            payload = callback_result
        self._shared_state["payload"] = payload
        self._shared_state["ready"] = True

    def poll(self, timeout=None):
        return bool(self._shared_state.get("ready"))

    def recv(self):
        self._shared_state["ready"] = False
        return self._shared_state.get("payload")

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, target, args, daemon, behavior):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.behavior = behavior or {}
        self.sentinel = object()
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = []
        self.closed = False
        self.exitcode = None
        self._alive = bool(self.behavior.get("alive_after_start", False))

    def start(self):
        self.started = True
        if self.behavior.get("invoke_target", True) and self.target is not None:
            self.target(*self.args)
            if not self.behavior.get("alive_after_target", False):
                self._alive = False
                self.exitcode = 0
        else:
            self._alive = bool(self.behavior.get("alive_after_start", True))

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        if not self.behavior.get("requires_kill", False):
            self._alive = False
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if self.terminated and self.behavior.get("requires_kill", False) and self.killed:
            self._alive = False

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, behaviors=None):
        self.behaviors = list(behaviors or [])
        self.processes = []

    def Pipe(self, duplex=False):
        shared_state = {"payload": None, "ready": False}
        return _FakePipeEndpoint(shared_state), _FakePipeEndpoint(shared_state)

    def Process(self, target, args, daemon):
        behavior = self.behaviors.pop(0) if self.behaviors else {}
        process = _FakeProcess(target, args, daemon, behavior)
        self.processes.append(process)
        return process


class _AsyncFakeProcess:
    def __init__(self, target, args, daemon, behavior=None):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.behavior = behavior or {}
        self.sentinel = object()
        self.started = False
        self.terminated = False
        self.killed = False
        self.closed = False
        self.exitcode = None
        self._alive = False
        self._thread = None

    def start(self):
        self.started = True
        self._alive = True

        def runner():
            try:
                if self.target is not None:
                    self.target(*self.args)
                if self.exitcode is None:
                    self.exitcode = 0
            finally:
                if not self.behavior.get("stay_alive_after_target", True):
                    self._alive = False

        import threading as _threading

        self._thread = _threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def is_alive(self):
        return bool(self._alive)

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def close(self):
        self.closed = True


class _AsyncFakeContext:
    def __init__(self, behaviors=None):
        self.behaviors = list(behaviors or [])
        self.processes = []
        self.parent_conns = []

    def Pipe(self, duplex=False):
        shared_state = {"payload": None, "ready": False}
        parent = _FakePipeEndpoint(shared_state)
        child = _FakePipeEndpoint(shared_state)
        self.parent_conns.append(parent)
        return parent, child

    def Process(self, target, args, daemon):
        behavior = self.behaviors.pop(0) if self.behaviors else {}
        process = _AsyncFakeProcess(target, args, daemon, behavior)
        self.processes.append(process)
        return process


class _ReadyThenEOFConn:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        raise EOFError()

    def close(self):
        self.closed = True


class _SlowStartupSupervisor:
    def __init__(self, name, delay_seconds, startup_timeout_seconds):
        self.name = name
        self.delay_seconds = delay_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self.ready = False
        self.shutdown_called = False

    def start(self, *, startup_cancel_token=None):
        if startup_cancel_token is not None and startup_cancel_token.is_cancelled():
            raise analysis_runtime.AnalysisRuntimeStartupError(f"{self.name}_worker_start_cancelled")
        time.sleep(self.delay_seconds)
        if startup_cancel_token is not None and startup_cancel_token.is_cancelled():
            raise analysis_runtime.AnalysisRuntimeStartupError(f"{self.name}_worker_start_cancelled")
        self.ready = True
        return {
            "worker_generation": 1,
            "spawn_metrics": {
                "ready_wait_ms": int(round(self.delay_seconds * 1000)),
                "analyzer_import_ms": int(round(self.delay_seconds * 1000)),
                "audio_import_ms": int(round(self.delay_seconds * 1000)) if self.name == "audio" else None,
                "audio_prewarm_first_call_ms": 9000 if self.name == "audio" else None,
                "audio_prewarm_second_call_ms": 1000 if self.name == "audio" else None,
                "audio_prewarm_total_ms": 10000 if self.name == "audio" else None,
            },
        }

    @property
    def delay_seconds(self):
        return self._delay_seconds

    @delay_seconds.setter
    def delay_seconds(self, value):
        self._delay_seconds = float(value)

    def health(self):
        return {
            "ready": self.ready,
            "worker_generation": 1 if self.ready else 0,
            "process_alive": self.ready,
        }

    def shutdown(self):
        self.shutdown_called = True
        self.ready = False


class _ProbeRLock:
    def __init__(self):
        self._lock = __import__("threading").RLock()
        self._owner = None
        self._depth = 0
        self._guard = __import__("threading").Lock()

    def acquire(self, blocking=True, timeout=-1):
        if timeout is None:
            acquired = self._lock.acquire(blocking)
        elif timeout < 0:
            acquired = self._lock.acquire(blocking)
        else:
            acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            with self._guard:
                self._owner = __import__("threading").get_ident()
                self._depth += 1
        return acquired

    def release(self):
        with self._guard:
            self._depth -= 1
            if self._depth <= 0:
                self._owner = None
                self._depth = 0
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def locked_any(self):
        with self._guard:
            return self._depth > 0

    def locked_by_current(self):
        with self._guard:
            return self._owner == __import__("threading").get_ident() and self._depth > 0


class _DelayedReadyEndpoint:
    def __init__(self, ready_event, generation, violations, probe):
        self.ready_event = ready_event
        self.generation = generation
        self.violations = violations
        self.probe = probe
        self.closed = False

    def send(self, payload):
        return None

    def poll(self, timeout=None):
        if self.probe.locked_by_current():
            self.violations.append("poll_locked")
        return self.ready_event.is_set() and not self.closed

    def recv(self):
        if self.probe.locked_by_current():
            self.violations.append("recv_locked")
        generation = self.generation() if callable(self.generation) else self.generation
        return {
            "type": "ready",
            "ok": True,
            "worker_generation": generation,
            "metrics": {
                "child_entry_ms": 1,
                "analyzer_import_ms": 15000,
                "total_worker_ms": 15000,
            },
        }

    def close(self):
        self.closed = True


class _LockCheckingProcess:
    def __init__(self, violations, probe):
        self.violations = violations
        self.probe = probe
        self.started = False
        self.terminated = False
        self.killed = False
        self.closed = False
        self.exitcode = None
        self._alive = False
        self.sentinel = self

    def start(self):
        if self.probe.locked_by_current():
            self.violations.append("start_locked")
        self.started = True
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        if self.probe.locked_by_current():
            self.violations.append("terminate_locked")
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def kill(self):
        if self.probe.locked_by_current():
            self.violations.append("kill_locked")
        self.killed = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        if self.probe.locked_by_current():
            self.violations.append("join_locked")

    def close(self):
        self.closed = True


class _DelayedReadyContext:
    def __init__(self, ready_event, violations, probe):
        self.ready_event = ready_event
        self.violations = violations
        self.probe = probe
        self.processes = []
        self.generation = None

    def Pipe(self, duplex=False):
        parent = _DelayedReadyEndpoint(self.ready_event, lambda: self.generation, self.violations, self.probe)
        child = _DelayedReadyEndpoint(self.ready_event, lambda: self.generation, self.violations, self.probe)
        return parent, child

    def Process(self, target, args, daemon):
        self.generation = args[2]
        process = _LockCheckingProcess(self.violations, self.probe)
        self.processes.append(process)
        return process


class _FakeRuntime:
    def __init__(self, run_scan_result=None, ready=True):
        self.run_scan_result = run_scan_result or (
            {
                "video": {"score": 0.9, "details": {"status": "ok", "analyzer": "video"}},
                "audio": {"score": 0.9, "details": {"status": "ok", "analyzer": "audio"}},
                "image": {"score": 0.9, "details": {"status": "ok", "analyzer": "image"}},
            },
            {
                "video": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
                "audio": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
                "image": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
            },
        )
        self.ready = ready
        self.start_calls = 0
        self.shutdown_calls = 0
        self.run_scan_calls = []

    def start(self):
        self.start_calls += 1
        return {"analyzer_runtime_start_ms": 1, "analyzer_runtime_ready_ms": 1, "workers": {}}

    def shutdown(self):
        self.shutdown_calls += 1

    def is_ready(self):
        return self.ready

    def health(self):
        return {"ready": self.ready, "workers": {}, "started": self.start_calls > 0, "stopped": self.shutdown_calls > 0}

    def run_scan(self, scan_id, media, *, deadline_seconds):
        self.run_scan_calls.append((scan_id, media, deadline_seconds))
        return self.run_scan_result


class _SubmitOnlySupervisor:
    def __init__(self, future, generation=1, completion=None):
        self._future = future
        self._generation = generation
        self._completion = completion or {
            "result": {
                "score": None,
                "details": {"status": "load_failed", "audio_warnings": ["audio_timeout"]},
            },
            "state": {
                "timed_out": True,
                "analyzer_error": False,
                "final_alive": False,
                "result_received": False,
                "worker_restarted": True,
                "worker_generation": generation,
                "process_exitcode": -15,
                "terminated": True,
                "killed": False,
            },
        }

    def submit(self, *, scan_id, path, deadline_at):
        setattr(self._future, "_analysis_job_id", f"{scan_id}-{id(self)}")
        return self._future

    def finalize_timed_out_job(self, job_id):
        if not self._future.done():
            self._future.set_result(self._completion)
        return self._completion

    def forget_job(self, job_id):
        return None

    def health(self):
        return {
            "ready": True,
            "worker_generation": self._generation,
            "process_alive": True,
        }


class _FakeAudioLibrosa:
    @staticmethod
    def _stft(y, n_fft, hop_length, win_length=None, center=True, pad_mode="constant"):
        win_length = win_length or n_fft
        samples = audio.np.asarray(y, dtype=audio.np.float32).reshape(-1)
        pad = n_fft // 2 if center else 0
        padded = audio.np.pad(samples, (pad, pad), mode=pad_mode)
        if padded.size < n_fft:
            padded = audio.np.pad(padded, (0, n_fft - padded.size), mode=pad_mode)
        frames = audio.np.ascontiguousarray(audio.np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length])
        if frames.size == 0:
            frames = padded[-n_fft:][audio.np.newaxis, :]
        window = audio.np.hanning(win_length).astype(audio.np.float32, copy=False)
        if win_length != n_fft:
            window = audio.np.pad(window, (0, n_fft - win_length), mode="constant")
        return audio.np.fft.rfft(audio.np.ascontiguousarray(frames * window), axis=-1).T

    @staticmethod
    def stft(y, n_fft, hop_length, win_length=None, window="hann", center=True, pad_mode="constant"):
        return _FakeAudioLibrosa._stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, pad_mode=pad_mode)

    class filters:
        @staticmethod
        def mel(*, sr, n_fft, n_mels):
            bins = n_fft // 2 + 1
            basis = audio.np.zeros((n_mels, bins), dtype=audio.np.float64)
            band_edges = audio.np.linspace(0, bins - 1, n_mels + 2)
            for row in range(n_mels):
                left = int(round(band_edges[row]))
                center = int(round(band_edges[row + 1]))
                right = int(round(band_edges[row + 2]))
                center = max(center, left + 1)
                right = max(right, center + 1)
                for col in range(left, min(center + 1, bins)):
                    basis[row, col] = (col - left) / max(center - left, 1)
                for col in range(center, min(right + 1, bins)):
                    basis[row, col] = max(basis[row, col], (right - col) / max(right - center, 1))
            return basis

    class feature:
        @staticmethod
        def spectral_centroid(*, y=None, sr=None, hop_length=512, n_fft=2048, S=None):
            if S is None:
                S = _FakeAudioLibrosa._stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude = audio.np.abs(audio.np.asarray(S))
            magnitude = audio.np.asarray(magnitude, dtype=audio.np.float64)
            freqs = audio.np.fft.rfftfreq(2 * (magnitude.shape[0] - 1), d=1.0 / float(sr))
            centroid = audio.np.sum(magnitude * freqs[:, audio.np.newaxis], axis=0) / audio.np.maximum(audio.np.sum(magnitude, axis=0), 1e-10)
            return centroid[audio.np.newaxis, :]

        @staticmethod
        def spectral_flatness(*, y=None, hop_length=512, n_fft=2048, S=None, power=2.0):
            if S is None:
                S = _FakeAudioLibrosa._stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude = audio.np.abs(audio.np.asarray(S))
            magnitude = audio.np.asarray(magnitude, dtype=audio.np.float64) ** power
            flatness = audio.np.exp(audio.np.mean(audio.np.log(audio.np.maximum(magnitude, 1e-12)), axis=0)) / audio.np.maximum(audio.np.mean(magnitude, axis=0), 1e-12)
            return flatness[audio.np.newaxis, :]

        @staticmethod
        def melspectrogram(*, y=None, sr=None, S=None, n_mels=128, n_fft=2048, hop_length=512):
            if S is None:
                magnitude = audio.np.abs(_FakeAudioLibrosa._stft(y, n_fft=n_fft, hop_length=hop_length))
                S = magnitude * magnitude
            power = audio.np.asarray(S, dtype=audio.np.float64)
            mel_basis = _FakeAudioLibrosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
            return mel_basis @ power

        @staticmethod
        def mfcc(*, y=None, sr=None, S=None, n_mfcc=5, hop_length=512, n_fft=2048):
            if S is None:
                mel_power = _FakeAudioLibrosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
                S = _FakeAudioLibrosa.power_to_db(mel_power, ref=audio.np.max)
            matrix = audio.np.asarray(S, dtype=audio.np.float64)
            n_mels = matrix.shape[0]
            mel_indices = audio.np.arange(n_mels, dtype=audio.np.float64) + 0.5
            basis = audio.np.cos(audio.np.pi / n_mels * audio.np.arange(n_mfcc, dtype=audio.np.float64)[:, audio.np.newaxis] * mel_indices[audio.np.newaxis, :])
            coeffs = audio.np.sqrt(2.0 / n_mels) * (basis @ matrix)
            coeffs[0] /= audio.np.sqrt(2.0)
            return coeffs

    @staticmethod
    def power_to_db(power, ref=audio.np.max):
        matrix = audio.np.asarray(power, dtype=audio.np.float64)
        reference = ref(matrix) if callable(ref) else ref
        reference = max(float(reference), 1e-10)
        return 10.0 * audio.np.log10(audio.np.maximum(matrix, 1e-10) / reference)


class PipelineTests(unittest.TestCase):
    def test_health_reports_build_and_model_metadata(self):
        with patch.dict(os.environ, {"APP_BUILD_SHA": "a" * 40, "APP_BUILD_VERSION": "test-build"}):
            payload = main.health()

        self.assertEqual(payload["server_build_sha"], "a" * 40)
        self.assertEqual(payload["server_build_version"], "test-build")
        self.assertEqual(payload["ai_model_version"], main.MODEL_VERSION)

    def test_quality_decision_diagnostics_are_not_added_to_persisted_payload(self):
        quality_result = {
            "media_quality": {
                "video": {
                    "present": True,
                    "usable": True,
                    "warnings": ["video_blurry"],
                    "decision_warnings": ["video_blurry"],
                    "evidence_blocking_warnings": [],
                }
            }
        }

        persisted = main._quality_for_persistence(quality_result)

        self.assertEqual(
            persisted["media_quality"]["video"],
            {"present": True, "usable": True, "warnings": ["video_blurry"]},
        )
        self.assertIn("decision_warnings", quality_result["media_quality"]["video"])

    def test_two_qualified_scans_remain_inactive_baseline(self):
        baseline = None
        signals = {
            "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31, "left_right_eye_asymmetry": 0.02}),
            "video": _signal(0.82, {"status": "ok"}),
            "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
        }
        for _ in range(2):
            baseline = baseline_signal_payload(baseline, signals=signals, scanned_at="2026-07-01T08:00:00Z")

        status = baseline_status_payload(baseline)
        self.assertEqual(baseline["scan_count"], 2)
        self.assertFalse(baseline["is_active"])
        self.assertFalse(status["is_active"])
        self.assertTrue(status["is_provisional"])

    def test_three_qualified_scans_create_active_baseline(self):
        baseline = None
        signals = {
            "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31, "left_right_eye_asymmetry": 0.02}),
            "video": _signal(0.82, {"status": "ok"}),
            "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
        }
        for _ in range(3):
            baseline = baseline_signal_payload(baseline, signals=signals, scanned_at="2026-07-01T08:00:00Z")

        status = baseline_status_payload(baseline)
        self.assertEqual(baseline["scan_count"], 3)
        self.assertTrue(baseline["is_active"])
        self.assertTrue(status["is_active"])
        self.assertFalse(status["is_provisional"])

    def test_baseline_uses_robust_median_and_mad(self):
        baseline = None
        values = [0.1, 0.1, 0.1, 0.9]
        for value in values:
            baseline = baseline_signal_payload(
                baseline,
                signals={
                    "camera": _signal(value, {"status": "ok", "avg_ear": value, "left_right_eye_asymmetry": 0.01}),
                    "video": _signal(0.82, {"status": "ok"}),
                    "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
                },
                scanned_at="2026-07-01T08:00:00Z",
            )

        face_stats = baseline["face_avg"]["feature_stats"]["open_eye_aperture"]
        self.assertEqual(face_stats["median"], 0.1)
        self.assertGreaterEqual(face_stats["mad"], 0.02)
        self.assertEqual(face_stats["count"], 4)

    def test_baseline_sample_history_is_bounded(self):
        baseline = None
        for index in range(12):
            baseline = baseline_signal_payload(
                baseline,
                signals={
                    "camera": _signal(0.2 + (index * 0.01), {"status": "ok", "avg_ear": 0.2 + (index * 0.01), "left_right_eye_asymmetry": 0.01}),
                    "video": _signal(0.82, {"status": "ok"}),
                    "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03 + (index * 0.001)}),
                },
                scanned_at="2026-07-01T08:00:00Z",
            )

        self.assertLessEqual(len(baseline["face_avg"]["feature_samples"]["open_eye_aperture"]), 9)
        self.assertLessEqual(len(baseline["voice_avg"]["feature_samples"]["normalized_voice_energy"]), 9)

    def test_baseline_signal_payload_uses_only_existing_fields(self):
        payload = baseline_signal_payload(
            None,
            signals={
                "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.3, "left_right_eye_asymmetry": 0.01}),
                "video": _signal(0.82, {"status": "ok"}),
                "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03, "speech_rate": 1.8}),
            },
            scanned_at="2026-07-01T08:00:00Z",
        )

        self.assertEqual(set(payload.keys()), {"scan_count", "face_avg", "voice_avg", "reaction_avg", "is_active"})

    def test_low_quality_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "weak", "weak": True, "retake_required": False, "media_quality": {"aggregate_quality": 0.2}, "confidence_multiplier": 0.3},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_missing_required_speech_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": ["speech_not_detected"]},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {}}},
            expected_phrase="please say continuity ready",
        )
        self.assertFalse(eligible)

    def test_elevated_fatigue_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "elevated_fatigue", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_high_risk_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "high_risk", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_retake_required_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": True, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": True},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_existing_legacy_baseline_json_remains_readable(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        legacy_baseline = {
            "scan_count": 6,
            "is_active": True,
            "face_avg": {"avg": 0.8, "std": 0.04},
            "voice_avg": {"avg": 0.78, "std": 0.05},
            "reaction_avg": {"avg": 0.9, "std": 0.02},
        }
        result = compute_result(signals=signals, quality=quality, baseline=legacy_baseline, baseline_used=True)
        self.assertTrue(result["baseline_used"])
        self.assertIn("baseline_drifts", result["face_metrics"])
        self.assertIn("open_eye_aperture", result["face_metrics"]["baseline_drifts"])
        self.assertIn("normalized_voice_energy", result["voice_metrics"]["baseline_drifts"])

    def test_static_eyes_closed_is_not_sustained_eye_closure(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"passed": True, "critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"status": "ok", "avg_ear": 0.22, "left_right_eye_asymmetry": 0.01, "eyes_closed": True}}, "voice": {"details": {"status": "ok", "rms_energy": 0.03}}},
        )
        self.assertTrue(eligible)

    def test_four_closed_frames_adjacent_at_30_fps_do_not_sustain(self):
        observations = [
            {"window_id": 0, "frame_index": 10, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 11, "timestamp": 0.03, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 12, "timestamp": 0.07, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 13, "timestamp": 0.10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 0)
        self.assertEqual(window_ms, 0)
        self.assertEqual(window_seconds, 0.0)

    def test_four_closed_samples_spanning_real_time_can_sustain(self):
        observations = [
            {"window_id": 0, "frame_index": 20, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 25, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 30, "timestamp": 0.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 36, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 4)
        self.assertGreaterEqual(window_seconds, 0.45)
        self.assertLessEqual(window_seconds, 1.2)
        self.assertGreater(window_ms, 0)

    def test_isolated_closed_samples_from_different_bursts_do_not_combine(self):
        observations = [
            {"window_id": 0, "frame_index": 2, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 4, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": False},
            {"window_id": 0, "frame_index": 6, "timestamp": 0.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 8, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 40, "timestamp": 1.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 44, "timestamp": 1.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": False},
            {"window_id": 1, "frame_index": 48, "timestamp": 1.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 52, "timestamp": 1.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 0)
        self.assertEqual(window_ms, 0)
        self.assertEqual(window_seconds, 0.0)

    def test_invalid_sample_breaks_the_sequence(self):
        break_cases = [
            {"eye_closed": False},
            {"sharp_enough": False},
            {"bright_enough": False},
            {"face_visible": False},
            {"landmark_valid": False},
            {"usable": False},
        ]
        for case in break_cases:
            with self.subTest(case=case):
                observations = [
                    {"window_id": 0, "frame_index": 20, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                    {"window_id": 0, "frame_index": 25, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                    {"window_id": 0, "frame_index": 30, "timestamp": 0.36, "usable": case.get("usable", True), "bright_enough": case.get("bright_enough", True), "sharp_enough": case.get("sharp_enough", True), "face_visible": case.get("face_visible", True), "landmark_valid": case.get("landmark_valid", True), "eye_closed": case.get("eye_closed", True)},
                    {"window_id": 0, "frame_index": 36, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                ]
                longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

                self.assertEqual(longest, 0)
                self.assertEqual(window_ms, 0)
                self.assertEqual(window_seconds, 0.0)

    def test_eye_closure_sampling_windows_are_bounded_and_spread_out(self):
        windows = video._eye_closure_sample_windows(frame_count=120, fps=30.0, duration_seconds=4.0)

        self.assertLessEqual(sum(len(window) for window in windows), 8)
        self.assertLessEqual(len(windows), 2)
        for window in windows:
            self.assertLessEqual(len(window), 4)
            self.assertEqual(len({sample["frame_index"] for sample in window}), 4)
            timestamps = [sample["timestamp"] for sample in window]
            self.assertGreaterEqual(timestamps[-1] - timestamps[0], 0.45)
            self.assertLessEqual(timestamps[-1] - timestamps[0], 1.2)

    def test_eye_closure_sampling_rejects_short_videos(self):
        windows = video._eye_closure_sample_windows(frame_count=8, fps=30.0, duration_seconds=0.2)
        self.assertEqual(windows, [])

    def test_naturally_narrow_eye_user_can_create_baseline_without_temporal_gate(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"passed": True, "critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"status": "ok", "avg_ear": 0.16, "left_right_eye_asymmetry": 0.01}}, "voice": {"details": {"status": "ok", "rms_energy": 0.03}}},
        )
        self.assertTrue(eligible)

    def test_baseline_drift_compares_matching_raw_metrics_only(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "avg_ear": 0.26, "left_right_eye_asymmetry": 0.03, "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "rms_energy": 0.024, "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        baseline = {
            "scan_count": 6,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.30, "mad": 0.01, "count": 6},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.005, "count": 6},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.002, "count": 6},
                    "speech_rate": {"median": 2.1, "mad": 0.2, "count": 6},
                },
            },
        }
        result = compute_result(signals=signals, quality=quality, baseline=baseline, baseline_used=True)
        self.assertIsNotNone(result["face_metrics"]["baseline_drifts"]["open_eye_aperture"])
        self.assertIsNotNone(result["face_metrics"]["baseline_drifts"]["left_right_eye_asymmetry"])
        self.assertIsNotNone(result["voice_metrics"]["baseline_drifts"]["normalized_voice_energy"])
        self.assertIsNone(result["voice_metrics"]["baseline_drifts"]["speech_rate"])

    def test_inactive_baseline_does_not_influence_scoring(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": [], "avg_ear": 0.18, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": [], "rms_energy": 0.014, "speech_rate": 1.1}),
        }
        quality = assess_quality(signals)
        inactive_baseline = {
            "scan_count": 4,
            "is_active": False,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        without_baseline = compute_result(signals=signals, quality=quality, baseline=None, baseline_used=False)
        ignored_baseline = compute_result(signals=signals, quality=quality, baseline=inactive_baseline, baseline_used=False)

        self.assertEqual(ignored_baseline["risk_level"], without_baseline["risk_level"])
        self.assertEqual(ignored_baseline["confidence"], without_baseline["confidence"])
        self.assertFalse(ignored_baseline["baseline_used"])

    def test_active_baseline_can_relax_static_false_penalties(self):
        signals = {
            "camera": _signal(0.78, {"status": "ok", "image_confidence": 0.78, "image_quality_score": 0.75, "image_warnings": [], "avg_ear": 0.18, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.8, {"status": "ok", "visual_confidence": 0.8, "visual_quality_score": 0.78, "visual_warnings": []}),
            "voice": _signal(0.77, {"status": "ok", "audio_confidence": 0.77, "audio_quality_score": 0.75, "audio_warnings": [], "rms_energy": 0.014, "speech_rate": 1.1}),
        }
        quality = assess_quality(signals)
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.02, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4},
                    "speech_rate": {"median": 1.1, "mad": 0.2, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        baseline_free = compute_result(signals=signals, quality=quality, baseline=None, baseline_used=False)
        personalized = compute_result(signals=signals, quality=quality, baseline=active_baseline, baseline_used=True)

        self.assertGreater(personalized["confidence"], baseline_free["confidence"])
        self.assertIn("baseline", personalized["explanation"].lower())
        self.assertIn("normal range", personalized["explanation"].lower())
        self.assertFalse(personalized["risk_level"] == "high_risk")

    def test_personalized_gate_requires_active_valid_baseline(self):
        quality_result = {"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": []}
        validation_result = {"passed": True, "critical_errors": [], "warnings": []}
        result = {"confidence": 0.8, "risk_level": "stable", "retake_required": False}
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        inactive_baseline = dict(active_baseline, scan_count=2, is_active=False)
        malformed_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": None, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        self.assertTrue(
            baseline_ready_for_personalized_scoring(
                active_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )
        self.assertFalse(
            baseline_ready_for_personalized_scoring(
                inactive_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )
        # A malformed face reference does not cancel a valid supported voice
        # reference. Personalization requires at least one valid supported reference.
        self.assertTrue(
            baseline_ready_for_personalized_scoring(
                malformed_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )

        no_valid_reference_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": None, "mad": 0.02, "count": 4}
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": None, "mad": 0.02, "count": 4}
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        self.assertFalse(
            baseline_ready_for_personalized_scoring(
                no_valid_reference_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )

    def test_personal_deviation_alone_cannot_become_high_risk(self):
        signals = {
            "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.8, "image_warnings": [], "avg_ear": 0.17, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.84, {"status": "ok", "visual_confidence": 0.84, "visual_quality_score": 0.82, "visual_warnings": []}),
            "voice": _signal(0.81, {"status": "ok", "audio_confidence": 0.81, "audio_quality_score": 0.8, "audio_warnings": [], "rms_energy": 0.03, "speech_rate": 2.0}),
        }
        quality = assess_quality(signals)
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.32, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.02, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 4},
                    "speech_rate": {"median": 2.0, "mad": 0.2, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        result = compute_result(signals=signals, quality=quality, baseline=active_baseline, baseline_used=True)

        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertIn("open_eye_aperture", result["fusion_details"]["baseline_flags"])

    def test_confidence_clamping(self):
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(-0.2), 0.0)
        self.assertIsNone(clamp_confidence(None))

    def test_risk_level_contract_is_exact(self):
        expected = {"stable", "low_focus", "elevated_fatigue", "high_risk"}
        self.assertEqual(scoring.VALID_RISK_LEVELS, expected)
        self.assertEqual(set(main.SCAN_RESULT_CHOICE_ALIASES["risk_level"].keys()), expected)

    def test_missing_media_handling(self):
        result = assess_quality({"camera": {}, "video": {}, "voice": {}})
        self.assertTrue(result["passed"])
        self.assertTrue(result["weak"])
        self.assertEqual(result["failure_reason"], "missing_media")
        self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_low_quality_media_warning(self):
        signals = {
            "camera": _signal(
                0.2,
                {
                    "status": "ok",
                    "image_quality_score": 0.2,
                    "image_warnings": ["image_blurry"],
                },
            ),
            "video": _signal(
                0.25,
                {
                    "status": "ok",
                    "duration_seconds": 5.0,
                    "visual_quality_score": 0.25,
                    "visual_warnings": ["video_blurry"],
                },
            ),
            "voice": _signal(
                0.22,
                {
                    "status": "ok",
                    "duration_seconds": 4.0,
                    "audio_quality_score": 0.22,
                    "audio_warnings": ["audio_too_noisy"],
                },
            ),
        }
        result = assess_quality(signals)
        self.assertTrue(result["passed"])
        self.assertTrue(result["weak"])
        self.assertEqual(result["failure_reason"], "low_quality_media")

    def test_no_undefined_values_in_payload(self):
        payload = sanitize_payload(
            {
                "ok": True,
                "text": "undefined",
                "nested": {"value": float("nan"), "good": 1.234567},
            }
        )
        self.assertNotIn("text", payload)
        self.assertNotIn("value", payload["nested"])
        self.assertEqual(payload["nested"]["good"], 1.234567)

    def test_fusion_with_only_audio(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(0.74, {"status": "ok", "audio_confidence": 0.74, "audio_quality_score": 0.7, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertLessEqual(result["confidence"], 0.55)

    def test_fusion_with_only_video(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(0.81, {"status": "ok", "visual_confidence": 0.81, "visual_quality_score": 0.79, "visual_warnings": []}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertLessEqual(result["confidence"], 0.55)

    def test_fusion_with_all_modalities(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertGreater(result["confidence"], 0.5)
        self.assertIn(result["risk_level"], {"stable", "low_focus", "elevated_fatigue", "high_risk"})

    def test_low_media_quality_reduces_confidence_and_score(self):
        good_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        weak_signals = {
            "camera": _signal(0.22, {"status": "ok", "image_confidence": 0.22, "image_quality_score": 0.2, "image_warnings": ["image_blurry"]}),
            "video": _signal(0.25, {"status": "ok", "visual_confidence": 0.25, "visual_quality_score": 0.2, "visual_warnings": ["video_blurry"]}),
            "voice": _signal(0.24, {"status": "ok", "audio_confidence": 0.24, "audio_quality_score": 0.2, "audio_warnings": ["audio_too_noisy"]}),
        }
        good = compute_result(signals=good_signals, quality=assess_quality(good_signals))
        weak = compute_result(signals=weak_signals, quality=assess_quality(weak_signals))

        self.assertLess(weak["confidence"], good["confidence"])
        self.assertLess(weak["readiness_score"], good["readiness_score"])
        self.assertIn("reduced", weak["explanation"].lower())

    def test_blurry_video_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        blurry_signals = {
            "camera": clean_signals["camera"],
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": ["video_blurry"]}),
            "voice": clean_signals["voice"],
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        blurry = compute_result(signals=blurry_signals, quality=assess_quality(blurry_signals))

        self.assertLess(blurry["confidence"], clean["confidence"])
        self.assertLess(blurry["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertIn("blur", blurry["explanation"].lower())

    def test_noisy_audio_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        noisy_signals = {
            "camera": clean_signals["camera"],
            "video": clean_signals["video"],
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": ["audio_too_noisy"]}),
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        noisy = compute_result(signals=noisy_signals, quality=assess_quality(noisy_signals))

        self.assertLess(noisy["confidence"], clean["confidence"])
        self.assertLess(noisy["modality_scores"]["audio"], clean["modality_scores"]["audio"])
        self.assertIn("noise", noisy["explanation"].lower())

    def test_weak_face_visibility_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        weak_face_quality = assess_quality(clean_signals)
        weak_face_quality["warnings"] = ["face_not_visible"]
        weak_face_quality["weak"] = True
        weak_face_quality["status"] = "weak"

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        weak_face = compute_result(signals=clean_signals, quality=weak_face_quality)

        self.assertLess(weak_face["confidence"], clean["confidence"])
        self.assertLess(weak_face["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertIn("face visibility", weak_face["explanation"].lower())

    def test_single_eye_blink_does_not_reduce_valid_scan(self):
        clean_signals = {
            "camera": _signal(0.8, {"status": "ok", "image_confidence": 0.8, "image_quality_score": 0.8, "image_warnings": []}),
            "video": _signal(
                0.84,
                {
                    "status": "ok",
                    "visual_confidence": 0.84,
                    "visual_quality_score": 0.82,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                },
            ),
            "voice": _signal(0.8, {"status": "ok", "audio_confidence": 0.8, "audio_quality_score": 0.79, "audio_warnings": []}),
        }
        blink_signals = {
            **clean_signals,
            "video": _signal(
                0.84,
                {
                    "status": "ok",
                    "visual_confidence": 0.84,
                    "visual_quality_score": 0.82,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                },
            ),
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        blink = compute_result(signals=blink_signals, quality=assess_quality(blink_signals))

        self.assertEqual(blink["risk_level"], clean["risk_level"])
        self.assertGreaterEqual(blink["confidence"], 0.45)

    def test_sustained_eye_closure_with_reliable_evidence_cannot_return_stable(self):
        signals = {
            "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.82, "image_warnings": []}),
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "visual_confidence": 0.8,
                    "visual_quality_score": 0.8,
                    "visual_warnings": ["sustained_eye_closure"],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                    "motion_stability_score": 0.88,
                    "eye_closure_sample_count": 8,
                    "closed_eye_ratio": 0.75,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_ms": 720.0,
                    "eye_closure_window_seconds": 0.72,
                    "avg_eye_aperture": 0.12,
                    "eye_aperture_std": 0.025,
                },
            ),
            "voice": _signal(0.8, {"status": "ok", "audio_confidence": 0.8, "audio_quality_score": 0.8, "audio_warnings": []}),
        }
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.22, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        result = compute_result(signals=signals, quality=assess_quality(signals), baseline=baseline, baseline_used=True)

        self.assertNotEqual(result["risk_level"], "stable")
        self.assertIn("eye closure", result["explanation"].lower())

    def test_video_and_audio_fatigue_signals_raise_fatigue_level(self):
        signals = {
            "camera": _signal(0.84, {"status": "ok", "image_confidence": 0.84, "image_quality_score": 0.84, "image_warnings": [], "face_detected": True}),
            "video": _signal(
                0.78,
                {
                    "status": "ok",
                    "visual_confidence": 0.78,
                    "visual_quality_score": 0.78,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                    "closed_eye_ratio": 0.58,
                    "avg_eye_aperture": 0.16,
                    "longest_eye_closure_streak": 5,
                    "eye_closure_window_seconds": 1.0,
                    "motion_stability_score": 0.74,
                    "eye_closure_sample_count": 12,
                },
            ),
            "voice": _signal(
                0.72,
                {
                    "status": "ok",
                    "audio_confidence": 0.72,
                    "audio_quality_score": 0.71,
                    "audio_warnings": [],
                    "speech_presence_score": 0.46,
                    "rms_energy": 0.011,
                    "silence_ratio": 0.61,
                    "speech_rate": None,
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
            ),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertIn(result["risk_level"], {"elevated_fatigue", "high_risk"})
        self.assertGreaterEqual(result["observed_fatigue_score"], 55)
        self.assertGreaterEqual(result["fatigue_evidence_score"], 0.45)
        self.assertIn("fatigue", result["explanation"].lower())

    def test_baseline_strengthens_fatigue_detection(self):
        signals = {
            "camera": _signal(0.84, {"status": "ok", "image_confidence": 0.84, "image_quality_score": 0.84, "image_warnings": [], "face_detected": True}),
            "video": _signal(
                0.78,
                {
                    "status": "ok",
                    "visual_confidence": 0.78,
                    "visual_quality_score": 0.78,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                    "closed_eye_ratio": 0.52,
                    "avg_eye_aperture": 0.17,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_seconds": 0.9,
                    "motion_stability_score": 0.76,
                    "eye_closure_sample_count": 10,
                },
            ),
            "voice": _signal(
                0.72,
                {
                    "status": "ok",
                    "audio_confidence": 0.72,
                    "audio_quality_score": 0.71,
                    "audio_warnings": [],
                    "speech_presence_score": 0.5,
                    "rms_energy": 0.012,
                    "silence_ratio": 0.58,
                    "speech_rate": None,
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
            ),
        }
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.27, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.02, "mad": 0.01, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.01, "count": 4},
                    "speech_rate": {"median": 1.9, "mad": 0.1, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        without_baseline = compute_result(signals=signals, quality=assess_quality(signals))
        with_baseline = compute_result(signals=signals, quality=assess_quality(signals), baseline=baseline, baseline_used=True)

        self.assertGreater(with_baseline["fatigue_evidence_score"], without_baseline["fatigue_evidence_score"])
        self.assertIn(with_baseline["risk_level"], {"elevated_fatigue", "high_risk"})
        self.assertNotEqual(with_baseline["risk_level"], "stable")

    def test_hard_safety_signals_are_not_rescued_by_baseline(self):
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        cases = [
            {"warnings": ["video_too_dark"], "failure_reason": None, "label": "dark"},
            {"warnings": ["video_blurry"], "failure_reason": None, "label": "blur"},
            {"warnings": ["face_not_visible"], "failure_reason": None, "label": "face"},
            {"warnings": ["audio_missing"], "failure_reason": "missing_media", "label": "missing_media"},
            {"warnings": ["audio_too_noisy"], "failure_reason": None, "label": "audio"},
            {"warnings": ["speech_not_detected"], "failure_reason": None, "label": "speech"},
        ]
        for case in cases:
            with self.subTest(case=case["label"]):
                signals = {
                    "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.82, "image_warnings": []}),
                    "video": _signal(
                        0.8,
                        {
                            "status": "ok",
                            "visual_confidence": 0.8,
                            "visual_quality_score": 0.8,
                            "visual_warnings": case["warnings"],
                        },
                    ),
                    "voice": _signal(
                        0.8,
                        {
                            "status": "ok",
                            "audio_confidence": 0.8,
                            "audio_quality_score": 0.8,
                            "audio_warnings": case["warnings"],
                        },
                    ),
                }
                quality = assess_quality(signals)
                quality["status"] = "failed"
                quality["weak"] = True
                quality["failure_reason"] = case["failure_reason"] or "low_quality_media"
                result = compute_result(signals=signals, quality=quality, baseline=baseline, baseline_used=True)
                self.assertNotEqual(result["risk_level"], "stable")
                self.assertNotEqual(result["suggested_action"], "continue_normal_activity")

    def test_eye_closure_only_caps_at_elevated_fatigue_before_baseline(self):
        self.assertEqual(
            scoring._risk_level(
                30,
                0.72,
                ["open_eye_aperture"],
                {"status": "passed", "warnings": ["sustained_eye_closure"], "weak": False},
            ),
            "elevated_fatigue",
        )

    def test_non_eye_high_risk_behavior_is_still_available(self):
        self.assertEqual(
            scoring._risk_level(
                30,
                0.72,
                [],
                {"status": "passed", "warnings": [], "weak": False},
            ),
            "high_risk",
        )

    def test_low_confidence_does_not_return_stable(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertLess(result["confidence"], 0.45)
        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertLess(result["readiness_score"], 52)

    def test_missing_video_with_good_audio_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("video", result["explanation"].lower())
        self.assertNotEqual(result["risk_level"], "high_risk")

    def test_missing_audio_with_good_video_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("audio", result["explanation"].lower())

    def test_missing_major_media_is_unknown_without_quality_metadata(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "open_failed", "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality={})

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("video", result["explanation"].lower())

    def test_low_media_quality_creates_explanation_warnings(self):
        signals = {
            "camera": _signal(0.3, {"status": "ok", "image_confidence": 0.3, "image_quality_score": 0.3, "image_warnings": ["image_blurry"]}),
            "video": _signal(0.3, {"status": "ok", "visual_confidence": 0.3, "visual_quality_score": 0.3, "visual_warnings": ["video_too_dark"]}),
            "voice": _signal(0.3, {"status": "ok", "audio_confidence": 0.3, "audio_quality_score": 0.3, "audio_warnings": ["audio_too_quiet"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        explanation = result["explanation"].lower()
        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("reduced", explanation)
        self.assertTrue("lighting" in explanation or "volume" in explanation or "blur" in explanation)

    def test_quiet_but_usable_speech_is_not_automatically_treated_as_fatigue(self):
        signals = {
            "camera": _signal(0.78, {"status": "ok", "image_confidence": 0.78, "image_quality_score": 0.77, "image_warnings": []}),
            "video": _signal(0.81, {"status": "ok", "visual_confidence": 0.81, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(
                0.74,
                {
                    "status": "ok",
                    "audio_confidence": 0.74,
                    "audio_quality_score": 0.62,
                    "audio_warnings": [],
                    "speech_presence_score": 0.64,
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
            ),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertIn(result["risk_level"], {"stable", "low_focus"})
        self.assertNotEqual(result["risk_level"], "high_risk")

    def test_poor_audio_quality_alone_cannot_become_high_risk(self):
        signals = {
            "camera": _signal(0.86, {"status": "ok", "image_confidence": 0.86, "image_quality_score": 0.85, "image_warnings": []}),
            "video": _signal(0.88, {"status": "ok", "visual_confidence": 0.88, "visual_quality_score": 0.87, "visual_warnings": []}),
            "voice": _signal(0.22, {"status": "ok", "audio_confidence": 0.22, "audio_quality_score": 0.2, "audio_warnings": ["audio_too_noisy", "speech_not_detected"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_invalid_capture_quality_alone_does_not_become_high_risk(self):
        signals = {
            "camera": _signal(0.95, {"status": "ok", "image_confidence": 0.95, "image_quality_score": 0.95, "image_warnings": []}),
            "video": _signal(0.95, {"status": "ok", "visual_confidence": 0.95, "visual_quality_score": 0.95, "visual_warnings": []}),
            "voice": _signal(0.95, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertLess(result["confidence"], 0.45)

    def test_task_score_requires_exact_positive_integer_attempts(self):
        invalid_attempts = [None, 0, -1, 3.0, "3", True]
        for attempts in invalid_attempts:
            with self.subTest(attempts=attempts):
                self.assertIsNone(
                    scoring.compute_task_score(
                        {
                            "reaction_time": 0.5,
                            "errors": 0,
                            "attempts": attempts,
                        }
                    )
                )

        self.assertIsNotNone(
            scoring.compute_task_score(
                {"reaction_time": 0.5, "errors": 0, "attempts": 3}
            )
        )

    def test_bool_only_eye_closure_is_not_confirmed(self):
        signals = {
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                },
            )
        }
        self.assertFalse(scoring._confirmed_sustained_eye_closure(signals))

    def test_complete_consistent_eye_closure_is_confirmed(self):
        signals = {
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                    "motion_stability_score": 0.9,
                    "eye_closure_sample_count": 8,
                    "closed_eye_ratio": 0.75,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_ms": 720.0,
                    "eye_closure_window_seconds": 0.72,
                    "avg_eye_aperture": 0.12,
                    "eye_aperture_std": 0.02,
                },
            )
        }
        self.assertTrue(scoring._confirmed_sustained_eye_closure(signals))

    def test_audio_usable_speech_requires_consistent_state_and_flags(self):
        self.assertTrue(
            scoring._audio_usable_speech(
                {
                    "speech_state": "usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                [],
            )
        )
        self.assertTrue(
            scoring._audio_usable_speech(
                {
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
                [],
            )
        )

        rejected = [
            ({"speech_state": "usable_speech"}, []),
            ({"usable_speech_detected": True}, []),
            (
                {
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                [],
            ),
            (
                {
                    "speech_state": "usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                ["audio_too_noisy"],
            ),
        ]
        for details, warnings in rejected:
            with self.subTest(details=details, warnings=warnings):
                self.assertFalse(
                    scoring._audio_usable_speech(details, warnings)
                )

    def test_duplicate_scan_result_prevention(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        client.get_scan_result_by_scan_id = MagicMock(return_value={"id": "existing"})
        client.update_item = MagicMock(return_value={"id": "existing"})
        client.create_item = MagicMock(return_value={"id": "new"})
        mode, payload = client.upsert_scan_result("scan-1", {"scan_id": "scan-1"})
        self.assertEqual(mode, "updated")
        self.assertEqual(payload["id"], "existing")
        client.create_item.assert_not_called()

    def test_duplicate_scan_result_conflict_recovers_by_update(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 409
        response._content = b'{"errors":[{"message":"duplicate key value violates unique constraint scan_results_scan_id_unique"}]}'
        conflict = requests.HTTPError(response=response)
        client.get_scan_result_by_scan_id = MagicMock(side_effect=[None, {"id": "existing"}])
        client.create_item = MagicMock(side_effect=conflict)
        client.update_item = MagicMock(return_value={"id": "existing"})

        mode, payload = client.upsert_scan_result("scan-1", {"scan_id": "scan-1"})

        self.assertEqual(mode, "updated_after_conflict")
        self.assertEqual(payload["id"], "existing")
        client.create_item.assert_called_once()
        client.update_item.assert_called_once()

    def test_one_business_profile_member_baseline_is_enforced_by_lookup_order(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        client.list_items = MagicMock(return_value=[])
        client.get_employee_baseline("member-1", "bp-1")
        _, kwargs = client.list_items.call_args
        self.assertEqual(kwargs["sort"], "-scan_count,-date_updated")

    def test_processing_readiness_uses_bounded_read_only_collection_read(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        client.list_items = MagicMock(return_value=[])

        client.check_processing_readiness()

        client.list_items.assert_called_once_with(
            "wellness_scans",
            fields=["id"],
            limit=1,
        )

    def test_directus_400_logs_sanitized_summary_without_response_body(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 400
        response.url = "http://example.com/items/scan_results"
        response._content = b'{"errors":[{"message":"Invalid payload. Invalid one-to-many update structure: risk_level"}]}'

        with unittest.mock.patch("directus_client.requests.request", return_value=response):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                with self.assertRaises(requests.HTTPError):
                    client.create_item(
                        "scan_results",
                        {
                            "scan_id": "scan-1",
                            "risk_level": "stable",
                            "confidence": 0.8,
                        },
                    )

        combined = "\n".join(logs.output)
        self.assertIn("response_summary={'status_code': 400", combined)
        self.assertIn("/items/scan_results", combined)
        self.assertIn("payload_keys=['confidence', 'risk_level', 'scan_id']", combined)
        self.assertNotIn("Invalid payload", combined)
        self.assertNotIn("one-to-many", combined)


    def test_directus_value_too_long_logs_only_sanitized_error_code(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 400
        response.url = "http://example.com/items/scan_results"
        response._content = json.dumps(
            {
                "errors": [
                    {
                        "message": (
                            "Value 'Conntinuity Intelligence Engine v1.2' for field "
                            "'ai_model_version' in collection 'scan_results' is too long."
                        ),
                        "extensions": {
                            "collection": "scan_results",
                            "field": "ai_model_version",
                            "value": "Conntinuity Intelligence Engine v1.2",
                            "code": "VALUE_TOO_LONG",
                        },
                    }
                ]
            }
        ).encode("utf-8")

        with unittest.mock.patch("directus_client.requests.request", return_value=response):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                with self.assertRaises(requests.HTTPError):
                    client.create_item(
                        "scan_results",
                        {
                            "scan_id": "scan-1",
                            "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                        },
                    )

        combined = "\n".join(logs.output)
        self.assertIn("'code': 'VALUE_TOO_LONG'", combined)
        self.assertIn("ai_model_version", combined)
        self.assertNotIn("for field", combined)
        self.assertNotIn("is too long", combined)

    def test_run_parallel_analysis_returns_real_results_before_deadline(self):
        fake_runtime = _FakeRuntime()

        with unittest.mock.patch.object(main, "get_analyzer_runtime", return_value=fake_runtime):
            results, worker_states = main._run_parallel_analysis(
                "scan-1",
                main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"),
            )

        self.assertEqual(results["video"]["details"]["analyzer"], "video")
        self.assertEqual(results["audio"]["details"]["analyzer"], "audio")
        self.assertEqual(results["image"]["details"]["analyzer"], "image")
        self.assertTrue(all(not state["timed_out"] for state in worker_states.values()))
        self.assertTrue(all(state["result_received"] for state in worker_states.values()))
        self.assertEqual(len(fake_runtime.run_scan_calls), 1)

    def test_finalize_analysis_worker_reads_completed_pipe_payload(self):
        shared_state = {
            "payload": {
                "ok": True,
                "result": {"score": 0.94, "details": {"status": "ok"}},
                "metrics": {"child_entry_ms": 11, "analyzer_execution_ms": 7, "result_send_ms": 2, "total_worker_ms": 18},
            },
            "ready": True,
        }
        conn = _FakePipeEndpoint(shared_state)
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": False})

        result, state = main._finalize_analysis_worker(
            scan_id="scan-1",
            analyzer_name="video",
            process=process,
            result_conn=conn,
            started_at=time.perf_counter() - 0.01,
            parent_process_start_ms=3,
            timeout_seconds=main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            timed_out=False,
        )

        self.assertEqual(result["score"], 0.94)
        self.assertEqual(state["child_entry_ms"], 11)
        self.assertEqual(state["analyzer_execution_ms"], 7)
        self.assertTrue(state["result_received"])
        self.assertFalse(state["timed_out"])
        self.assertFalse(state["alive"])

    def test_finalize_analysis_process_kills_still_alive_worker(self):
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": True, "requires_kill": True})

        state = main._finalize_analysis_process(process)

        self.assertTrue(state["terminated"])
        self.assertTrue(state["killed"])
        self.assertFalse(state["alive"])
        self.assertEqual(process.join_calls, [1.0, 1.0])

    def test_finalize_analysis_worker_terminates_timed_out_worker(self):
        conn = _FakePipeEndpoint({"payload": None, "ready": False})
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": True})

        result, state = main._finalize_analysis_worker(
            scan_id="scan-1",
            analyzer_name="image",
            process=process,
            result_conn=conn,
            started_at=time.perf_counter() - 0.01,
            parent_process_start_ms=4,
            timeout_seconds=main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            timed_out=True,
        )

        self.assertTrue(state["terminated"])
        self.assertFalse(state["alive"])
        self.assertFalse(state["killed"])
        self.assertIn("image_timeout", result["details"]["image_warnings"])

    def test_timeout_placeholder_is_not_valid_fatigue_evidence(self):
        placeholder = main._analysis_timeout_placeholder("video")

        self.assertFalse(main._result_has_valid_evidence(placeholder))
        self.assertIn("video_timeout", placeholder["details"]["visual_warnings"])

    def test_partial_modality_timeout_preserves_success_path(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": [], "timings_ms": {}}},
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "audio": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "image": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "baseline_ready_for_personalized_scoring", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            transcribe_optional = stack.enter_context(unittest.mock.patch.object(main, "_transcribe_audio_file_optional", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertEqual(result["failure_reason"], main.FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT)
        write_success.assert_not_called()
        transcribe_optional.assert_not_called()

    def test_process_scan_sync_partial_timeout_with_sufficient_evidence_completes(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": main._analysis_timeout_placeholder("audio"),
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "final_alive": False, "result_received": True, "analyzer_error": False, "process_exitcode": 0},
                            "audio": {"timed_out": True, "final_alive": False, "result_received": False, "analyzer_error": False, "process_exitcode": -15},
                            "image": {"timed_out": False, "final_alive": False, "result_received": True, "analyzer_error": False, "process_exitcode": 0},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "baseline_ready_for_personalized_scoring",
                    return_value=False,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertEqual(result["failure_reason"], main.FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT)
        write_success.assert_not_called()

    def test_process_scan_sync_timeout_path_finalizes_failed_scan(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": main._analysis_timeout_placeholder("video"),
                            "audio": main._analysis_timeout_placeholder("audio"),
                            "image": main._analysis_timeout_placeholder("image"),
                        },
                        {
                            "video": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "audio": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "image": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": None, "audio": 0.0, "video": 0.0, "image": 0.0},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": ["audio_timeout"],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False, "warnings": ["audio_timeout"]},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(main.FAILURE_REASON_LOW_QUALITY_MEDIA, ["video", "audio", "image"])))
            mark_failed_terminal = stack.enter_context(unittest.mock.patch.object(main, "_mark_scan_failed_terminal", return_value={"wellness_scan": "failed_updated"}))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", side_effect=AssertionError("write_success should not run on timeout failure")))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        self.assertEqual(result["failure_reason"], "validation_timeout")
        mark_failed_terminal.assert_called_once()

    def test_process_scan_sync_production_scores_preserve_full_multimodal_evidence(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(_production_multimodal_results(), _successful_worker_states()),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": True, "present": True},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 3,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "baseline_ready_for_personalized_scoring",
                    return_value=False,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            mark_failed_terminal = stack.enter_context(unittest.mock.patch.object(main, "_mark_scan_failed_terminal"))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            upsert_baseline = stack.enter_context(unittest.mock.patch.object(main.directus, "upsert_employee_baseline", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_COMPLETED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        write_success.assert_called_once()
        mark_failed_terminal.assert_not_called()
        written_result = write_success.call_args.kwargs["result"]
        self.assertEqual(written_result["voice_confidence"], 0.9158)
        self.assertEqual(written_result["modality_scores"]["video"], 0.7666)
        self.assertEqual(written_result["modality_scores"]["audio"], 0.9158)
        self.assertEqual(written_result["modality_scores"]["image"], 0.8969)
        upsert_baseline.assert_not_called()

    def test_process_scan_sync_writeback_failure_without_recovery_attempts_terminal_failed_update(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(_production_multimodal_results(), _successful_worker_states()),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": True, "present": True},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 3,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "voice_confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "baseline_ready_for_personalized_scoring",
                    return_value=False,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_fused_result_valid_modalities",
                    return_value=["video", "audio", "image"],
                )
            )
            write_failure = main.ProcessingError(main.FAILURE_REASON_WRITEBACK_FAILED, "RuntimeError")
            write_success = MagicMock(side_effect=write_failure)
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            mark_failed_terminal = MagicMock(return_value={"wellness_scan": "failed_updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_mark_scan_failed_terminal", mark_failed_terminal))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            upsert_baseline = stack.enter_context(unittest.mock.patch.object(main.directus, "upsert_employee_baseline", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        self.assertEqual(result["writeback_status"]["wellness_scan"], "failed_updated")
        self.assertEqual(result["writeback_status"]["recovery"], "unavailable")
        write_success.assert_called_once()
        mark_failed_terminal.assert_called_once()
        upsert_baseline.assert_not_called()

    def test_recover_completed_scan_if_result_exists_finalizes_completed_without_duplicate_result_write(self):
        main.directus.get_scan_result_by_scan_id = MagicMock(return_value={"id": "scan-result-1"})
        main.directus.update_wellness_scan = MagicMock(return_value={"id": "scan-1"})
        main.directus.upsert_scan_result = MagicMock()

        recovery = main._recover_completed_scan_if_result_exists("scan-1")

        self.assertTrue(recovery)
        main.directus.get_scan_result_by_scan_id.assert_called_once_with("scan-1")
        main.directus.update_wellness_scan.assert_called_once()
        main.directus.upsert_scan_result.assert_not_called()

    def test_mark_scan_failed_terminal_logs_unconfirmed_failure(self):
        with unittest.mock.patch.object(main, "_mark_scan_failed", return_value={"wellness_scan": "failed:RuntimeError"}):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                status = main._mark_scan_failed_terminal("scan-1", "writeback_failed", "sanitized message")

        self.assertEqual(status["wellness_scan"], "failed:RuntimeError")
        self.assertIn("terminal_failure_writeback_unconfirmed", "\n".join(logs.output))

    def test_mark_scan_failed_sanitizes_exception_details(self):
        with unittest.mock.patch.object(main.directus, "update_wellness_scan", side_effect=RuntimeError("token=secret")):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                status = main._mark_scan_failed("scan-1", "writeback_failed", "sanitized message")

        self.assertEqual(status["wellness_scan"], "failed:RuntimeError")
        combined = "\n".join(logs.output)
        self.assertIn("error_type=RuntimeError", combined)
        self.assertNotIn("token=secret", combined)

    def test_run_parallel_analysis_accepts_ready_pipe_payload_before_delayed_process_exit(self):
        fake_runtime = _FakeRuntime(
            run_scan_result=(
                {
                    "video": {"score": 0.9, "details": {"status": "ok", "analyzer": "video"}},
                    "audio": {"score": 0.9, "details": {"status": "ok", "analyzer": "audio"}},
                    "image": {"score": 0.9, "details": {"status": "ok", "analyzer": "image"}},
                },
                {
                    "video": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 2},
                    "audio": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 2},
                    "image": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 2},
                },
            )
        )

        with unittest.mock.patch.object(main, "get_analyzer_runtime", return_value=fake_runtime):
            results, worker_states = main._run_parallel_analysis(
                "scan-1",
                main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"),
            )

        self.assertTrue(all(state["result_received"] for state in worker_states.values()))
        self.assertTrue(all(not state["timed_out"] for state in worker_states.values()))
        self.assertEqual(results["video"]["details"]["analyzer"], "video")
        self.assertEqual(results["audio"]["details"]["analyzer"], "audio")
        self.assertEqual(results["image"]["details"]["analyzer"], "image")
        self.assertEqual(worker_states["video"]["worker_generation"], 2)

    def test_finalize_analysis_worker_reports_missing_payload_as_analyzer_error(self):
        conn = _FakePipeEndpoint({"payload": None, "ready": False})
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": False})

        result, state = main._finalize_analysis_worker(
            scan_id="scan-1",
            analyzer_name="audio",
            process=process,
            result_conn=conn,
            started_at=time.perf_counter() - 0.01,
            parent_process_start_ms=5,
            timeout_seconds=main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            timed_out=False,
        )

        self.assertFalse(state["timed_out"])
        self.assertTrue(state["analyzer_error"])
        self.assertFalse(state["result_received"])
        self.assertEqual(result["details"]["status"], "error")
        self.assertIn("audio_analysis_error", result["details"]["audio_warnings"])

    def test_analysis_worker_closes_pipe_after_success(self):
        fake_video_module = types.ModuleType("video")
        fake_video_module.analyze_video = lambda path: {"score": 0.7, "details": {"status": "ok"}}
        shared_state = {"payload": None, "ready": False}
        conn = _FakePipeEndpoint(shared_state)

        with unittest.mock.patch.dict(sys.modules, {"video": fake_video_module}):
            analysis_worker.run_analysis_worker(conn, "video", "clip.mp4", "video_missing", "scan-1", time.perf_counter() - 0.01)

        self.assertTrue(conn.closed)
        self.assertTrue(shared_state["ready"])
        self.assertTrue(shared_state["payload"]["ok"])

    def test_analysis_worker_closes_pipe_after_structured_error(self):
        fake_audio_module = types.ModuleType("audio")

        def raise_error(path):
            raise RuntimeError("boom")

        fake_audio_module.analyze_audio = raise_error
        shared_state = {"payload": None, "ready": False}
        conn = _FakePipeEndpoint(shared_state)

        with unittest.mock.patch.dict(sys.modules, {"audio": fake_audio_module}):
            analysis_worker.run_analysis_worker(conn, "audio", "clip.wav", "audio_missing", "scan-1", time.perf_counter() - 0.01)

        self.assertTrue(conn.closed)
        self.assertTrue(shared_state["ready"])
        self.assertFalse(shared_state["payload"]["ok"])
        self.assertEqual(shared_state["payload"]["error_type"], "RuntimeError")

    def test_spawn_worker_smoke_receives_payload_and_exits(self):
        ctx = multiprocessing.get_context("spawn")
        try:
            parent_conn, child_conn = ctx.Pipe(duplex=False)
        except PermissionError as exc:
            self.skipTest(f"multiprocessing Pipe unavailable in this sandbox: {exc}")
        process = ctx.Process(
            target=analysis_worker.run_analysis_worker,
            args=(child_conn, "video", None, "video_missing", "scan-smoke", time.perf_counter()),
            daemon=True,
        )
        process.start()
        child_conn.close()

        self.assertTrue(parent_conn.poll(5.0))
        payload = parent_conn.recv()
        process.join(5.0)

        self.assertTrue(payload["ok"])
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        parent_conn.close()
        process.close()

    def test_run_parallel_analysis_leaves_environment_unchanged(self):
        fake_runtime = _FakeRuntime()
        original_value = os.environ.get("AI_SERVER_ANALYSIS_CHILD")

        with unittest.mock.patch.object(main, "get_analyzer_runtime", return_value=fake_runtime):
            main._run_parallel_analysis("scan-1", main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"))

        self.assertEqual(os.environ.get("AI_SERVER_ANALYSIS_CHILD"), original_value)

    def test_warm_runtime_start_reports_ready_once(self):
        fake_context = _FakeContext(behaviors=[
            {"alive_after_start": True, "alive_after_target": True},
            {"alive_after_start": True, "alive_after_target": True},
            {"alive_after_start": True, "alive_after_target": True},
        ])

        def ready_worker(result_conn, analyzer_name, generation):
            result_conn.send(
                {
                    "type": "ready",
                    "ok": True,
                    "worker_generation": generation,
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_import_ms": 2,
                        "total_worker_ms": 3,
                    },
                }
            )

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(
                analysis_runtime,
                "_wait_handles",
                side_effect=lambda handles, timeout=None: [handle for handle in handles if hasattr(handle, "poll") and handle.poll()],
            ):
                runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
                    "video": ready_worker,
                    "audio": ready_worker,
                    "image": ready_worker,
                })
                startup = runtime.start()
                second_start = runtime.start()

        self.assertTrue(runtime.is_ready())
        self.assertGreaterEqual(startup["analyzer_runtime_ready_ms"], 0)
        self.assertTrue(second_start["ready"])
        self.assertEqual(len(fake_context.processes), 3)
        self.assertTrue(all(process.started for process in fake_context.processes))
        self.assertEqual(startup["workers"]["video"]["spawn_metrics"]["analyzer_import_ms"], 2)
        runtime.shutdown()

    def test_worker_supervisor_reuses_same_generation_for_two_jobs(self):
        fake_context = _FakeContext(behaviors=[{"alive_after_target": True}])

        def ready_worker(result_conn, analyzer_name, generation):
            result_conn.send(
                {
                    "type": "ready",
                    "ok": True,
                    "worker_generation": generation,
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_import_ms": 2,
                        "total_worker_ms": 3,
                    },
                }
            )

        def on_send(payload, state):
            if isinstance(payload, dict) and payload.get("type") == "job":
                return {
                    "type": "result",
                    "job_id": payload["job_id"],
                    "scan_id": payload["scan_id"],
                    "worker_generation": payload["worker_generation"],
                    "ok": True,
                    "result": {"score": 0.91, "details": {"status": "ok", "analyzer": "video"}},
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_execution_ms": 2,
                        "response_send_ms": 1,
                        "total_worker_ms": 3,
                    },
                }
            return payload

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(
                analysis_runtime,
                "_wait_handles",
                side_effect=lambda handles, timeout=None: [handle for handle in handles if hasattr(handle, "poll") and handle.poll()],
            ):
                supervisor = analysis_runtime.WorkerSupervisor("video", worker_entry=ready_worker)
                supervisor.start()
                supervisor._parent_conn._on_send = on_send

                job1 = analysis_runtime._WorkerJob(
                    job_id="job-1",
                    scan_id="scan-1",
                    path="clip.mp4",
                    deadline_at=time.perf_counter() + 1.0,
                    submitted_at=time.perf_counter(),
                    future=analysis_runtime.Future(),
                )
                job2 = analysis_runtime._WorkerJob(
                    job_id="job-2",
                    scan_id="scan-1",
                    path="clip.mp4",
                    deadline_at=time.perf_counter() + 1.0,
                    submitted_at=time.perf_counter(),
                    future=analysis_runtime.Future(),
                )

                first = supervisor._dispatch_job(job1)
                second = supervisor._dispatch_job(job2)

                self.assertTrue(first["state"]["result_received"])
                self.assertTrue(second["state"]["result_received"])
                self.assertEqual(first["state"]["worker_generation"], second["state"]["worker_generation"])
                self.assertEqual(len(fake_context.processes), 1)
                self.assertTrue(fake_context.processes[0].started)
                supervisor.shutdown()

    def test_worker_supervisor_timeout_schedules_async_generation_restart(self):
        fake_context = _FakeContext(behaviors=[
            {"alive_after_start": True, "alive_after_target": True, "requires_kill": True},
        ])

        def ready_worker(result_conn, analyzer_name, generation):
            result_conn.send(
                {
                    "type": "ready",
                    "ok": True,
                    "worker_generation": generation,
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_import_ms": 2,
                        "total_worker_ms": 3,
                    },
                }
            )

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(
                analysis_runtime,
                "_wait_handles",
                side_effect=lambda handles, timeout=None: [handle for handle in handles if hasattr(handle, "poll") and handle.poll()],
            ):
                supervisor = analysis_runtime.WorkerSupervisor("image", worker_entry=ready_worker)
                supervisor.start()
                supervisor._parent_conn._on_send = lambda payload, state: _FakePipeEndpoint.NO_RESPONSE if isinstance(payload, dict) and payload.get("type") == "job" else payload
                restart_release = __import__("threading").Event()

                def simulated_restart(expected_generation, scheduled_at):
                    restart_release.wait(timeout=0.2)
                    with supervisor._lock:
                        supervisor._generation = expected_generation + 1
                        supervisor._restart_in_progress = False
                        supervisor._restart_ready = True
                        supervisor._restart_error_type = None

                supervisor._restart_worker_async = simulated_restart

                job = analysis_runtime._WorkerJob(
                    job_id="job-timeout",
                    scan_id="scan-1",
                    path="clip.jpg",
                    deadline_at=time.perf_counter() + 0.01,
                    submitted_at=time.perf_counter() - 0.05,
                    future=analysis_runtime.Future(),
                )

                completion = supervisor._dispatch_job(job)

        self.assertTrue(completion["state"]["timed_out"])
        self.assertFalse(completion["state"]["analyzer_error"])
        self.assertTrue(completion["state"]["terminated"])
        self.assertTrue(completion["state"]["killed"])
        self.assertTrue(completion["state"]["worker_restarted"])
        self.assertTrue(completion["state"]["worker_restart_scheduled"])
        self.assertTrue(completion["state"]["worker_restart_in_progress"])
        restart_release.set()
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and supervisor._generation < 2:
            time.sleep(0.01)
        self.assertEqual(supervisor._generation, 2)
        self.assertEqual(len(fake_context.processes), 1)
        supervisor.shutdown()

    def test_active_timeout_does_not_wait_for_replacement_ready(self):
        shared_state = {}
        fake_parent_conn = _FakePipeEndpoint(
            shared_state,
            on_send=lambda payload, state: _FakePipeEndpoint.NO_RESPONSE,
        )
        fake_process = _FakeProcess(
            target=None,
            args=(),
            daemon=True,
            behavior={"alive_after_start": True, "requires_kill": True},
        )
        supervisor = analysis_runtime.WorkerSupervisor("audio", recovery_timeout_seconds=0.01)
        supervisor._started = True
        supervisor._stopped = False
        supervisor._ready = True
        supervisor._generation = 4
        supervisor._process = fake_process
        supervisor._parent_conn = fake_parent_conn
        supervisor._child_conn = object()
        supervisor._jobs_by_id = {}
        supervisor._completed_jobs = analysis_runtime.OrderedDict()
        supervisor._expired_jobs = analysis_runtime.OrderedDict()

        restart_started = []
        restart_release = __import__("threading").Event()

        def delayed_restart(expected_generation, scheduled_at):
            restart_started.append((expected_generation, scheduled_at))
            restart_release.wait(timeout=0.3)
            with supervisor._lock:
                supervisor._restart_in_progress = False
                supervisor._restart_ready = True
                supervisor._generation = expected_generation + 1

        supervisor._restart_worker_async = delayed_restart
        job = analysis_runtime._WorkerJob(
            job_id="job-timeout",
            scan_id="scan-timeout",
            path="audio.wav",
            deadline_at=time.perf_counter() + 0.01,
            submitted_at=time.perf_counter() - 0.05,
            future=analysis_runtime.Future(),
        )

        with unittest.mock.patch.object(
            analysis_runtime,
            "_wait_handles",
            side_effect=lambda handles, timeout=None: [],
        ):
            started = time.perf_counter()
            completion = supervisor._dispatch_job(job)
            elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.15)
        self.assertTrue(completion["state"]["timed_out"])
        self.assertFalse(completion["state"]["analyzer_error"])
        self.assertFalse(completion["state"]["final_alive"])
        self.assertTrue(completion["state"]["terminated"])
        self.assertTrue(completion["state"]["killed"])
        self.assertTrue(completion["state"]["worker_restart_scheduled"])
        self.assertTrue(completion["state"]["worker_restart_in_progress"])
        self.assertFalse(completion["state"]["worker_restart_ready"])
        self.assertEqual(restart_started[0][0], 4)
        restart_release.set()
        supervisor.shutdown()

    def test_replacement_ready_wait_does_not_block_jobs_health_or_shutdown(self):
        ready_event = __import__("threading").Event()
        violations = []
        probe = _ProbeRLock()
        fake_context = _DelayedReadyContext(ready_event, violations, probe)
        supervisor = analysis_runtime.WorkerSupervisor(
            "audio",
            startup_timeout_seconds=0.5,
            recovery_timeout_seconds=0.02,
            poll_interval_seconds=0.005,
            context_factory=lambda: fake_context,
        )
        supervisor._lock = probe
        supervisor._started = True
        supervisor._generation = 7

        def fake_wait(handles, timeout=None):
            deadline = time.perf_counter() + min(float(timeout or 0.01), 0.02)
            while time.perf_counter() < deadline:
                ready = []
                for handle in handles:
                    if hasattr(handle, "poll") and handle.poll():
                        ready.append(handle)
                    elif isinstance(handle, _LockCheckingProcess) and not handle.is_alive():
                        ready.append(handle)
                if ready:
                    return ready
                time.sleep(0.001)
            return []

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
            with supervisor._lock:
                self.assertTrue(supervisor._schedule_restart_locked(7))
                restart_thread = supervisor._restart_thread
                self.assertFalse(supervisor._schedule_restart_locked(7))
            supervisor._start_restart_thread(restart_thread)
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline and not fake_context.processes:
                time.sleep(0.005)
            self.assertTrue(fake_context.processes)

            job = analysis_runtime._WorkerJob(
                job_id="job-during-restart",
                scan_id="scan-during-restart",
                path="audio.wav",
                deadline_at=time.perf_counter() + 0.03,
                submitted_at=time.perf_counter(),
                future=analysis_runtime.Future(),
            )
            started = time.perf_counter()
            completion = supervisor._dispatch_job(job)
            job_elapsed = time.perf_counter() - started

            queued = analysis_runtime._WorkerJob(
                job_id="queued-expired",
                scan_id="scan-queued",
                path="audio.wav",
                deadline_at=time.perf_counter() - 0.01,
                submitted_at=time.perf_counter(),
                future=analysis_runtime.Future(),
            )
            supervisor._jobs_by_id[queued.job_id] = queued
            finalize_started = time.perf_counter()
            queued_completion = supervisor.finalize_timed_out_job(queued.job_id)
            finalize_elapsed = time.perf_counter() - finalize_started

            health_started = time.perf_counter()
            health = supervisor.health()
            health_elapsed = time.perf_counter() - health_started

            shutdown_started = time.perf_counter()
            supervisor.shutdown()
            shutdown_elapsed = time.perf_counter() - shutdown_started

        self.assertLess(job_elapsed, 0.12)
        self.assertTrue(completion["state"]["timed_out"])
        self.assertFalse(completion["state"]["analyzer_error"])
        self.assertLess(finalize_elapsed, 0.08)
        self.assertTrue(queued_completion["state"]["timed_out"])
        self.assertLess(health_elapsed, 0.05)
        self.assertFalse(health["ready"])
        self.assertLess(shutdown_elapsed, 0.2)
        self.assertFalse(restart_thread.is_alive())
        self.assertFalse(fake_context.processes[0].is_alive())
        self.assertTrue(fake_context.processes[0].terminated)
        self.assertTrue(fake_context.processes[0].closed)
        self.assertIsNone(supervisor._process)
        self.assertEqual(supervisor._generation, 7)
        self.assertEqual(violations, [])

    def test_replacement_candidate_publish_is_atomic_and_stale_token_is_rejected(self):
        def fake_wait(handles, timeout=None):
            ready = []
            for handle in handles:
                if hasattr(handle, "poll") and handle.poll():
                    ready.append(handle)
                elif isinstance(handle, _LockCheckingProcess) and not handle.is_alive():
                    ready.append(handle)
            if ready:
                return ready
            time.sleep(min(float(timeout or 0.01), 0.01))
            return []

        ready_event = __import__("threading").Event()
        violations = []
        probe = _ProbeRLock()
        fake_context = _DelayedReadyContext(ready_event, violations, probe)
        supervisor = analysis_runtime.WorkerSupervisor(
            "audio",
            startup_timeout_seconds=0.5,
            recovery_timeout_seconds=0.02,
            poll_interval_seconds=0.005,
            context_factory=lambda: fake_context,
        )
        supervisor._lock = probe
        supervisor._started = True
        supervisor._generation = 2

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
            with supervisor._lock:
                self.assertTrue(supervisor._schedule_restart_locked(2))
                restart_thread = supervisor._restart_thread
            supervisor._start_restart_thread(restart_thread)
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline and not fake_context.processes:
                time.sleep(0.005)
            ready_event.set()
            restart_thread.join(timeout=1.0)

        self.assertFalse(restart_thread.is_alive())
        self.assertTrue(supervisor.health()["ready"])
        self.assertEqual(supervisor._generation, 3)
        self.assertIs(supervisor._process, fake_context.processes[0])
        self.assertIsNone(supervisor._restart_candidate_process)
        self.assertEqual(violations, [])
        supervisor.shutdown()

        stale_ready = __import__("threading").Event()
        stale_violations = []
        stale_probe = _ProbeRLock()
        stale_context = _DelayedReadyContext(stale_ready, stale_violations, stale_probe)
        stale_supervisor = analysis_runtime.WorkerSupervisor(
            "audio",
            startup_timeout_seconds=0.5,
            recovery_timeout_seconds=0.02,
            poll_interval_seconds=0.005,
            context_factory=lambda: stale_context,
        )
        stale_supervisor._lock = stale_probe
        stale_supervisor._started = True
        stale_supervisor._generation = 5

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
            with stale_supervisor._lock:
                self.assertTrue(stale_supervisor._schedule_restart_locked(5))
                stale_thread = stale_supervisor._restart_thread
            stale_supervisor._start_restart_thread(stale_thread)
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline and not stale_context.processes:
                time.sleep(0.005)
            with stale_supervisor._lock:
                stale_supervisor._restart_token += 1
                stale_supervisor._active_restart_token = stale_supervisor._restart_token
            stale_ready.set()
            stale_thread.join(timeout=1.0)

        self.assertFalse(stale_thread.is_alive())
        self.assertFalse(stale_supervisor.health()["ready"])
        self.assertEqual(stale_supervisor._generation, 5)
        self.assertIsNone(stale_supervisor._process)
        self.assertFalse(stale_context.processes[0].is_alive())
        self.assertTrue(stale_context.processes[0].terminated)
        self.assertEqual(stale_violations, [])
        stale_supervisor.shutdown()

    def test_persistent_runtime_spawn_smoke_reuses_same_worker(self):
        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        try:
            try:
                runtime.start()
            except PermissionError as exc:
                self.skipTest(f"multiprocessing unavailable in this sandbox: {exc}")
            except analysis_runtime.AnalysisRuntimeStartupError as exc:
                if (
                    isinstance(exc.__cause__, PermissionError)
                    or "WinError 5" in str(exc)
                    or "Access is denied" in str(exc)
                ):
                    self.skipTest(f"multiprocessing unavailable in this sandbox: {exc}")
                raise
            first = runtime.run_scan("scan-smoke-1", main.Media(image=None, audio=None, video=None), deadline_seconds=2.0)
            second = runtime.run_scan("scan-smoke-2", main.Media(image=None, audio=None, video=None), deadline_seconds=2.0)
        finally:
            runtime.shutdown()

        self.assertTrue(all(result["details"]["status"] == "missing" for result in first[0].values()))
        self.assertTrue(all(result["details"]["status"] == "missing" for result in second[0].values()))
        self.assertEqual(
            first[1]["video"]["worker_generation"],
            second[1]["video"]["worker_generation"],
        )

    def test_run_scan_waits_for_pending_future_completion(self):
        delayed_future = analysis_runtime.Future()
        video_future = analysis_runtime.Future()
        video_future.set_result(
            {
                "result": {"score": 0.9, "details": {"status": "ok", "analyzer": "video"}},
                "state": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
            }
        )
        image_future = analysis_runtime.Future()
        image_future.set_result(
            {
                "result": {"score": 0.9, "details": {"status": "ok", "analyzer": "image"}},
                "state": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
            }
        )

        runtime = analysis_runtime.WarmAnalyzerRuntime()
        runtime._started = True
        runtime._stopped = False
        runtime._runtime_state = "ready"
        runtime._supervisors = {
            "video": _SubmitOnlySupervisor(video_future),
            "audio": _SubmitOnlySupervisor(
                delayed_future,
                generation=3,
                completion={
                    "result": {
                        "score": None,
                        "details": {"status": "timeout", "audio_warnings": ["audio_timeout"]},
                    },
                    "state": {
                        "timed_out": True,
                        "analyzer_error": False,
                        "result_received": False,
                        "final_alive": False,
                        "terminated": True,
                        "killed": False,
                        "worker_restarted": True,
                        "process_exitcode": -15,
                        "worker_generation": 3,
                    },
                },
            ),
            "image": _SubmitOnlySupervisor(image_future),
        }

        started_at = time.perf_counter()
        results, worker_states = runtime.run_scan(
            "scan-1",
            main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"),
            deadline_seconds=0.01,
        )

        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.10)
        self.assertEqual(results["audio"]["details"]["status"], "timeout")
        self.assertTrue(worker_states["audio"]["timed_out"])
        self.assertFalse(worker_states["audio"]["analyzer_error"])
        self.assertTrue(worker_states["audio"]["worker_restarted"])
        self.assertFalse(worker_states["audio"]["result_received"])
        self.assertTrue(delayed_future.done())
        with self.assertRaises(InvalidStateError):
            delayed_future.set_result(
                {
                    "result": {"score": 0.8, "details": {"status": "ok", "analyzer": "audio"}},
                    "state": {"timed_out": False, "analyzer_error": False, "final_alive": False, "result_received": True, "worker_generation": 1},
                }
            )

    def test_timeout_only_affects_the_expired_queued_job_and_not_the_active_job(self):
        shared_state = {}

        def on_send(payload, state):
            if payload.get("job_id") == "job-a":
                return {
                    "type": "result",
                    "job_id": "job-a",
                    "scan_id": "scan-a",
                    "worker_generation": 1,
                    "ok": True,
                    "result": {"score": 0.91, "details": {"status": "ok", "analyzer": "audio"}},
                    "metrics": {"analyzer_execution_ms": 2, "response_send_ms": 1, "child_entry_ms": 1, "total_worker_ms": 3},
                }
            return _FakePipeEndpoint.NO_RESPONSE

        fake_parent_conn = _FakePipeEndpoint(shared_state, on_send=on_send)
        fake_process = _FakeProcess(target=None, args=(), daemon=True, behavior={"alive_after_start": True})
        supervisor = analysis_runtime.WorkerSupervisor("audio")
        supervisor._lock = __import__("threading").RLock()
        supervisor._started = True
        supervisor._stopped = False
        supervisor._ready = True
        supervisor._generation = 1
        supervisor._process = fake_process
        supervisor._parent_conn = fake_parent_conn
        supervisor._child_conn = object()
        supervisor._active_job_id = "job-a"
        supervisor._active_worker_generation = 1
        supervisor._jobs_by_id = {}
        supervisor._completed_jobs = analysis_runtime.OrderedDict()
        supervisor._expired_jobs = analysis_runtime.OrderedDict()

        job_a = analysis_runtime._WorkerJob(
            job_id="job-a",
            scan_id="scan-a",
            path="audio-a.wav",
            deadline_at=time.perf_counter() + 1.0,
            submitted_at=time.perf_counter() - 0.05,
            future=analysis_runtime.Future(),
            worker_generation=1,
        )
        job_b = analysis_runtime._WorkerJob(
            job_id="job-b",
            scan_id="scan-b",
            path="audio-b.wav",
            deadline_at=time.perf_counter() - 0.01,
            submitted_at=time.perf_counter() - 0.05,
            future=analysis_runtime.Future(),
            worker_generation=1,
        )
        supervisor._jobs_by_id = {"job-a": job_a, "job-b": job_b}

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=lambda handles, timeout=None: handles):
            completion_b = supervisor.finalize_timed_out_job("job-b")
            completion_a = supervisor._dispatch_job(job_a)

        self.assertTrue(completion_b["state"]["timed_out"])
        self.assertFalse(completion_b["state"]["analyzer_error"])
        self.assertFalse(fake_process.terminated)
        self.assertFalse(fake_process.killed)
        self.assertEqual(supervisor._jobs_by_id.get("job-a"), job_a)
        self.assertNotIn("job-b", supervisor._jobs_by_id)
        self.assertEqual(completion_a["result"]["details"]["status"], "ok")
        self.assertFalse(completion_a["state"]["timed_out"])
        self.assertFalse(completion_a["state"]["analyzer_error"])
        self.assertFalse(fake_process.terminated)
        self.assertFalse(fake_process.killed)

    def test_run_scan_finalizes_pending_workers_concurrently(self):
        class _SlowSupervisor:
            def __init__(self, name):
                self.name = name
                self._generation = 9
                self.finalize_started = []
                self.finalize_finished = []
                self.forgotten = []

            def submit(self, *, scan_id, path, deadline_at):
                future = analysis_runtime.Future()
                future._analysis_job_id = f"{scan_id}-{self.name}"
                future._analysis_analyzer_name = self.name
                self.future = future
                return future

            def finalize_timed_out_job(self, job_id):
                self.finalize_started.append(time.perf_counter())
                time.sleep(0.2)
                self.finalize_finished.append(time.perf_counter())
                return {
                    "result": {
                        "score": None,
                        "details": {"status": "timeout", "analyzer": self.name},
                    },
                    "state": {
                        "timed_out": True,
                        "analyzer_error": False,
                        "result_received": False,
                        "final_alive": False,
                        "terminated": True,
                        "killed": False,
                        "worker_restarted": True,
                        "process_exitcode": -15,
                        "worker_generation": self._generation,
                    },
                }

            def forget_job(self, job_id):
                self.forgotten.append(job_id)

            def health(self):
                return {
                    "ready": True,
                    "worker_generation": self._generation,
                    "process_alive": True,
                }

        runtime = analysis_runtime.WarmAnalyzerRuntime()
        runtime._started = True
        runtime._stopped = False
        runtime._runtime_state = "ready"
        supervisors = {name: _SlowSupervisor(name) for name in ("video", "audio", "image")}
        runtime._supervisors = supervisors

        started_at = time.perf_counter()
        results, worker_states = runtime.run_scan(
            "scan-concurrent",
            main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"),
            deadline_seconds=0.01,
        )
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.45)
        self.assertTrue(all(result["details"]["status"] == "timeout" for result in results.values()))
        self.assertTrue(all(state["timed_out"] for state in worker_states.values()))
        start_times = [supervisors[name].finalize_started[0] for name in supervisors]
        self.assertLess(max(start_times) - min(start_times), 0.1)
        self.assertTrue(all(supervisors[name].forgotten for name in supervisors))

    def test_job_registries_stay_bounded_after_many_sequential_jobs(self):
        supervisor = analysis_runtime.WorkerSupervisor("audio")
        supervisor._process = None
        supervisor._parent_conn = None
        supervisor._child_conn = None
        supervisor._ready = False
        supervisor._started = True
        supervisor._stopped = False

        peak_completed = 0
        peak_expired = 0
        with unittest.mock.patch.object(analysis_runtime, "_log_perf", lambda *args, **kwargs: None):
            for index in range(1000):
                job = analysis_runtime._WorkerJob(
                    job_id=f"job-{index}",
                    scan_id=f"scan-{index}",
                    path="audio.wav",
                    deadline_at=time.perf_counter() - 0.01,
                    submitted_at=time.perf_counter(),
                    future=analysis_runtime.Future(),
                )
                supervisor._jobs_by_id[job.job_id] = job
                completion = supervisor._finalize_job_locked(
                    job,
                    timed_out=True,
                    analyzer_error=False,
                    result_received=False,
                    process_state={"process_exited": True, "alive": False, "terminated": False, "killed": False, "process_exitcode": None},
                    worker_restarted=False,
                    child_metrics={},
                    queue_wait_ms=0,
                    dispatch_ms=0,
                    response_ms=0,
                    cleanup_worker=False,
                    clear_active=False,
                )
                self.assertTrue(job.future.done())
                self.assertTrue(completion["state"]["timed_out"])
                peak_completed = max(peak_completed, len(supervisor._completed_jobs))
                peak_expired = max(peak_expired, len(supervisor._expired_jobs))
                supervisor.forget_job(job.job_id)

        self.assertLessEqual(peak_completed, supervisor._completed_job_limit)
        self.assertLessEqual(peak_expired, supervisor._expired_job_limit)
        self.assertEqual(len(supervisor._completed_jobs), 0)
        self.assertEqual(len(supervisor._expired_jobs), 0)
        self.assertEqual(len(supervisor._jobs_by_id), 0)

    def test_process_scan_sync_uses_final_alive_not_worker_process_liveness(self):
        lifecycle = []

        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_fused_result_valid_modalities",
                    return_value=["video", "audio", "image"],
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": [], "timings_ms": {}}},
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": True, "worker_generation": 1},
                            "audio": {"timed_out": False, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": True, "worker_generation": 1},
                            "image": {"timed_out": False, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": True, "worker_generation": 1},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": True, "present": True},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 3,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "voice_confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "baseline_ready_for_personalized_scoring", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_log_validation_decision",
                    side_effect=lambda *args, **kwargs: lifecycle.append(("decision", kwargs)),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_log_validation_lifecycle",
                    side_effect=lambda *args, **kwargs: lifecycle.append(("lifecycle", kwargs)),
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_transcribe_audio_file_optional", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_COMPLETED)
        lifecycle_entry = next(item for item in lifecycle if item[0] == "lifecycle")
        self.assertEqual(lifecycle_entry[1]["running_modalities"], [])
        self.assertTrue(lifecycle_entry[1]["all_workers_terminal"])
        write_success.assert_called_once()

    def test_process_scan_sync_queued_timeout_is_not_workers_not_terminal(self):
        lifecycle = []

        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": main._analysis_timeout_placeholder("audio"),
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": True, "worker_generation": 1},
                            "audio": {"timed_out": True, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": False, "worker_generation": 1},
                            "image": {"timed_out": False, "analyzer_error": False, "final_alive": False, "worker_process_alive": True, "result_received": True, "worker_generation": 1},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "voice_confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "baseline_ready_for_personalized_scoring", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_log_validation_decision",
                    side_effect=lambda *args, **kwargs: lifecycle.append(("decision", kwargs)),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_log_validation_lifecycle",
                    side_effect=lambda *args, **kwargs: lifecycle.append(("lifecycle", kwargs)),
                )
            )

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertEqual(result["failure_reason"], main.FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT)
        decision_entry = next(item for item in lifecycle if item[0] == "decision")
        lifecycle_entry = next(item for item in lifecycle if item[0] == "lifecycle")
        self.assertNotEqual(decision_entry[1]["terminal_reason"], "workers_not_terminal")
        self.assertEqual(lifecycle_entry[1]["running_modalities"], [])
        self.assertTrue(lifecycle_entry[1]["all_workers_terminal"])
        write_success.assert_not_called()

    def test_audio_analyze_audio_accepts_optional_scan_id(self):
        decode_calls = []
        fake_result = {
            "audio_confidence": 0.9,
            "timings_ms": {},
        }

        with unittest.mock.patch.object(audio, "_normalize_audio_path", return_value="clip.wav"):
            with unittest.mock.patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")):
                with unittest.mock.patch.object(audio, "librosa", object()):
                    with unittest.mock.patch.object(
                        audio,
                        "_decode_audio_once",
                        side_effect=lambda path, scan_id=None: decode_calls.append(scan_id) or (
                            object(),
                            16000,
                            5.0,
                            3.0,
                            "wave",
                            1,
                        ),
                    ):
                        with unittest.mock.patch.object(audio, "_build_success_details", return_value=fake_result):
                            with unittest.mock.patch.object(audio, "_log_audio_perf"):
                                default_result = audio.analyze_audio("clip.wav")
                                scanned_result = audio.analyze_audio("clip.wav", scan_id="scan-1")

        self.assertEqual(decode_calls, [None, "scan-1"])
        self.assertEqual(default_result["score"], 0.9)
        self.assertEqual(scanned_result["score"], 0.9)

    def test_audio_decode_happens_once_and_reuses_normalized_samples(self):
        sentinel = audio.np.zeros(16, dtype=audio.np.float32)
        captured = {}
        features = {
            "rms_energy": 0.03,
            "noise_estimate": 0.1,
            "silence_ratio": 0.1,
            "speech_presence_score": 0.4,
            "clipping_ratio": 0.0,
            "tonal_concentration": 0.2,
            "rms_variation": 0.1,
            "peak_volume": 0.2,
            "voice_clarity_score": 0.7,
            "centroid": 1000.0,
            "flatness": 0.2,
            "zcr_mean": 0.1,
            "mfcc_summary": [0.0] * 5,
            "timings_ms": {},
        }

        with unittest.mock.patch.object(audio, "_normalize_audio_path", return_value="clip.wav"):
            with unittest.mock.patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")):
                with unittest.mock.patch.object(audio, "librosa", object()):
                    with unittest.mock.patch.object(
                        audio,
                        "_decode_audio_once",
                        return_value=(sentinel, 16000, 5.0, 3.0, "wave", 1),
                        ) as decode_once:
                        with unittest.mock.patch.object(audio, "_ensure_1d_float32", return_value=sentinel):
                            def capture_features(y, sr, scan_id=None):
                                captured["y"] = y
                                return features

                            with unittest.mock.patch.object(
                                audio,
                                "_feature_pipeline",
                                side_effect=capture_features,
                            ):
                                with unittest.mock.patch.object(
                                    audio,
                                    "_speech_state_and_warnings",
                                    return_value=([], "usable_speech", False, True),
                                ):
                                    with unittest.mock.patch.object(audio, "_log_audio_perf"):
                                        audio.analyze_audio("clip.wav", scan_id="scan-1")

        self.assertEqual(decode_once.call_count, 1)
        self.assertIs(captured["y"], sentinel)

    def test_audio_analyze_audio_does_not_use_nested_worker_or_queue(self):
        with unittest.mock.patch.object(audio, "_normalize_audio_path", return_value="clip.wav"):
            with unittest.mock.patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")):
                with unittest.mock.patch.object(audio, "librosa", object()):
                    with unittest.mock.patch.object(
                        audio,
                        "_decode_audio_once",
                        return_value=(object(), 16000, 5.0, 3.0, "wave", 1),
                    ):
                        with unittest.mock.patch.object(
                            audio,
                            "_build_success_details",
                            return_value={"audio_confidence": 0.9, "timings_ms": {}},
                        ):
                            with unittest.mock.patch.object(audio, "analyze_audio_worker", side_effect=AssertionError("nested worker not allowed")):
                                with unittest.mock.patch.object(audio, "_log_audio_perf"):
                                    result = audio.analyze_audio("clip.wav", scan_id="scan-1")

        self.assertEqual(result["score"], 0.9)

    def test_audio_substep_logs_are_sanitized(self):
        log_messages = []

        def capture_log(message, *args, **kwargs):
            log_messages.append(message % args if args else message)

        with unittest.mock.patch.object(audio, "_normalize_audio_path", return_value="clip.wav"):
            with unittest.mock.patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")):
                with unittest.mock.patch.object(audio, "librosa", object()):
                    with unittest.mock.patch.object(
                        audio,
                        "_decode_audio_once",
                        return_value=(object(), 16000, 5.0, 3.0, "wave", 1),
                    ):
                        with unittest.mock.patch.object(
                            audio,
                            "_build_success_details",
                            return_value={"audio_confidence": 0.9, "timings_ms": {}},
                        ):
                            with unittest.mock.patch.object(audio.logger, "info", side_effect=capture_log):
                                with unittest.mock.patch.object(audio, "_log_audio_perf", wraps=audio._log_audio_perf):
                                    audio.analyze_audio("clip.wav", scan_id="scan-1")

        combined = "\n".join(log_messages)
        self.assertIn("scan-1", combined)
        self.assertNotIn("clip.wav", combined)
        self.assertNotIn("token=", combined)

    def test_audio_feature_pipeline_reuses_shared_frame_matrix_and_preserves_reference_values(self):
        sr = 16000
        duration_seconds = 3.0
        sample_count = int(sr * duration_seconds)
        base_time = audio.np.linspace(0.0, duration_seconds, sample_count, endpoint=False)
        signals = [
            ("silence", audio.np.zeros(sample_count, dtype=audio.np.float32)),
            ("single_sine", audio.np.asarray(audio.np.sin(2.0 * audio.np.pi * 220.0 * base_time), dtype=audio.np.float32)),
            (
                "multi_tone",
                audio.np.asarray(
                    0.55 * audio.np.sin(2.0 * audio.np.pi * 180.0 * base_time)
                    + 0.33 * audio.np.sin(2.0 * audio.np.pi * 420.0 * base_time)
                    + 0.12 * audio.np.sin(2.0 * audio.np.pi * 860.0 * base_time),
                    dtype=audio.np.float32,
                ),
            ),
            (
                "voice_like",
                audio.np.asarray(
                    (0.20 + 0.12 * audio.np.sin(2.0 * audio.np.pi * 1.7 * base_time))
                    * (
                        0.55 * audio.np.sin(2.0 * audio.np.pi * 140.0 * base_time)
                        + 0.30 * audio.np.sin(2.0 * audio.np.pi * 280.0 * base_time)
                        + 0.15 * audio.np.sin(2.0 * audio.np.pi * 420.0 * base_time)
                    ),
                    dtype=audio.np.float32,
                ),
            ),
            ("clipped", audio.np.asarray(audio.np.clip(1.15 * audio.np.sin(2.0 * audio.np.pi * 260.0 * base_time), -1.0, 1.0), dtype=audio.np.float32)),
            (
                "noise",
                audio.np.asarray(
                    audio.np.random.default_rng(42).normal(0.0, 0.08, sample_count),
                    dtype=audio.np.float32,
                ),
            ),
        ]

        class _FakeFilters:
            @staticmethod
            def mel(*, sr, n_fft, n_mels):
                bins = n_fft // 2 + 1
                basis = audio.np.zeros((n_mels, bins), dtype=audio.np.float64)
                band_edges = audio.np.linspace(0, bins - 1, n_mels + 2)
                for row in range(n_mels):
                    left = int(round(band_edges[row]))
                    center = int(round(band_edges[row + 1]))
                    right = int(round(band_edges[row + 2]))
                    center = max(center, left + 1)
                    right = max(right, center + 1)
                    for col in range(left, min(center + 1, bins)):
                        basis[row, col] = (col - left) / max(center - left, 1)
                    for col in range(center, min(right + 1, bins)):
                        basis[row, col] = max(basis[row, col], (right - col) / max(right - center, 1))
                return basis

        class _FakeLibrosa:
            filters = _FakeFilters()

            @staticmethod
            def power_to_db(power, ref=audio.np.max):
                matrix = audio.np.asarray(power, dtype=audio.np.float64)
                reference = ref(matrix) if callable(ref) else ref
                reference = max(float(reference), 1e-10)
                return 10.0 * audio.np.log10(audio.np.maximum(matrix, 1e-10) / reference)

        frame_matrix_calls = []
        original_frame_matrix = audio._frame_matrix

        def spy_frame_matrix(samples_value, frame_length, hop_length):
            frame_matrix_calls.append((frame_length, hop_length))
            return original_frame_matrix(samples_value, frame_length, hop_length)

        with unittest.mock.patch.object(audio, "librosa", _FakeAudioLibrosa):
            for label, samples in signals:
                frame_matrix_calls.clear()
                frame_length = min(2048, max(512, int(sr * 0.032)))
                hop_length = max(256, int(frame_length / 4))
                reference = {
                    "rms": audio._rms_numpy(samples, frame_length=frame_length, hop_length=hop_length),
                    "zcr": audio._zero_crossing_rate_numpy(samples, frame_length=frame_length, hop_length=hop_length),
                    "centroid": audio._spectral_centroid(samples, sr, hop_length=hop_length, n_fft=frame_length),
                    "flatness": audio._spectral_flatness(samples, sr, hop_length=hop_length, n_fft=frame_length),
                    "mfcc_summary": audio._mfcc_summary_like(samples, sr, hop_length=hop_length, n_fft=frame_length),
                }
                reference["rms_energy"] = float(audio.np.mean(reference["rms"])) if reference["rms"].size else 0.0
                reference["silence_ratio"] = float(audio.np.mean(reference["rms"] < max(audio.MIN_RMS_ENERGY * 0.6, reference["rms_energy"] * 0.35))) if reference["rms"].size else 1.0
                reference["zcr_mean"] = float(audio.np.mean(reference["zcr"])) if reference["zcr"].size else 0.0
                reference["rms_variation"] = float(audio.np.std(reference["rms"]) / max(reference["rms_energy"], 1e-6)) if reference["rms"].size else 0.0
                reference["full_spectrum"] = audio.np.abs(audio.np.fft.rfft(samples * audio.np.hanning(samples.size).astype(audio.np.float32, copy=False))) if samples.size else audio.np.asarray([], dtype=audio.np.float32)
                reference["dominant_concentration"] = float(audio.np.max(reference["full_spectrum"]) / max(float(audio.np.sum(reference["full_spectrum"])), 1e-6)) if reference["full_spectrum"].size else 0.0
                reference["tonal_concentration"] = float(audio.np.clip(0.55 * reference["dominant_concentration"] + 0.25 * (1.0 - reference["flatness"]) + 0.20 * audio.clamp01(1.0 - reference["rms_variation"] / 1.5, 0.0), 0.0, 1.0))

                with unittest.mock.patch.object(audio, "_frame_matrix", side_effect=spy_frame_matrix):
                    features = audio._feature_pipeline(samples, sr, scan_id=f"scan-{label}")

                self.assertEqual(len(frame_matrix_calls), 1, msg=label)
                self.assertAlmostEqual(features["rms_energy"], reference["rms_energy"], delta=1e-6, msg=label)
                self.assertAlmostEqual(features["silence_ratio"], reference["silence_ratio"], delta=1e-6, msg=label)
                self.assertAlmostEqual(features["zcr_mean"], reference["zcr_mean"], delta=1e-6, msg=label)
                self.assertAlmostEqual(features["centroid"], reference["centroid"], delta=1e-6, msg=label)
                self.assertAlmostEqual(features["flatness"], reference["flatness"], delta=1e-6, msg=label)
                self.assertAlmostEqual(features["tonal_concentration"], reference["tonal_concentration"], delta=1e-6, msg=label)
                for actual, expected in zip(features["mfcc_summary"], reference["mfcc_summary"]):
                    self.assertAlmostEqual(actual, expected, delta=1e-5, msg=label)
                self.assertGreaterEqual(features["speech_presence_score"], 0.0, msg=label)

    def test_audio_analyze_audio_decodes_once_when_success_details_are_stubbed(self):
        frame_calls = []
        original_frame_matrix = audio._frame_matrix

        def spy_frame_matrix(samples_value, frame_length, hop_length):
            frame_calls.append((frame_length, hop_length))
            return original_frame_matrix(samples_value, frame_length, hop_length)

        fake_result = {
            "audio_confidence": 0.9,
            "timings_ms": {},
        }
        decode_once = MagicMock(return_value=(audio.np.zeros(16, dtype=audio.np.float32), 16000, 5.0, 3.0, "wave", 1))

        with unittest.mock.patch.object(audio, "_normalize_audio_path", return_value="clip.wav"):
            with unittest.mock.patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")):
                with unittest.mock.patch.object(audio, "librosa", _FakeAudioLibrosa):
                    with unittest.mock.patch.object(audio, "_frame_matrix", side_effect=spy_frame_matrix):
                        with unittest.mock.patch.object(audio, "_decode_audio_once", decode_once):
                            with unittest.mock.patch.object(audio, "_build_success_details", return_value=fake_result):
                                with unittest.mock.patch.object(audio, "_log_audio_perf"):
                                    result = audio.analyze_audio("clip.wav", scan_id="scan-1")

        self.assertEqual(result["score"], 0.9)
        decode_once.assert_called_once()
        self.assertEqual(len(frame_calls), 0)

    def test_worker_supervisor_startup_timeout_defaults_to_fifteen_seconds(self):
        supervisor = analysis_runtime.WorkerSupervisor("audio")
        self.assertEqual(supervisor._startup_timeout_seconds, 15.0)

    def test_worker_supervisor_startup_fails_closed_without_ready(self):
        fake_context = _FakeContext(behaviors=[{"invoke_target": False, "alive_after_start": True, "alive_after_target": True}])

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(analysis_runtime, "_wait_handles", return_value=[]):
                supervisor = analysis_runtime.WorkerSupervisor("audio", worker_entry=analysis_runtime._smoke_worker_main)
                with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError):
                    supervisor.start()

        self.assertEqual(len(fake_context.processes), 1)
        self.assertTrue(fake_context.processes[0].terminated or fake_context.processes[0].killed)

    def test_audio_worker_does_not_report_ready_before_prewarm_completes(self):
        conn = _ReadyThenEOFConn()
        order = []

        def prewarm(progress=None):
            order.append("prewarm")
            self.assertTrue(all(item.get("type") == "startup_progress" for item in conn.sent))
            return {
                "audio_prewarm_first_call_ms": 11,
                "audio_prewarm_second_call_ms": 3,
                "audio_prewarm_decode_ms": 1,
                "audio_prewarm_vad_ms": 1,
                "audio_prewarm_pitch_ms": 0,
                "audio_prewarm_spectral_ms": 2,
                "audio_prewarm_mel_ms": 1,
                "audio_prewarm_mfcc_ms": 1,
                "audio_prewarm_total_ms": 14,
                "audio_warm_benchmark_passed": True,
            }

        with unittest.mock.patch.object(analysis_worker, "_load_analyzer_callable", return_value=lambda path, scan_id=None: {}):
            with unittest.mock.patch.object(analysis_worker, "_prewarm_analyzer", side_effect=lambda name, progress=None: prewarm(progress)):
                analysis_worker.worker_main(conn, "audio", 7)

        self.assertEqual(order, ["prewarm"])
        ready_payloads = [item for item in conn.sent if item.get("type") == "ready"]
        self.assertEqual(len(ready_payloads), 1)
        ready = ready_payloads[0]
        self.assertEqual(ready["type"], "ready")
        self.assertTrue(ready["ok"])
        self.assertEqual(ready["worker_generation"], 7)
        self.assertTrue(ready["metrics"]["audio_warm_benchmark_passed"])
        self.assertIn("audio_import_ms", ready["metrics"])

    def test_audio_worker_prewarm_failure_does_not_publish_ready(self):
        conn = _ReadyThenEOFConn()

        with unittest.mock.patch.object(analysis_worker, "_load_analyzer_callable", return_value=lambda path, scan_id=None: {}):
            with unittest.mock.patch.object(analysis_worker, "_prewarm_analyzer", return_value={"audio_warm_benchmark_passed": False}):
                analysis_worker.worker_main(conn, "audio", 1)

        ready = [item for item in conn.sent if item.get("type") == "ready"]
        self.assertEqual(len(ready), 1)
        self.assertFalse(ready[0]["ok"])
        self.assertEqual(ready[0]["terminal_reason"], "audio_prewarm_second_result_invalid")

    def test_audio_worker_ready_failure_reason_distinguishes_import_failed(self):
        conn = _ReadyThenEOFConn()

        with unittest.mock.patch.object(analysis_worker, "_load_analyzer_callable", side_effect=RuntimeError("boom")):
            analysis_worker.worker_main(conn, "audio", 1)

        ready = [item for item in conn.sent if item.get("type") == "ready"]
        self.assertEqual(len(ready), 1)
        self.assertFalse(ready[0]["ok"])
        self.assertEqual(ready[0]["terminal_reason"], "audio_import_failed")
        self.assertEqual(ready[0]["failure_stage"], "import")

    def test_audio_worker_ready_failure_reason_distinguishes_first_call_failed(self):
        conn = _ReadyThenEOFConn()
        error = audio.AudioPrewarmError(
            "audio_prewarm_first_call_failed",
            failure_stage="first_call",
            metrics={"audio_prewarm_first_call_ms": 3},
        )

        with unittest.mock.patch.object(analysis_worker, "_load_analyzer_callable", return_value=lambda path, scan_id=None: {}):
            with unittest.mock.patch.object(analysis_worker, "_prewarm_analyzer", side_effect=error):
                analysis_worker.worker_main(conn, "audio", 1)

        ready = [item for item in conn.sent if item.get("type") == "ready"]
        self.assertEqual(len(ready), 1)
        self.assertFalse(ready[0]["ok"])
        self.assertEqual(ready[0]["terminal_reason"], "audio_prewarm_first_call_failed")
        self.assertEqual(ready[0]["failure_stage"], "first_call")

    def test_audio_prewarm_executes_once_per_worker_generation(self):
        calls = []

        def prewarm(progress=None):
            calls.append(time.time())
            return {
                "audio_prewarm_first_call_ms": 10,
                "audio_prewarm_second_call_ms": 2,
                "audio_prewarm_decode_ms": 1,
                "audio_prewarm_vad_ms": 1,
                "audio_prewarm_pitch_ms": 0,
                "audio_prewarm_spectral_ms": 1,
                "audio_prewarm_mel_ms": 1,
                "audio_prewarm_mfcc_ms": 1,
                "audio_prewarm_total_ms": 12,
                "audio_warm_benchmark_passed": True,
            }

        for generation in [1, 2]:
            conn = _ReadyThenEOFConn()
            with unittest.mock.patch.object(analysis_worker, "_load_analyzer_callable", return_value=lambda path, scan_id=None: {}):
                with unittest.mock.patch.object(analysis_worker, "_prewarm_analyzer", side_effect=lambda name, progress=None: prewarm(progress)):
                    analysis_worker.worker_main(conn, "audio", generation)
            self.assertEqual(conn.sent[0]["worker_generation"], generation)

        self.assertEqual(len(calls), 2)

    def test_required_audio_timeout_blocks_completed_full_result(self):
        quality_result = {
            "media_quality": {
                "video": {"present": True, "usable": True},
                "audio": {"present": False, "usable": False},
                "image": {"present": True, "usable": True},
            }
        }
        reason, required = main._required_modality_gate(quality_result, timed_out_modalities=["audio"])

        self.assertEqual(required, ["video", "audio"])
        self.assertEqual(reason, main.FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT)

    def test_required_modality_gate_allows_degraded_video_with_usable_evidence(self):
        signals = {
            "video": {"score": 0.65, "details": {"status": "ok", "duration_seconds": 5.0, "visual_quality_score": 0.65, "visual_warnings": ["video_too_dark"]}},
            "voice": {"score": 0.92, "details": {"status": "ok", "duration_seconds": 4.0, "audio_quality_score": 0.92, "audio_warnings": []}},
            "camera": {"score": 0.86, "details": {"status": "ok", "image_quality_score": 0.86, "image_warnings": []}},
        }
        assessed = main.assess_quality(signals)

        reason, required = main._required_modality_gate(assessed, timed_out_modalities=[])

        self.assertEqual(required, ["video", "audio", "image"])
        self.assertIsNone(reason)
        self.assertTrue(assessed["media_quality"]["video"]["usable"])
        self.assertIn("video_too_dark", assessed["warnings"])

    def test_required_modality_gate_still_blocks_missing_face_evidence(self):
        signals = {
            "video": {"score": 0.65, "details": {"status": "ok", "duration_seconds": 5.0, "visual_quality_score": 0.65, "visual_warnings": ["face_not_visible"]}},
            "voice": {"score": 0.92, "details": {"status": "ok", "duration_seconds": 4.0, "audio_quality_score": 0.92, "audio_warnings": []}},
            "camera": {"score": 0.86, "details": {"status": "ok", "image_quality_score": 0.86, "image_warnings": []}},
        }
        assessed = main.assess_quality(signals)

        reason, _required = main._required_modality_gate(assessed, timed_out_modalities=[])

        self.assertEqual(reason, main.FAILURE_REASON_LOW_QUALITY_MEDIA)
        self.assertFalse(assessed["media_quality"]["video"]["usable"])

    def test_required_modality_gate_reports_analyzer_error_as_technical_failure(self):
        quality_result = {
            "media_quality": {
                "video": {"present": False, "usable": False},
                "audio": {"present": True, "usable": True},
                "image": {"present": True, "usable": True},
            }
        }
        reason, _required = main._required_modality_gate(
            quality_result,
            timed_out_modalities=[],
            analyzer_error_modalities=["video"],
        )

        self.assertEqual(reason, main.FAILURE_REASON_ANALYSIS_EXCEPTION)

    def test_full_multimodal_result_requires_numeric_voice_confidence(self):
        result = {
            "voice_confidence": None,
            "fusion_details": {
                "signal_profiles": {
                    "video": {"present": True, "score": 0.8},
                    "audio": {"present": False, "score": None},
                    "image": {"present": True, "score": 0.9},
                }
            },
        }
        reason = main._required_full_multimodal_evidence_failure(
            result=result,
            worker_states={"audio": {"timed_out": False}},
            valid_modalities=["video", "image"],
        )

        self.assertEqual(reason, main.FAILURE_REASON_AUDIO_MISSING)

    def test_production_multimodal_schema_survives_validation_fusion_and_classification(self):
        analysis_results = _production_multimodal_results()
        worker_states = _successful_worker_states()
        media = main.Media(video="video.mp4", audio="audio.wav", image="image.jpg")

        validation_result = main.validate_scan_inputs(
            policy=main.VALIDATION_POLICY,
            media=media,
            video_result=analysis_results["video"],
            audio_result=analysis_results["audio"],
            image_result=analysis_results["image"],
            expected_phrase=None,
            transcript=None,
        )
        self.assertTrue(validation_result["passed"])

        raw_valid_modalities = [
            modality
            for modality, result in analysis_results.items()
            if main._raw_analysis_has_valid_evidence(modality, result, worker_states[modality])
        ]
        signals = {
            "video": analysis_results["video"],
            "voice": analysis_results["audio"],
            "camera": analysis_results["image"],
        }
        quality_result = assess_quality(signals, speech_required=False)
        result = compute_result(
            signals=signals,
            task=None,
            previous_confidence=None,
            baseline=None,
            baseline_used=False,
            quality=quality_result,
            ml_result={"confidence": 0.5},
        )
        valid_modalities = main._fused_result_valid_modalities(
            result,
            raw_valid_modalities=raw_valid_modalities,
        )

        self.assertEqual(raw_valid_modalities, ["video", "audio", "image"])
        self.assertEqual(valid_modalities, ["video", "audio", "image"])
        self.assertIsInstance(result["voice_confidence"], float)
        self.assertTrue(math.isfinite(result["voice_confidence"]))
        self.assertIsNone(
            main._required_full_multimodal_evidence_failure(
                result=result,
                worker_states=worker_states,
                valid_modalities=valid_modalities,
            )
        )

    def test_full_multimodal_gate_rejects_missing_timeout_error_and_noncanonical_audio(self):
        base_results = _production_multimodal_results()
        base_states = _successful_worker_states()
        signals = {
            "video": base_results["video"],
            "voice": base_results["audio"],
            "camera": base_results["image"],
        }
        fused = compute_result(
            signals=signals,
            task=None,
            previous_confidence=None,
            baseline=None,
            baseline_used=False,
            quality=assess_quality(signals, speech_required=False),
            ml_result=None,
        )

        negative_cases = []
        missing_results = _production_multimodal_results()
        missing_results["audio"] = {"score": None, "details": {"status": "missing"}}
        negative_cases.append(("missing", missing_results, _successful_worker_states(), main.FAILURE_REASON_AUDIO_MISSING))

        timeout_results = _production_multimodal_results()
        timeout_results["audio"] = main._analysis_worker_placeholder("audio")
        timeout_states = _successful_worker_states()
        timeout_states["audio"]["result_received"] = False
        timeout_states["audio"]["timed_out"] = True
        negative_cases.append(("timeout", timeout_results, timeout_states, main.FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT))

        error_results = _production_multimodal_results()
        error_results["audio"] = main._analysis_worker_placeholder("audio")
        error_states = _successful_worker_states()
        error_states["audio"]["result_received"] = False
        error_states["audio"]["analyzer_error"] = True
        negative_cases.append(("analyzer_error", error_results, error_states, main.FAILURE_REASON_AUDIO_MISSING))

        score_only_results = _production_multimodal_results()
        score_only_results["audio"]["details"].pop("audio_confidence")
        negative_cases.append(("score_only", score_only_results, _successful_worker_states(), main.FAILURE_REASON_AUDIO_MISSING))

        for label, analysis_results, worker_states, expected_reason in negative_cases:
            with self.subTest(label=label):
                raw_valid = [
                    modality
                    for modality, result in analysis_results.items()
                    if main._raw_analysis_has_valid_evidence(modality, result, worker_states[modality])
                ]
                valid_modalities = main._fused_result_valid_modalities(
                    fused,
                    raw_valid_modalities=raw_valid,
                )
                self.assertNotIn("audio", valid_modalities)
                self.assertEqual(
                    main._required_full_multimodal_evidence_failure(
                        result=fused,
                        worker_states=worker_states,
                        valid_modalities=valid_modalities,
                    ),
                    expected_reason,
                )

        no_voice_confidence = dict(fused)
        no_voice_confidence["voice_confidence"] = None
        self.assertEqual(
            main._required_full_multimodal_evidence_failure(
                result=no_voice_confidence,
                worker_states=base_states,
                valid_modalities=["video", "audio", "image"],
            ),
            main.FAILURE_REASON_AUDIO_MISSING,
        )

    def test_successful_audio_populates_voice_confidence_and_enters_fusion(self):
        signals = {
            "camera": _signal(0.92, {"status": "ok", "image_confidence": 0.92, "image_quality_score": 0.92, "face_detected": True, "avg_ear": 0.3}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.82, "duration_seconds": 5.0, "face_frames": 10, "sampled_frames": 12, "face_or_subject_visibility": 0.8}),
            "voice": _signal(0.67, {"status": "ok", "audio_confidence": 0.67, "audio_quality_score": 0.7, "duration_seconds": 3.0, "rms_energy": 0.03, "zero_crossing_rate": 0.08, "spectral_centroid": 1200.0, "silent": False}),
        }
        quality_result = assess_quality(signals, speech_required=False)
        result = compute_result(signals=signals, task=None, previous_confidence=None, baseline=None, baseline_used=False, quality=quality_result, ml_result=None)

        self.assertIsInstance(result["voice_confidence"], float)
        self.assertEqual(result["voice_confidence"], 0.67)
        self.assertEqual(result["voice_metrics"]["voice_score"], 0.67)
        self.assertTrue(result["fusion_details"]["signal_profiles"]["audio"]["present"])
        self.assertEqual(result["modality_scores"]["audio"], 0.67)

    def test_audio_feature_order_remains_model_contract(self):
        from ml.features import FEATURE_ORDER

        self.assertEqual(
            FEATURE_ORDER,
            [
                "camera_score",
                "face_detected",
                "avg_ear",
                "eyes_closed",
                "audio_score",
                "audio_energy",
                "audio_zcr",
                "audio_centroid",
                "audio_duration",
                "audio_silent",
                "video_score",
                "video_sway_std",
                "video_face_rate",
                "video_face_frames",
                "video_sampled_frames",
                "task_reaction_time",
                "task_errors",
                "task_present",
                "missing_camera",
                "missing_audio",
                "missing_video",
            ],
        )

    def test_audio_prewarm_structural_health_allows_quality_warnings(self):
        result = {
            "score": 0.05,
            "details": {
                "status": "ok",
                "audio_confidence": 0.05,
                "audio_quality_score": 0.1,
                "duration_seconds": 3.0,
                "rms_energy": 0.001,
                "spectral_centroid": 1200.0,
                "zero_crossing_rate": 0.02,
                "mfcc_summary": [0.0, 1.0, 2.0, 3.0, 4.0],
                "audio_warnings": ["speech_not_detected", "audio_too_quiet"],
            },
        }

        self.assertTrue(audio._audio_prewarm_result_valid(result))

    def test_audio_prewarm_structural_health_rejects_failed_analyzer_result(self):
        result = {
            "score": None,
            "details": {
                "status": "load_failed",
                "audio_warnings": ["audio_decode_failed"],
            },
        }

        self.assertFalse(audio._audio_prewarm_result_valid(result))

    def test_real_audio_prewarm_analyzer_runs_twice_and_deletes_temp_wav(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        if os.path.exists(path):
            os.remove(path)

        with unittest.mock.patch.object(audio.tempfile, "mkstemp", return_value=(os.open(path, os.O_CREAT | os.O_RDWR), path)):
            metrics = audio.prewarm_audio_analyzer()

        self.assertTrue(metrics["audio_warm_benchmark_passed"])
        self.assertTrue(metrics["audio_prewarm_temp_deleted"])
        self.assertFalse(os.path.exists(path))
        self.assertGreater(metrics["audio_prewarm_first_call_ms"], 0)
        self.assertGreaterEqual(metrics["audio_prewarm_second_call_ms"], 0)
        for key in ["audio_prewarm_first_result", "audio_prewarm_second_result"]:
            summary = metrics[key]
            self.assertEqual(summary["status"], "ok")
            self.assertIsInstance(summary["audio_warnings"], list)
            for numeric_key in [
                "audio_confidence",
                "duration_seconds",
                "rms_energy",
                "spectral_centroid",
                "zero_crossing_rate",
            ]:
                value = summary[numeric_key]
                self.assertIsInstance(value, (int, float))
                self.assertTrue(audio.math.isfinite(float(value)))
            self.assertEqual(summary["mfcc_length"], 5)

    def test_audio_prewarm_rejects_actual_failed_statuses(self):
        for status in ["missing", "empty_audio", "load_failed", "weak"]:
            result = {
                "score": None,
                "details": {
                    "status": status,
                    "audio_warnings": ["audio_decode_failed"],
                    "audio_confidence": 0.1,
                    "audio_quality_score": 0.1,
                    "duration_seconds": 3.0,
                    "rms_energy": 0.01,
                    "spectral_centroid": 1200.0,
                    "zero_crossing_rate": 0.01,
                    "mfcc_summary": [0.0] * 5,
                },
            }
            self.assertFalse(audio._audio_prewarm_result_valid(result), msg=status)

    def test_runtime_startup_allows_slow_audio_prewarm_under_derived_budget(self):
        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        runtime._supervisors = {
            "video": _SlowStartupSupervisor("video", 6.2, 15.0),
            "audio": _SlowStartupSupervisor("audio", 12.0, 30.0),
            "image": _SlowStartupSupervisor("image", 6.2, 15.0),
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()

        started = time.perf_counter()
        result = runtime.start()
        elapsed = time.perf_counter() - started

        self.assertGreaterEqual(runtime._startup_budget_seconds, 32.0)
        self.assertLess(elapsed, 20.0)
        self.assertGreaterEqual(elapsed, 18.0)
        self.assertTrue(runtime.is_ready())
        self.assertEqual(set(result["workers"].keys()), {"video", "audio", "image"})
        self.assertTrue(all(worker["ready"] for worker in runtime.health()["workers"].values()))
        self.assertEqual(main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS, 8.0)

    def test_runtime_startup_timeout_alignment_cleans_workers(self):
        fake_context = _FakeContext(behaviors=[
            {"invoke_target": False, "alive_after_start": True, "alive_after_target": True},
            {"invoke_target": False, "alive_after_start": True, "alive_after_target": True},
            {"invoke_target": False, "alive_after_start": True, "alive_after_target": True},
        ])
        runtime = analysis_runtime.WarmAnalyzerRuntime(context_factory=lambda: fake_context)
        runtime._supervisors = {
            "video": analysis_runtime.WorkerSupervisor("video", startup_timeout_seconds=0.05, recovery_timeout_seconds=0.01, context_factory=lambda: fake_context),
            "audio": analysis_runtime.WorkerSupervisor("audio", startup_timeout_seconds=0.10, recovery_timeout_seconds=0.01, context_factory=lambda: fake_context),
            "image": analysis_runtime.WorkerSupervisor("image", startup_timeout_seconds=0.05, recovery_timeout_seconds=0.01, context_factory=lambda: fake_context),
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", return_value=[]):
            with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError):
                runtime.start()

        self.assertFalse(runtime.is_ready())
        self.assertTrue(fake_context.processes)
        self.assertTrue(all(not process.is_alive() for process in fake_context.processes))
        self.assertTrue(all(process.terminated or process.killed for process in fake_context.processes))
        self.assertTrue(all(supervisor.health()["ready"] is False for supervisor in runtime._supervisors.values()))

    def test_runtime_startup_budget_derived_from_slowest_worker(self):
        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        runtime._supervisors = {
            "video": _SlowStartupSupervisor("video", 0.1, 15.0),
            "audio": _SlowStartupSupervisor("audio", 0.2, 30.0),
            "image": _SlowStartupSupervisor("image", 0.1, 15.0),
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()
        first_budget = runtime._startup_budget_seconds

        runtime.start()
        runtime.shutdown()
        runtime._stopped = False
        runtime._supervisors = {
            "video": _SlowStartupSupervisor("video", 0.01, 15.0),
            "audio": _SlowStartupSupervisor("audio", 0.02, 30.0),
            "image": _SlowStartupSupervisor("image", 0.01, 15.0),
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()
        second_budget = runtime._startup_budget_seconds
        runtime.start()

        self.assertEqual(first_budget, 32.0)
        self.assertEqual(second_budget, 32.0)
        self.assertTrue(runtime.is_ready())

    def test_runtime_deadline_cancels_running_startup_future_and_late_ready_is_rejected(self):
        release_ready = __import__("threading").Event()
        worker_entered = __import__("threading").Event()
        fake_context = _AsyncFakeContext(behaviors=[{"stay_alive_after_target": True}])

        def delayed_ready_worker(result_conn, analyzer_name, worker_generation):
            worker_entered.set()
            release_ready.wait(timeout=1.0)
            try:
                result_conn.send({
                    "type": "ready",
                    "ok": True,
                    "analyzer": analyzer_name,
                    "worker_generation": worker_generation,
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_import_ms": 1,
                        "total_worker_ms": 1,
                    },
                })
            except Exception:
                pass

        supervisor = analysis_runtime.WorkerSupervisor(
            "audio",
            startup_timeout_seconds=0.05,
            recovery_timeout_seconds=0.01,
            context_factory=lambda: fake_context,
            worker_entry=delayed_ready_worker,
        )
        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        runtime._supervisors = {"audio": supervisor}
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()

        def fake_wait(handles, timeout=None):
            time.sleep(min(float(timeout or 0.0), 0.01))
            return [handle for handle in handles if hasattr(handle, "poll") and handle.poll(0)]

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
            with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError):
                runtime.start()

        self.assertTrue(worker_entered.is_set())
        release_ready.set()
        time.sleep(0.05)
        self.assertFalse(runtime.is_ready())
        self.assertFalse(supervisor.health()["ready"])
        self.assertTrue(fake_context.processes)
        self.assertTrue(all(not process.is_alive() for process in fake_context.processes))
        self.assertTrue(all(process.terminated or process.killed for process in fake_context.processes))

        replacement = _SlowStartupSupervisor("audio", 0.01, 5.0)
        runtime._stopped = False
        runtime._supervisors = {"audio": replacement}
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()
        runtime.start()
        self.assertTrue(runtime.is_ready())

    def test_supervisor_start_cancel_token_rejects_late_ready_without_publishing(self):
        release_ready = __import__("threading").Event()
        fake_context = _AsyncFakeContext(behaviors=[{"stay_alive_after_target": True}])

        def delayed_ready_worker(result_conn, analyzer_name, worker_generation):
            release_ready.wait(timeout=1.0)
            try:
                result_conn.send({
                    "type": "ready",
                    "ok": True,
                    "analyzer": analyzer_name,
                    "worker_generation": worker_generation,
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_import_ms": 1,
                        "total_worker_ms": 1,
                    },
                })
            except Exception:
                pass

        supervisor = analysis_runtime.WorkerSupervisor(
            "audio",
            startup_timeout_seconds=5.0,
            recovery_timeout_seconds=0.01,
            context_factory=lambda: fake_context,
            worker_entry=delayed_ready_worker,
        )
        token = analysis_runtime.StartupCancelToken()
        future_holder = {}

        import concurrent.futures as _futures

        with _futures.ThreadPoolExecutor(max_workers=1) as executor:
            def fake_wait(handles, timeout=None):
                time.sleep(min(float(timeout or 0.0), 0.01))
                return [handle for handle in handles if hasattr(handle, "poll") and handle.poll(0)]

            with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
                future = executor.submit(supervisor.start, startup_cancel_token=token)
                time.sleep(0.02)
                future_holder["cancel_returned"] = future.cancel()
                token.cancel()
                release_ready.set()
                with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError):
                    future.result(timeout=2.0)

        self.assertFalse(future_holder["cancel_returned"])
        self.assertFalse(supervisor.health()["ready"])
        self.assertTrue(fake_context.processes)
        self.assertTrue(all(not process.is_alive() for process in fake_context.processes))

    def test_liveness_is_200_while_readiness_is_503_during_startup(self):
        class StartingRuntime:
            def is_ready(self):
                return False

        with unittest.mock.patch.object(main, "_directus_processing_ready", return_value=True), unittest.mock.patch.object(
            main, "get_analyzer_runtime", return_value=StartingRuntime()
        ):
            live = main.health()
            ready = main.readiness()

        self.assertEqual(live["status"], "alive")
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(json.loads(ready.body), {"ready": False})

    def test_process_returns_503_before_runtime_ready_without_marking_processing(self):
        class StartingRuntime:
            def __init__(self):
                self.start_background_calls = 0

            def is_ready(self):
                return False

            def start_background(self):
                self.start_background_calls += 1

        runtime = StartingRuntime()
        with unittest.mock.patch.object(main, "get_analyzer_runtime", return_value=runtime), unittest.mock.patch.object(
            main, "_authenticate_process_user", return_value="user-1"
        ) as auth_mock, unittest.mock.patch.object(
            main, "_resolve_scan_auth_context", return_value={"status": main.SCAN_STATUS_MEDIA_READY}
        ), unittest.mock.patch.object(
            main, "_authorize_scan_access"
        ), unittest.mock.patch.object(
            main, "_ensure_scan_media_ready"
        ), unittest.mock.patch.object(main.directus, "update_wellness_scan") as update_mock:
            response = main.process_scan(main.ScanRequest(scan_id="scan-123"), main.BackgroundTasks(), authorization="Bearer token")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body)["error"], "analyzer_runtime_not_ready")
        self.assertEqual(runtime.start_background_calls, 1)
        auth_mock.assert_called_once()
        update_mock.assert_not_called()

    def test_runtime_starts_audio_before_video_and_image_concurrently(self):
        events = []

        class OrderedSupervisor(_SlowStartupSupervisor):
            def start(self, *, startup_cancel_token=None):
                events.append(("start", self.name, time.perf_counter()))
                result = super().start(startup_cancel_token=startup_cancel_token)
                events.append(("ready", self.name, time.perf_counter()))
                return result

        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        runtime._supervisors = {
            "video": OrderedSupervisor("video", 0.05, 15.0),
            "audio": OrderedSupervisor("audio", 0.05, 90.0),
            "image": OrderedSupervisor("image", 0.05, 15.0),
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()

        runtime.start()

        audio_ready_at = next(ts for event, name, ts in events if event == "ready" and name == "audio")
        video_start_at = next(ts for event, name, ts in events if event == "start" and name == "video")
        image_start_at = next(ts for event, name, ts in events if event == "start" and name == "image")
        self.assertGreaterEqual(video_start_at, audio_ready_at)
        self.assertGreaterEqual(image_start_at, audio_ready_at)
        self.assertLess(abs(video_start_at - image_start_at), 0.04)
        self.assertTrue(runtime.is_ready())

    def test_audio_startup_progress_is_accepted_until_ready(self):
        fake_context = _AsyncFakeContext(behaviors=[{"stay_alive_after_target": True}])

        def progress_then_ready_worker(result_conn, analyzer_name, worker_generation):
            for stage in ["import_started", "import_completed", "first_call_started"]:
                result_conn.send({
                    "type": "startup_progress",
                    "analyzer": analyzer_name,
                    "worker_generation": worker_generation,
                    "stage": stage,
                    "elapsed_ms": 1,
                })
                time.sleep(0.01)
            result_conn.send({
                "type": "ready",
                "ok": True,
                "analyzer": analyzer_name,
                "worker_generation": worker_generation,
                "metrics": {"child_entry_ms": 1, "analyzer_import_ms": 1, "total_worker_ms": 1},
            })

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(
                analysis_runtime,
                "_wait_handles",
                side_effect=lambda handles, timeout=None: [handle for handle in handles if hasattr(handle, "poll") and handle.poll()] or (time.sleep(min(float(timeout or 0.0), 0.01)) or []),
            ):
                supervisor = analysis_runtime.WorkerSupervisor("audio", worker_entry=progress_then_ready_worker)
                result = supervisor.start()

        self.assertEqual(result["spawn_metrics"]["last_startup_stage"], "first_call_started")
        self.assertTrue(supervisor.health()["ready"])

    def test_audio_startup_inactivity_timeout_reports_last_stage(self):
        fake_context = _FakeContext(behaviors=[{"alive_after_target": True}])

        def progress_only_worker(result_conn, analyzer_name, worker_generation):
            result_conn.send({
                "type": "startup_progress",
                "analyzer": analyzer_name,
                "worker_generation": worker_generation,
                "stage": "first_call_started",
                "elapsed_ms": 1,
            })

        def fake_wait(handles, timeout=None):
            ready = [handle for handle in handles if hasattr(handle, "poll") and handle.poll()]
            if ready:
                return ready
            time.sleep(min(float(timeout or 0.0), 0.01))
            return []

        with unittest.mock.patch.object(analysis_runtime.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
                supervisor = analysis_runtime.WorkerSupervisor(
                    "audio",
                    startup_timeout_seconds=1.0,
                    startup_progress_inactivity_timeout_seconds=0.02,
                    recovery_timeout_seconds=0.01,
                    worker_entry=progress_only_worker,
                )
                with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError) as raised:
                    supervisor.start()

        self.assertIn("startup_inactivity_timeout", str(raised.exception))
        self.assertEqual(supervisor.health()["last_startup_stage"], "first_call_started")
        self.assertFalse(supervisor.health()["ready"])

    def test_background_startup_retries_are_serialized_and_bounded(self):
        attempts = []

        class TokenAwareFailingSupervisor(_SlowStartupSupervisor):
            def start(self, *, startup_cancel_token=None):
                attempts.append((self.name, startup_cancel_token, time.perf_counter()))
                if startup_cancel_token is not None:
                    startup_cancel_token.cancel()
                raise analysis_runtime.AnalysisRuntimeStartupError(
                    f"{self.name}_worker_ready_timeout",
                    analyzer_name=self.name,
                    terminal_reason=f"{self.name}_worker_ready_timeout",
                )

        class FailingRuntime(analysis_runtime.WarmAnalyzerRuntime):
            def _build_supervisors(self):
                return {
                    "video": _SlowStartupSupervisor("video", 0.0, 15.0),
                    "audio": TokenAwareFailingSupervisor("audio", 0.0, 90.0),
                    "image": _SlowStartupSupervisor("image", 0.0, 15.0),
                }

            def _cleanup_workers(self):
                for supervisor in self._supervisors.values():
                    supervisor.shutdown()
                self._reset_supervisors()

        runtime = FailingRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        with unittest.mock.patch.object(analysis_runtime, "BACKGROUND_STARTUP_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01)):
            runtime.start_background()
            runtime._startup_thread.join(timeout=2.0)

        self.assertEqual(len(attempts), analysis_runtime.MAX_BACKGROUND_STARTUP_ATTEMPTS)
        self.assertEqual(len({id(token) for _, token, _ in attempts}), analysis_runtime.MAX_BACKGROUND_STARTUP_ATTEMPTS)
        self.assertTrue(all(token.is_cancelled() for _, token, _ in attempts))
        self.assertEqual(runtime.health()["runtime_state"], "failed")
        self.assertTrue(runtime.health()["final_failure_latched"])
        self.assertFalse(runtime.is_ready())

    def test_retry_uses_fresh_token_after_cancelled_attempt_and_then_succeeds(self):
        tokens = []

        class AudioFailOnceSupervisor(_SlowStartupSupervisor):
            calls = 0

            def start(self, *, startup_cancel_token=None):
                self.__class__.calls += 1
                tokens.append(startup_cancel_token)
                if self.__class__.calls == 1:
                    self.ready = False
                    if startup_cancel_token is not None:
                        startup_cancel_token.cancel()
                    raise analysis_runtime.AnalysisRuntimeStartupError(
                        "audio_worker_ready_timeout",
                        analyzer_name="audio",
                        terminal_reason="audio_worker_ready_timeout",
                    )
                return super().start(startup_cancel_token=startup_cancel_token)

        class RetryRuntime(analysis_runtime.WarmAnalyzerRuntime):
            def _build_supervisors(self):
                return {
                    "video": _SlowStartupSupervisor("video", 0.0, 15.0),
                    "audio": AudioFailOnceSupervisor("audio", 0.0, 90.0),
                    "image": _SlowStartupSupervisor("image", 0.0, 15.0),
                }

            def _cleanup_workers(self):
                for supervisor in self._supervisors.values():
                    supervisor.shutdown()
                self._reset_supervisors()

        AudioFailOnceSupervisor.calls = 0
        runtime = RetryRuntime()
        with unittest.mock.patch.object(analysis_runtime, "BACKGROUND_STARTUP_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01)):
            runtime.start_background()
            runtime._startup_thread.join(timeout=2.0)

        self.assertTrue(runtime.is_ready())
        self.assertEqual(AudioFailOnceSupervisor.calls, 2)
        self.assertEqual(runtime.health()["startup_attempts"], 2)
        self.assertEqual(len(tokens), 2)
        self.assertIsNot(tokens[0], tokens[1])
        self.assertTrue(tokens[0].is_cancelled())
        self.assertFalse(tokens[1].is_cancelled())
        self.assertFalse(runtime.health()["final_failure_latched"])

    def test_final_failure_latch_blocks_restarts_and_repeated_process_requests(self):
        start_calls = []

        class AlwaysFailSupervisor(_SlowStartupSupervisor):
            def start(self, *, startup_cancel_token=None):
                start_calls.append((self.name, startup_cancel_token))
                if startup_cancel_token is not None:
                    startup_cancel_token.cancel()
                raise analysis_runtime.AnalysisRuntimeStartupError(
                    f"{self.name}_worker_ready_timeout",
                    analyzer_name=self.name,
                    terminal_reason=f"{self.name}_worker_ready_timeout",
                )

        class LatchingRuntime(analysis_runtime.WarmAnalyzerRuntime):
            def _build_supervisors(self):
                return {
                    "video": _SlowStartupSupervisor("video", 0.0, 15.0),
                    "audio": AlwaysFailSupervisor("audio", 0.0, 90.0),
                    "image": _SlowStartupSupervisor("image", 0.0, 15.0),
                }

            def _cleanup_workers(self):
                for supervisor in self._supervisors.values():
                    supervisor.shutdown()
                self._reset_supervisors()

        runtime = LatchingRuntime()
        with unittest.mock.patch.object(analysis_runtime, "BACKGROUND_STARTUP_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01)):
            runtime.start_background()
            runtime._startup_thread.join(timeout=2.0)

        first_thread = runtime._startup_thread
        attempts_after_failure = runtime.health()["startup_attempts"]
        calls_after_failure = len(start_calls)
        for _ in range(3):
            runtime.start_background()
        with unittest.mock.patch.object(main, "get_analyzer_runtime", return_value=runtime), unittest.mock.patch.object(
            main, "_authenticate_process_user", return_value="user-1"
        ) as auth_mock, unittest.mock.patch.object(
            main, "_resolve_scan_auth_context", return_value={"status": main.SCAN_STATUS_MEDIA_READY}
        ), unittest.mock.patch.object(
            main, "_authorize_scan_access"
        ), unittest.mock.patch.object(
            main, "_ensure_scan_media_ready"
        ), unittest.mock.patch.object(main.directus, "update_wellness_scan") as update_mock:
            for index in range(3):
                response = main.process_scan(main.ScanRequest(scan_id=f"scan-{index}"), main.BackgroundTasks(), authorization="Bearer token")
                self.assertEqual(response.status_code, 503)

        self.assertIs(runtime._startup_thread, first_thread)
        self.assertFalse(runtime.health()["startup_thread_alive"])
        self.assertEqual(runtime.health()["startup_attempts"], attempts_after_failure)
        self.assertEqual(len(start_calls), calls_after_failure)
        self.assertTrue(runtime.health()["final_failure_latched"])
        self.assertEqual(runtime.readiness()["runtime_state"], "failed")
        self.assertFalse(runtime.readiness()["ready"])
        self.assertEqual(auth_mock.call_count, 3)
        update_mock.assert_not_called()

    def test_shutdown_interrupts_retry_wait_and_joins_startup_thread(self):
        entered = __import__("threading").Event()

        class WaitingRuntime(analysis_runtime.WarmAnalyzerRuntime):
            def start(self, *, startup_cancel_token=None):
                entered.set()
                raise analysis_runtime.AnalysisRuntimeStartupError("audio_worker_ready_timeout")

            def _cleanup_workers(self):
                pass

        runtime = WaitingRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": analysis_runtime._smoke_worker_main,
            "image": analysis_runtime._smoke_worker_main,
        })
        with unittest.mock.patch.object(analysis_runtime, "BACKGROUND_STARTUP_RETRY_DELAYS_SECONDS", (5.0, 5.0, 5.0)):
            runtime.start_background()
            self.assertTrue(entered.wait(timeout=1.0))
            runtime.shutdown()

        self.assertFalse(runtime.health()["startup_thread_alive"])

    def test_shutdown_terminates_active_audio_startup_candidate(self):
        entered = __import__("threading").Event()
        release = __import__("threading").Event()
        fake_context = _AsyncFakeContext(behaviors=[{"stay_alive_after_target": True}])

        def delayed_prewarm_worker(result_conn, analyzer_name, worker_generation):
            result_conn.send({
                "type": "startup_progress",
                "analyzer": analyzer_name,
                "worker_generation": worker_generation,
                "stage": "first_call_started",
                "elapsed_ms": 1,
            })
            entered.set()
            release.wait(timeout=5.0)
            try:
                result_conn.send({
                    "type": "ready",
                    "ok": True,
                    "analyzer": analyzer_name,
                    "worker_generation": worker_generation,
                    "metrics": {"child_entry_ms": 1, "analyzer_import_ms": 1, "total_worker_ms": 1},
                })
            except Exception:
                pass

        runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
            "video": analysis_runtime._smoke_worker_main,
            "audio": delayed_prewarm_worker,
            "image": analysis_runtime._smoke_worker_main,
        })
        runtime._supervisors = {
            "audio": analysis_runtime.WorkerSupervisor(
                "audio",
                startup_timeout_seconds=5.0,
                startup_progress_inactivity_timeout_seconds=5.0,
                recovery_timeout_seconds=0.01,
                context_factory=lambda: fake_context,
                worker_entry=delayed_prewarm_worker,
            )
        }
        runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()

        def fake_wait(handles, timeout=None):
            ready = [handle for handle in handles if hasattr(handle, "poll") and handle.poll(0)]
            if ready:
                return ready
            time.sleep(min(float(timeout or 0.0), 0.01))
            return []

        with unittest.mock.patch.object(analysis_runtime, "_wait_handles", side_effect=fake_wait):
            runtime.start_background()
            self.assertTrue(entered.wait(timeout=1.0))
            runtime.shutdown()

        release.set()
        self.assertFalse(runtime.health()["startup_thread_alive"])
        self.assertTrue(fake_context.processes)
        self.assertTrue(all(not process.is_alive() for process in fake_context.processes))
        self.assertTrue(all(process.terminated or process.killed for process in fake_context.processes))
        self.assertFalse(runtime.is_ready())

    def test_video_and_image_startup_failures_are_attributed_to_real_analyzer(self):
        class SelectiveFailSupervisor(_SlowStartupSupervisor):
            def __init__(self, name, fail_name):
                super().__init__(name, 0.0, 15.0 if name != "audio" else 90.0)
                self.fail_name = fail_name

            def start(self, *, startup_cancel_token=None):
                if self.name == self.fail_name:
                    if startup_cancel_token is not None:
                        startup_cancel_token.cancel()
                    raise analysis_runtime.AnalysisRuntimeStartupError(
                        f"{self.name}_worker_ready_timeout",
                        analyzer_name=self.name,
                        terminal_reason=f"{self.name}_worker_ready_timeout",
                    )
                return super().start(startup_cancel_token=startup_cancel_token)

        for failing_analyzer in ("video", "image"):
            runtime = analysis_runtime.WarmAnalyzerRuntime(worker_entry_map={
                "video": analysis_runtime._smoke_worker_main,
                "audio": analysis_runtime._smoke_worker_main,
                "image": analysis_runtime._smoke_worker_main,
            })
            runtime._supervisors = {
                "video": SelectiveFailSupervisor("video", failing_analyzer),
                "audio": SelectiveFailSupervisor("audio", failing_analyzer),
                "image": SelectiveFailSupervisor("image", failing_analyzer),
            }
            runtime._startup_budget_seconds = runtime._derive_startup_budget_seconds()
            with self.assertRaises(analysis_runtime.AnalysisRuntimeStartupError) as raised:
                runtime.start()
            self.assertEqual(raised.exception.analyzer_name, failing_analyzer)
            self.assertEqual(runtime.health()["terminal_reason"], f"{failing_analyzer}_worker_ready_timeout")



if __name__ == "__main__":
    unittest.main()
