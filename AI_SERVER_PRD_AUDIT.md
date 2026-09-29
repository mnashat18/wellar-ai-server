# Wellar AI AI Server — PRD vs Actual Implementation

Audit date: 2026-08-16  
Scope: static, read-only audit of the repository at `D:\flutter\last\ai-server`. No model was downloaded, trained, modified, or executed. No benchmark was run. The standalone PRD attachment was not present in the workspace; the v1.1 claims quoted in the audit request are the comparison baseline.

Status legend: ✅ IMPLEMENTED · PARTIAL · ❌ MISSING · CHANGED · EXTRA · ⚪ DEAD / UNUSED · ❓ UNCLEAR

## 1. Executive Summary

The current server is a Directus-integrated, background FastAPI analysis service, not a synchronous three-file inference API. The production `/process` request contains only a `scan_id` and bearer token. The server authenticates the Directus user and membership, loads the scan, media references, task data, baseline, and schema from Directus, returns HTTP 202, performs analysis in background, and writes the detailed result back to Directus (`main.py`, `ScanRequest`, `process_scan`, `_process_scan_sync`, `_write_success`).

The product has three main evidence families, but they do not match the PRD descriptions exactly:

- PARTIAL Face: it analyzes both a thumbnail and a video with OpenCV and MediaPipe Face Mesh. It measures image/video quality, face visibility, eye aspect ratio (EAR), eye closure, eye asymmetry, and camera motion. It does not measure gaze/eye movement or blink rate. “Head stability” is actually whole-frame pixel-difference camera/motion stability; “alertness” and “eye strain” are not independent model outputs (`vision.py`, `_analyze_image_array`; `video.py`, `analyze_video`).
- PARTIAL Voice: it extracts energy, silence, clipping, zero-crossing rate, spectral centroid/flatness, five MFCC summary values, tonal concentration, RMS variation, heuristic speech presence, clarity, and quality. Speech rate and pitch stability are explicitly `None`; no tremor feature is calculated. “Fatigue” is a downstream rule based mainly on usable-speech presence, energy, silence, and personal energy drift, not a dedicated cognitive-fatigue voice model (`audio.py`, `_feature_pipeline`, `_build_success_details`; `scoring.py`, `_audio_fatigue_signal`).
- PARTIAL Reaction/focus: Directus supplies `reaction_time`, `errors`, and `attempts`. A simple piecewise score is calculated. There is no task engine, tap sequence, coordination metric, processing-speed model, or reaction baseline in this repository (`main.py`, `Task`, `_merge_task`; `scoring.py`, `compute_task_score`; `baseline.py`, `REACTION_FEATURES`).

CHANGED Readiness remains an integer 0–100, but it is not a fixed face/voice/reaction weighted average. It uses adaptive quality-weighted fusion of video (base 45%), audio (35%), thumbnail image (15%), and task (5%), optional local-ML blending, quality penalties, baseline drift penalties, and eye-closure caps (`scoring.py`, `BASE_WEIGHTS`, `_adaptive_weights`, `compute_result`).

CHANGED Personalization activates after three eligible scans, not seven. Only thumbnail eye aperture, eye asymmetry, and voice energy are personalized. Reaction is not baselined. Eligible samples are stored in Directus `employee_baselines`, capped at nine samples per feature, and summarized with median/MAD (`config.py`, baseline constants; `baseline.py`, `baseline_signal_payload`, `baseline_ready_for_personalized_scoring`).

The four risk concepts still exist, but actual API/DB values are snake_case: `stable`, `low_focus`, `elevated_fatigue`, `high_risk`. Classification is a rule tree that includes quality/reliability, fatigue evidence, confidence, eye closure, and readiness—not four simple score bands (`scoring.py`, `VALID_RISK_LEVELS`, `_risk_level`).

Privacy is PARTIAL. Downloaded media and converted WAV files are normally deleted in `finally` blocks. Downloads use unpredictable OS temporary names and a 20 MB cap. There is no scheduled “within 24 hours” sweeper, and deferred cleanup of a download still running after the scan deadline depends on a future callback. Directus retains the original media outside this service, so end-to-end retention cannot be verified here (`utils.py`, `download_temp_file`, `remove_temp_file`; `main.py`, `_process_scan_sync`, `_defer_temp_file_cleanup`).

## 2. AI Architecture

### Real architecture

```text
Angular / Flutter client
        |
        | POST /process {scan_id} + Bearer token
        v
FastAPI app (main.py)
        |
        +--> Directus authentication, ownership, membership, scan state
        +--> HTTP 202 accepted
        |
        v
FastAPI BackgroundTask -> _process_scan_sync
        |
        +--> Directus: scan context, media IDs, task, prior result, baseline
        +--> parallel bounded media downloads -> OS temp files
        +--> optional ffmpeg conversion for audio validation/transcription
        |
        v
WarmAnalyzerRuntime (persistent supervised child processes)
        +--> video.py: OpenCV + MediaPipe Face Mesh
        +--> audio.py: WAV/ffmpeg decode + librosa/NumPy
        +--> vision.py: OpenCV + MediaPipe Face Mesh thumbnail
        |
        +--> quality.py + validation.py
        +--> optional Whisper phrase transcription/validation
        +--> ml/features.py -> optional local PyTorch classifier
        +--> scoring.py deterministic fusion/classification/explanation/action
        +--> baseline.py eligibility/personalization update
        |
        v
Directus writeback
        +--> scan_results
        +--> wellness_scans / member / scan_request
        +--> employee_baselines
        +--> high-risk alert + notification dispatch
        |
        v
delete service-created temp files
```

### Component map

| Area | Actual component and evidence |
|---|---|
| Application entry point | `main.py`, global `app = FastAPI()` (line 42). `Dockerfile` starts `uvicorn main:app` on `${PORT:-8000}`. |
| Routers | ⚪ No `APIRouter` modules or `include_router`; all routes attach directly to `app` in `main.py`. |
| Schemas | `main.py`: `Media`, `Task`, `ScanRequest`, `BaselineRequest`, `ProcessResponse`, `ScanResultResponse`, `BaselineStatusResponse`. Only selected routes declare response models. |
| Runtime/process isolation | `analysis_runtime.py`: `WorkerSupervisor`, `WarmAnalyzerRuntime`; `analysis_worker.py`: `worker_main`, dynamic analyzer import, audio prewarm, structured result/error messages. |
| Face/image | `vision.py`: `analyze_face`, `_analyze_image_array`. |
| Face/video | `video.py`: `analyze_video`, sampling, quality, motion, Face Mesh/EAR. |
| Voice | `audio.py`: `analyze_audio`, `_feature_pipeline`, optional `transcribe_audio`; Whisper is used only for phrase transcription. |
| Validation/quality | `quality.py`: `assess_quality`; `validation.py`: `ValidationPolicy`, `validate_scan_inputs` and modality validators. |
| Scoring/classification | `scoring.py`: `compute_task_score`, `compute_result`, `_risk_level`, `_confidence_from_profiles`. |
| Local ML | `ml/features.py`, `ml/model.py`, `ml/runtime.py`; optional file `models/latest.pt`. No model artifact is present in the audited tree, so the default runtime falls back to deterministic scoring because `REQUIRE_LOCAL_MODEL=False` (`config.py`; `ml/runtime.py`, `MLRuntime.load/predict`). |
| Baseline | `baseline.py`; persistent storage through Directus `employee_baselines`; schema foundation in `sql/2026_07_01_phase2_baseline_foundation.sql`. |
| External AI | OpenAI Whisper package is local, lazily loaded for transcription (`audio.py`, `_load_whisper_model`, `transcribe_audio`). No hosted LLM or external inference API was found. |
| Storage/backend | `directus_client.py`, `DirectusClient`; REST reads/writes to Directus. No local application database. |
| Temporary files | `utils.py`, `download_temp_file/remove_temp_file`; `main.py`, `_convert_audio_to_wav`, `_process_scan_sync`. |
| Logging | `logger.py`, `get_logger`; structured event/performance logging throughout `main.py`, runtime, worker, and audio. |
| Error handling | FastAPI request errors plus structured `ProcessingError`, worker placeholders/timeouts, validation failures, and Directus terminal writeback (`main.py`; `analysis_runtime.py`; `validation.py`). |

## 3. API Inventory

| Method | Endpoint | Purpose | Input | Output | Main Handler | Reachability/status |
|---|---|---|---|---|---|---|
| GET | `/` | Basic liveness | None | `{"status":"ok"}` | `main.py:root` | ✅ Active, minimal |
| GET | `/health` | Process liveness | None | `{"ok":true,"status":"alive"}` | `main.py:health` | ✅ Active; does not prove analyzers/models are ready |
| GET | `/ready` | Analyzer/model/config readiness | None | Model/Directus/validation/runtime health; 503 if analyzer runtime not ready | `main.py:readiness` | ✅ Active |
| GET | `/readiness` | Alias for `/ready` | None | Same as `/ready` | `main.py:readiness_alias` | CHANGED Duplicate compatibility alias |
| GET | `/debug/scan/{scan_id}` | Inspect IDs/status/baseline count | Path `scan_id` | Directus-derived diagnostic fields | `main.py:debug_scan` | ⚪ Disabled by default; registered only in dev/test plus `DEBUG_SCAN_ENDPOINT_ENABLED=true` (lines 79–82, 3296–3297). No explicit route auth in handler. |
| GET | `/baseline/status` | Read baseline state | Required query `member_id`, `business_profile_id` | `BaselineStatusResponse` plus Pydantic filtering | `main.py:baseline_status` | ✅ Reachable when Directus configured; ❓ no bearer auth/ownership check is visible on this endpoint |
| POST | `/baseline` | Manually analyze an existing owned Directus scan and add eligible sample | JSON `scan_id`, optional `media`, `manually_unreliable`; Bearer header | Baseline record/status and model version | `main.py:set_baseline` | ✅ Active; rejects arbitrary media differing from Directus references |
| POST | `/process` | Queue production scan | JSON `{scan_id}`; unknown JSON fields ignored; Bearer header | Small ack/error envelope, normally HTTP 202 `accepted` | `main.py:process_scan` | ✅ Primary production endpoint; detailed result is written to Directus, not returned |

There are no independent public face, voice, reaction/focus, scoring, or full-result retrieval endpoints. The “full scan” is `/process`, but its media and task inputs are indirect through Directus (`main.py`, `ScanRequest`, `_resolve_scan_context`, `_merge_media`, `_merge_task`). `ProcessResponse` and `ScanResultResponse` exist but are not attached as `/process` response models, so they are PARTIAL/declarative rather than authoritative route contracts (`main.py`, lines 246–268, 3420–3547).

## 4. Face Analysis

The server has two visual modalities: required thumbnail image (`camera`) and required video (`video`) under default policy. This is CHANGED from a single conceptual “Face Analysis.”

| Signal | Actual algorithm/model | Input → output | Threshold/calculation | Used in final scoring? |
|---|---|---|---|---|
| Face detection/visibility | MediaPipe Face Mesh, `max_num_faces=1`, detection 0.5; video tracking 0.5 | Thumbnail/video frames → face boolean/count/rate | Thumbnail visibility 1.0 when any face; video `face_frames / sampled_frames` | Yes, indirectly in modality confidence, validation, quality, and evidence gate (`vision.py:_analyze_image_array`; `video.py:analyze_video`) |
| Eye aperture | Six MediaPipe landmarks per eye; standard EAR geometry | Landmarks → left/right/average EAR | Thumbnail closed if average EAR `<0.18`; video per-sample closed if `<=0.16` | Yes. Thumbnail EAR supplies baseline; video EAR supplies fatigue evidence (`vision.py:_eye_aspect_ratio`; `video.py:_eye_aspect_ratio`; `scoring.py:_video_fatigue_signal`) |
| Sustained eye closure | Contiguous burst samples with reliability gates | Video burst EAR/timestamps → boolean, ratio, streak, window | Requires reliable landmarks, closed ratio `>=0.75`, streak `>=4`, average EAR `<=0.14`, EAR std `<=0.025`, duration 0.3–1.2 s | Yes. Deducts 0.14 fused score, caps readiness at 30, and forces `elevated_fatigue` if otherwise reliable (`video.py`, lines 838–870; `scoring.py:compute_result`, `_risk_level`) |
| Eye asymmetry | Absolute left EAR − right EAR | Thumbnail/video landmarks → asymmetry | Video reliability uses maximum asymmetry constant 0.08; thumbnail asymmetry is baselined | Yes for personalization; video reports it but primary baseline reads thumbnail (`video.py:_eye_landmark_reliability`; `baseline.py:current_baseline_features`) |
| Blink rate | No blink event counter or time-normalized blink calculation | — | — | ❌ MISSING |
| Eye movement/gaze | No iris/gaze displacement feature; `refine_landmarks=False` | — | — | ❌ MISSING |
| Head stability | No head pose, yaw/pitch/roll, or facial centroid trajectory | Whole-frame grayscale differences → `motion_stability_score`, `sway_std` | Warn `unstable_camera` below 0.4 | PARTIAL: camera/scene motion, not specifically head stability (`video.py:_motion_stability_from_diffs`, `analyze_video`) |
| Alertness | No separately trained alertness output | Eye/video/audio heuristics → downstream fatigue evidence | Rule-based only | PARTIAL/CHANGED (`scoring.py:_fatigue_signal_context`) |
| Eye strain | No eye-strain model or metric | — | — | ❌ MISSING |
| Visual quality | OpenCV mean brightness, Laplacian variance, resolution, usable frames | Image/video → normalized scores/warnings | Image dark mean `<75`; image sharpness `<0.35`; video dark mean `<75`; blur variance `<65`; duration `<1.5 s`; validation defaults are stricter at video `<3 s` and quality `<0.5` | Yes, strongly: score penalties, adaptive weights, validation, confidence (`vision.py`; `video.py`; `validation.py:ValidationPolicy`; `scoring.py`) |

Multiple-face behavior is PARTIAL: `max_num_faces=1` means at most one face is processed, but the code does not detect or reject multiple faces (`vision.py`, detector construction; `video.py`, detector construction).

## 5. Voice Analysis

The analyzer decodes a bounded mono signal (native WAV or ffmpeg fallback), resamples as needed, and uses NumPy/librosa. It is a handcrafted acoustic-quality/speech-presence pipeline, not a voice-emotion or clinical fatigue model (`audio.py`, `_decode_audio_once`, `_feature_pipeline`).

| Feature/output | Actual calculation | Product use |
|---|---|---|
| Duration/sample rate | Decoded sample count/sample rate; bounded analysis window | Quality and short-audio warning |
| RMS energy and RMS variation | Frame RMS mean and coefficient-like `std/mean` | Speech presence, clarity, quality, silence, fatigue rule, baseline energy |
| Peak/clipping | Max absolute sample and proportion `abs(sample)>=threshold` | Quality/headroom and `audio_clipping` |
| Silence ratio | Fraction of frames below adaptive energy threshold | Quality, speech state, fatigue rule |
| ZCR | Frame sign-change rate | Noise estimate and ML feature |
| Spectral centroid | Librosa magnitude centroid mean | Speech presence and ML feature |
| Spectral flatness | Librosa flatness mean | Noise/tonality heuristic |
| MFCC summary | Five MFCC row means | Returned/internal feature only; not used by deterministic scoring or current ML feature vector |
| Noise estimate | `0.65*flatness + 0.35*ZCR` (special-cased near silence) | Quality, warnings, clarity |
| Tonal concentration | Dominant FFT concentration + inverse flatness + low RMS variation | Rejects tone-like input as no speech |
| Speech presence | Handcrafted activity/energy/centroid/silence/cleanliness/variation/tonality formula | Voice clarity and usable-speech gate |
| Voice clarity | `speech_presence * (0.35 + 0.25*energy + 0.40*cleanliness)` | Audio confidence |
| Audio quality | `0.24*duration + 0.22*level + 0.20*activity + 0.19*noise + 0.15*headroom` | Audio confidence, validation, adaptive fusion |
| Audio confidence | `0.45*audio_quality + 0.55*voice_clarity` | Primary voice modality score |
| Speech transcription | Local Whisper, lazily loaded | Optional/required phrase validation only; not acoustic score or explanation generation (`audio.py:transcribe_audio`; `main.py:_transcribe_audio_file_optional`) |
| Speech pace/rate | Explicitly `speech_rate: None` | ❌ MISSING (`audio.py:_build_success_details`; `baseline.py:current_baseline_features`) |
| Pitch stability/tremor | Pitch extraction is explicitly skipped and `pitch_stability_score: None` | ❌ MISSING (`audio.py:_feature_pipeline`, timing `pitch_ms=0`; `_build_success_details`) |
| Stress | No stress model/output | ❌ MISSING |
| Cognitive fatigue | Downstream heuristic from usable speech, presence, RMS energy, silence, and optional energy drift | PARTIAL/CHANGED; not directly measured (`scoring.py:_audio_fatigue_signal`) |

Important thresholds in `_speech_state_and_warnings`: minimum analyzer duration 1.5 s; no-speech when silence `>0.80` or tone-like/weak evidence; too much silence `>0.55`; noisy `>0.72`; clipping above `MAX_CLIPPING_RATIO`. Default validation additionally expects at least 2.0 s and audio quality 0.5 (`audio.py`; `validation.py:ValidationPolicy`).

## 6. Reaction / Focus Analysis

CHANGED The repository contains no client-side task implementation. It expects task results already stored in Directus and merges only:

- `reaction_time: float | None`
- `errors: int | None`
- `attempts: int | None`

Evidence: `main.py:Task`, `_merge_task`; `scoring.py:compute_task_score`.

Actual scoring requires positive integer attempts. It averages the available components:

- attempts: 0.95 for `>=3`, otherwise 0.60;
- reaction time: 0.92 at `<=0.55 s`; 0.72 at `<=0.85`; 0.45 at `<=1.15`; otherwise 0.22;
- errors: 0.95 for 0; 0.75 for 1; 0.50 for 2–3; 0.20 for 4+.

Malformed/missing attempts invalidate the whole task score. Reaction time and errors are optional once attempts are valid. Task contributes only a 5% base fusion weight before adaptive renormalization (`scoring.py:compute_task_score`, `BASE_WEIGHTS`).

| PRD claim | Status | Actual |
|---|---|---|
| Reaction time | ✅ IMPLEMENTED | One supplied scalar, piecewise scored |
| Accuracy | PARTIAL | Error count only; no target/correct count or percentage |
| Processing speed | PARTIAL | Reaction time is a proxy; no separate metric |
| Coordination | ❌ MISSING | No coordinate/timing sequence analysis |
| Focus/cognitive/tap task engine | ❌ MISSING | No such endpoint/module; only stored summary fields |
| Reaction personalization | ❌ MISSING | `REACTION_FEATURES=[]`, `reaction_avg={}` (`baseline.py`) |

Searches for focus/cognitive/tap/reaction terminology found the task fields and scoring/tests, but no separate task algorithm. Therefore the implementation has not merely been renamed; the task execution lives outside this server or is unavailable here (❓ backend/client implementation not provable from this repo).

## 7. Multimodal Processing Flow

1. `/process` validates `scan_id`, bearer token, active Directus user, scan ownership, active business membership, scan status, required media IDs, and warm analyzer readiness (`main.py:process_scan`, `_authenticate_process_user`, `_authorize_scan_access`, `_ensure_scan_media_ready`).
2. It marks the Directus scan `processing`, schedules `process_scan_background`, and returns HTTP 202 (`main.py:process_scan`).
3. Background processing loads the complete scan context, media, task, baseline row(s), previous result, expected phrase, and identifiers (`main.py:_process_scan_sync`).
4. Image/audio/video assets download concurrently with `ThreadPoolExecutor(max_workers=3)` and a wall deadline; temporary paths use `tempfile.mkstemp` (`main.py`, lines 2695–2740; `utils.py:download_temp_file`).
5. Non-WAV audio may be converted by ffmpeg to a 3-second, mono, 16 kHz WAV for validation/transcription (`main.py:_convert_audio_to_wav`). The primary audio analyzer also has its own native/ffmpeg decode path (`audio.py:_decode_audio_once`).
6. Video, audio, and image run in parallel through three persistent supervised child analyzers with a scan deadline (`analysis_runtime.py:WarmAnalyzerRuntime.run_scan`; `analysis_worker.py:worker_main`).
7. `quality.assess_quality` and `validation.validate_scan_inputs` evaluate modality evidence. Required full multimodal evidence is fail-closed in production defaults (`main.py:_required_modality_gate`, `_required_full_multimodal_evidence_failure`).
8. Phrase transcription runs optionally with a timeout. Default `REQUIRE_PHRASE_MATCH=false`, but an expected phrase can still generate non-blocking validation warnings (`main.py:_transcribe_audio_file_optional`; `validation.py:validate_phrase_result`).
9. Features are built and the optional local PyTorch model predicts. If the model is optional and absent, deterministic scoring proceeds; if configured as required, failure is terminal (`main.py`, ML stage; `ml/runtime.py`).
10. `compute_result` fuses signals, personalizes when eligible, classifies, and creates deterministic explanation/action (`scoring.py:compute_result`).
11. A result and related state are written to Directus; eligible baselines are updated. High-risk results may create alerts and notifications (`main.py:_write_success`, `_dispatch_high_risk_notifications`).
12. Service-created temp files are deleted in `finally` (`main.py:_process_scan_sync`, lines 3221–3225).

Concurrency/robustness facts:

- The FastAPI route and background handler are synchronous `def` functions. FastAPI executes sync work in its threadpool, while analysis uses child processes (`main.py`; `analysis_runtime.py`).
- Media downloads are parallel; modality analyses are parallel. Scoring/validation/writeback are sequential.
- There is no modality retry. Worker supervision can restart unhealthy workers, but a scan timeout/error yields a missing/error placeholder and, because video/audio/image are required by default, typically a terminal failed scan (`analysis_runtime.py`; `main.py:_required_full_multimodal_evidence_failure`).
- Directus calls used for critical state/writeback have selected three-attempt retry logic (`main.py:_retry_directus_call`); media analysis itself is not retried.
- Missing optional task produces no task score and remaining weights renormalize. Missing required video/audio/image fails before a valid result is persisted under default policy.
- In-process duplicate scan work is guarded by `_ACTIVE_SCAN_IDS` plus Directus scan-state idempotency (`main.py:_claim_active_scan`, `process_scan`). This lock is process-local, so cross-instance concurrency relies on backend state and is not an atomic distributed claim (risk).

## 8. Readiness Scoring

### Actual Readiness Scoring Formula

1. Build raw modality scores:
   - video = `visual_confidence`;
   - audio = `audio_confidence`;
   - image = `image_confidence`;
   - task = `compute_task_score`.
2. For video/audio/image, subtract warning penalties (cumulative, capped 0.42) plus a low-quality penalty when quality `<0.5`; total modality deduction is capped 0.55 (`scoring.py:WARNING_SCORE_PENALTIES`, `_degraded_signal_score`).
3. Start with base weights video 0.45, audio 0.35, image 0.15, task 0.05. For each present score, multiply its base weight by `max(0.05, quality - min(0.12*warning_count, 0.45))`, then renormalize to sum to 1 (`scoring.py:BASE_WEIGHTS`, `_adaptive_weights`).
4. Compute the weighted sum. If local ML returned a valid “confidence” (actually probability-weighted readiness), blend `0.84*fused + 0.16*ML_readiness` (`scoring.py:compute_result`; `ml/runtime.py:MLRuntime.predict`). The ML class label itself is not used for final classification.
5. Subtract global quality-warning penalty `min(0.025*count,0.16)`, at least 0.12 for low-quality failure, 0.20 for missing-media failure, or 0.06 for a weak scan (`scoring.py:compute_result`).
6. If personalized baseline flags exist, subtract 0.03 per drift flag. A flag means the current measurement is at least 2.5 MAD below its median, using MAD floor 0.02 (`scoring.py:_baseline_drift`, `compute_result`).
7. If sustained eye closure is fully confirmed, subtract 0.14 and later cap readiness at 30 (`scoring.py:compute_result`).
8. Clamp fused value to 0–1 and round `100*fused` to an integer. Thus ✅ the stored readiness score remains 0–100 (`scoring.py:compute_result`).
9. If the scan is unreliable—failed/retake quality, missing video or audio, confidence `<0.45`, or certain weak low scores—cap readiness at 51. Missing-media failure caps at 35; low-quality failure caps at 45 (`scoring.py:_invalid_scan_outcome`, `compute_result`).

Face confidence reported to consumers is separate from the fusion: `0.4*image + 0.6*video` when both exist. Voice confidence is the adjusted audio score. Task performance is `round(task_score*100)` (`scoring.py:compute_result`).

### Confidence

Confidence is a reliability score, not class probability:

`0.32*fused + 0.22*quality_multiplier + 0.18*modality_coverage + agreement_bonus + 0.06(if baseline) + 0.05*ML_readiness - missing_major_penalty - warning_penalty - conflict_penalty`.

Coverage excludes task and is available visual/audio/image count divided by three. Missing one of video/audio costs 0.18; missing both costs 0.32. Warning penalty is 0.03 each, capped 0.18. Agreement bonus is at most 0.18; conflict cost is at most 0.25. Confidence ceilings are 0.98 normally, 0.55 for one modality, 0.35 for image-only, 0.78 without both major modalities, 0.72 for weak quality, and 0.68 for high conflict. Invalid scans are capped below 0.44 (`scoring.py:_agreement_factor`, `_confidence_from_profiles`, `_invalid_scan_outcome`).

## 9. Risk Classification

CHANGED Values are snake_case in code/Directus. Title-case labels are only compatibility aliases (`main.py:SCAN_RESULT_CHOICE_ALIASES`). There is no Python enum; validation uses sets and fallback logic (`scoring.py:VALID_RISK_LEVELS`, `compute_result`).

Rules are evaluated top-to-bottom:

| Classification | Actual Threshold/Condition | Code Location |
|---|---|---|
| `elevated_fatigue` | Confirmed sustained eye closure, regardless of numeric bands (only after scan reliability handling) | `scoring.py:_risk_level`, lines 721–723 |
| `low_focus` | Quality-limited scan: failed/retake/missing major modality/confidence `<0.45` | `scoring.py:_quality_limited_scan`, `_risk_level` |
| `high_risk` | Fatigue evidence `>=0.82` and confidence `>=0.5` | `scoring.py:_risk_level`, lines 726–727 |
| `high_risk` | Otherwise readiness `<35` and confidence `>=0.62` | `scoring.py:_risk_level`, lines 728–729 |
| `elevated_fatigue` | Fatigue evidence `>=0.5` | `scoring.py:_risk_level`, lines 730–731 |
| `low_focus` | Weak quality and readiness `<52` without baseline flags | `scoring.py:_risk_level`, lines 732–733 |
| `elevated_fatigue` | Readiness `<52` and confidence `>=0.5` | `scoring.py:_risk_level`, lines 734–735 |
| `low_focus` | Readiness `<68` | `scoring.py:_risk_level`, lines 736–737 |
| `stable` | All remaining cases (effectively readiness `>=68` after earlier gates) | `scoring.py:_risk_level`, line 738 |

Edge cases:

- A bad/missing scan is deliberately `low_focus`, not `high_risk`, and requires rescan.
- Confirmed sustained eye closure is capped at `elevated_fatigue`; it alone cannot create `high_risk` (`scoring.py:_risk_level`).
- An unknown computed label is forced to `low_focus`, though normal branches only return valid values (`scoring.py:compute_result`).
- SQL allows legacy `unknown`, but runtime never emits it (`sql/2026_07_01_phase2_baseline_foundation.sql`, risk check; `scoring.py`).

## 10. Confidence / Explanation / Recommendation

| PRD output | Status | Real behavior/evidence |
|---|---|---|
| Readiness score | ✅ IMPLEMENTED | Integer 0–100, deterministic hybrid formula (`scoring.py:compute_result`) |
| Risk classification | ✅ IMPLEMENTED / CHANGED | Four concepts remain; snake_case and rule-tree semantics (`scoring.py:_risk_level`) |
| Confidence | ✅ IMPLEMENTED / CHANGED | Deterministic reliability formula, not model confidence (`scoring.py:_confidence_from_profiles`) |
| Plain-language explanation | ✅ IMPLEMENTED | Hardcoded rule/template composition from warnings, quality, fatigue, risk, and baseline notes; sanitized to 500 chars (`scoring.py:_explanation`) |
| Suggested action | ✅ IMPLEMENTED | Hardcoded rule mapping: `manager_review`, `rescan_recommended`, `rest_advised`, `review_required`, `continue_normal_activity` (`scoring.py:_suggested_action`) |

No LLM generates explanations or recommendations. Whisper transcribes an expected phrase only. Explanation language is deterministic for identical inputs; this is tested in `tests/test_scoring_unit.py`, including determinism, length, path leakage, and wording tests.

## 11. Personal Baseline

CHANGED The PRD’s first-seven-scans design is obsolete.

- Provisional after 2 eligible scans; active/use after 3; “high confidence” after 5; keep at most 9 per-feature samples (`config.py`, `BASELINE_*`).
- Only eligible scans count. Eligibility requires passed/non-weak capture, no retake, required features, no critical validation error, aggregate capture quality and reliability each `>=0.65`, result confidence `>=0.60`, no blocklisted warning, stable risk, completed required speech/task, and not manually unreliable (`baseline.py:evaluate_baseline_eligibility`).
- Features are thumbnail `avg_ear`, thumbnail left/right EAR asymmetry, and voice RMS energy. `speech_rate` is retained as a retired `None` field. Reaction features are empty (`baseline.py:FACE_FEATURES`, `VOICE_FEATURES`, `REACTION_FEATURES`, `current_baseline_features`).
- Storage uses robust schema v2 feature sample lists and median/MAD, not simple face/voice/reaction averages despite field names `face_avg`, `voice_avg`, `reaction_avg` (`baseline.py:_build_feature_payload`, `baseline_signal_payload`).
- Personalization uses an active, unique Directus baseline row with valid references and a high-quality current scan (`baseline.py:baseline_ready_for_personalized_scoring`).
- A negative drift flag occurs at z `<=-2.5` MAD (floor 0.02). Flags reduce fused readiness 0.03 each and contribute bounded fatigue boosts. The implementation treats lower eye aperture, lower asymmetry, and lower voice energy uniformly as “below baseline”; lower asymmetry being a fatigue direction is scientifically questionable but is the literal code (`scoring.py:_baseline_drift`, `compute_result`).
- Before activation or when current quality gates fail, generic deterministic/optional-ML scoring is used (`main.py:_process_scan_sync`; `scoring.py:compute_result`).
- Baselines live in the main backend database through Directus, not `BASELINE_PATH`; the config file path is legacy/unused (`main.py:_baseline_for_member`, `_baseline_rows_for_member`; `directus_client.py`; `config.py:BASELINE_PATH`).

There is an inconsistency to resolve: the 2026 SQL migration backfill labels provisional at 3 and active at 5, while current Python config activates at 3 and marks provisional at 2 (`sql/2026_07_01_phase2_baseline_foundation.sql`, lines 42–55; `config.py`, lines 32–35). Runtime `baseline_status_payload` recomputes from Python count constants but also requires stored `is_active`, so migrated legacy rows may behave differently until rewritten.

## 12. Media Privacy & Cleanup

### Face Media Retention

Classification: PARTIAL.

- Directus image/video assets are streamed into OS-created files via `tempfile.mkstemp`; suffixes are fixed but names are random (`utils.py:download_temp_file`).
- Downloads enforce same-origin Directus URL checks and a 20,000,000-byte default maximum (`utils.py:download_temp_file`; `config.py:MAX_DOWNLOAD_BYTES`).
- `_process_scan_sync` tracks temporary image/video paths and removes them in `finally` (`main.py`, lines 2689, 2738–2740, 3221–3225).
- `/baseline` also removes analyzer-created temporary files in `finally` (`main.py:set_baseline`, lines 3415–3417).
- If a download outlives the wall deadline, `_cleanup_download_future`/`_defer_temp_file_cleanup` attaches later cleanup. That is best effort and process-lifetime dependent (`main.py`, lines 697–716, 2707–2719).
- No scheduled sweeper or 24-hour cleanup job exists. Original media remains in Directus; retention there is outside this repository.
- Logs include scan IDs, asset/media kind, durations, statuses, and performance, but the reviewed main flow does not log raw frame data. Some errors may log exception text; exact production handler destination/retention is environment-dependent (`logger.py`; `main.py`).

Conclusion: service-created face media has strong normal/exception cleanup, but end-to-end “deleted after processing/within 24 hours” is not fully provable and backend originals are not deleted here.

### Voice Media Retention

Classification: PARTIAL.

- Downloaded audio and converted WAV paths are tracked and deleted in the same production `finally` (`main.py:_process_scan_sync`). `_convert_audio_to_wav` deletes failed/empty outputs immediately (`main.py`, lines 1469–1516).
- `audio.prewarm_audio_analyzer` creates deterministic temporary test audio and has its own cleanup path; it is runtime initialization, not user media (`audio.py:prewarm_audio_analyzer`).
- Whisper model state persists in memory, but raw audio is not intentionally persisted by the audio module (`audio.py:_load_whisper_model`, `transcribe_audio`).
- As above, no 24-hour sweeper exists and Directus source audio is not deleted by this service.

Conclusion: local temporary voice copies are normally removed, but whole-system retention is CANNOT VERIFY; under the requested four labels, overall voice compliance is PARTIAL.

## 13. Error Handling

| Failure Case | Current Behavior | HTTP Status / persisted state | Safe? |
|---|---|---|---|
| Missing/blank `scan_id` | Pydantic rejects missing/empty before handler; handler also has blank check | 422 | Yes |
| Missing/malformed bearer | Reject before scan lookup | 401 `invalid_authorization` | Yes (`main.py:_authenticate_process_user`) |
| Foreign scan | Hidden as not found | 404 | Yes (`main.py:_authorize_scan_access`) |
| Missing membership | Reject | 403 | Yes |
| Missing required media IDs | Reject before queue | 409 `scan_media_not_ready` | Yes |
| Corrupted/unreadable video | `open_failed`/missing placeholder; required evidence gate fails | `/process` already returned 202; Directus scan later `failed` | Fail-closed (`video.py:analyze_video`; `main.py`) |
| Corrupted/unreadable image | `invalid_image` placeholder; required evidence gate fails | Async failed state | Fail-closed (`vision.py:analyze_face`) |
| Corrupted audio | `load_failed`/decode failure placeholder or worker error | Async failed state | Fail-closed (`audio.py:analyze_audio`; runtime) |
| No face | Warnings `face_not_visible`/`subject_not_visible`; default full evidence/quality may fail depending evidence | Usually scored weak or failed by evidence gates | PARTIAL; does not distinguish occlusion vs detector absence |
| Multiple faces | Only first face processed; no warning/rejection | Normal processing | No—identity ambiguity is not handled |
| Silent/no-speech audio | `speech_not_detected`, `too_much_silence`; unusable voice; required audio evidence normally fails | Async failed/retake | Yes/fail-closed |
| Short audio | Analyzer warns below 1.5 s; validation warns below default 2.0 s | Weak/failed depending required gates | Yes |
| Malformed reaction data | Pydantic applies only if direct object constructed; Directus merge coercion and scorer reject invalid attempts; task omitted | Scan can proceed without task | Mostly safe; task is only 5% and optional |
| Analyzer/model timeout | Structured timeout placeholder; worker finalized/restarted | Required modality causes async failure | Yes/fail-closed (`analysis_runtime.py`; `main.py`) |
| Optional local model absent | ML result `None`; deterministic scoring continues | Normal | Intended (`REQUIRE_LOCAL_MODEL=False`) |
| Required local model absent | Readiness/model stage fails | 503 readiness and/or async terminal failure | Yes |
| Unsupported format | Video/image open failure; audio uses WAV then ffmpeg fallback | Async failed | Mostly safe; no explicit MIME allowlist |
| Download too large | Stops above `MAX_DOWNLOAD_BYTES` | Async download/missing-media failure | Yes (`utils.py:download_temp_file`) |
| Internal exception | Background wrapper logs error and marks scan terminal failed | Client already has 202; state in Directus | Fail-closed, but client must poll backend |
| Directus write failure | Retries selected calls; terminal failure/recovery logic | `/process` already 202; recovery or `writeback_failed` | PARTIAL; distributed partial-write risk remains |

## 14. Performance / Reliability

- ✅ Models/analyzers are not intentionally loaded per scan. Warm child processes persist; audio is prewarmed twice and Whisper is lazily cached (`analysis_runtime.py`, `analysis_worker.py`, `audio.py`).
- ✅ Video samples a bounded plan (maximum eight planned coverage/burst samples in current constants) rather than decoding the entire video (`video.py:MAX_SAMPLED_FRAMES`, `_build_sample_plan`).
- ✅ Audio is bounded and downloads are capped at 20 MB (`audio.py:MAX_AUDIO_ANALYSIS_SEC/MAX_AUDIO_SAMPLES`; `config.py:MAX_DOWNLOAD_BYTES`).
- ✅ Downloads and analyzers are parallel and have wall deadlines/timeouts (`main.py`; `analysis_runtime.py`).
- Risk: `/process` uses FastAPI in-process `BackgroundTasks`. A server restart after returning 202 can lose queued work; there is no durable job queue in this repo (`main.py:process_scan`).
- Risk: sync Directus requests, subprocess calls, and background work consume process/thread resources. Child analyzers mitigate CPU/GIL contention, but capacity depends on deployment worker count (`main.py`, `directus_client.py`).
- Risk: each application process creates its own warm analyzer runtime and in-memory active-scan set. Multiple Uvicorn/Gunicorn workers multiply memory/model use and weaken duplicate protection (`main.py`; `analysis_runtime.py`).
- Risk: Whisper may be large and is loaded only when phrase transcription is attempted. There is a timeout wrapper, but a timed-out thread cannot forcibly stop native model computation immediately (`main.py:_transcribe_audio_file_optional`; `audio.py:_load_whisper_model`).
- Risk: ffmpeg conversion has a 3.5 s timeout while overall deadlines are tight; cold process/model startup is handled at readiness but can make deployment startup expensive (`main.py:_convert_audio_to_wav`; runtime prewarm).
- Risk: no explicit request-body upload limit is needed for `/process` because it accepts only a small ID, but `/baseline` accepts strings and can expose large response data from Directus. Media size is controlled on download.
- ✅ Temp file collisions are unlikely because `mkstemp` is atomic (`utils.py`; `main.py`).
- ❓ No repository benchmark establishes PRD latency. Performance logs exist (`[PERF]`, `[WORKER_PERF]`, audio timing), but static inspection cannot validate targets.

## 15. Tests

Tests are extensive and predominantly `unittest` with mocks/synthetic arrays. Present suites:

- `tests/test_vision_unit.py`: image/EAR/quality and invalid-image behavior.
- `tests/test_video_unit.py`: sampling, video metrics, eye-closure/reliability, malformed inputs using mocked OpenCV/MediaPipe objects.
- `tests/test_audio_unit.py`: decode/features/quality/speech states/prewarm with generated or mocked audio.
- `tests/test_scoring_unit.py`: task, weights, penalties, confidence, baseline drift, fatigue, risk/action invariants, explanation, serialization, missing signals.
- `tests/test_baseline_unit.py`: baseline counts, robust feature payloads, eligibility, personalization gates.
- `tests/test_quality_unit.py`, `tests/test_validation_unit.py`, root `test_validation.py`: quality and failure policy; root suite also exercises FastAPI/Directus orchestration heavily with mocks.
- `test_pipeline.py`: broad runtime, worker, scoring, baseline, Directus, and integration-style mocked pipeline coverage.
- `tests/test_config.py`, `tests/test_logger.py`, `tests/test_utils.py`: configuration, logging, downloads/sanitization/temp helpers.

No tests were executed in this audit, per the static/read-only constraint. Important gaps/limitations:

- No checked-in real model artifact and no end-to-end real-model accuracy/calibration test.
- Most media and Directus flows are mocked/synthetic; no evidence of a representative labeled fatigue dataset or clinical validity suite.
- No load/concurrency/latency benchmark or multi-instance idempotency test.
- No full end-to-end test proving Directus originals and local temp files are deleted across process crash/restart; helper cleanup paths are tested, but 24-hour retention is not.
- No multiple-face rejection test because the behavior is not implemented.
- No gaze, blink-rate, pace, tremor, stress, or coordination tests because those features are absent.
- `test_process.ps1` is a manual process helper rather than automated product validation (⚪ test-only support).

## 16. Extra Features Outside PRD

| Status | Extra feature | Evidence |
|---|---|---|
| EXTRA | Separate thumbnail quality/face modality in addition to video | `vision.py`; `main.py:_merge_media`; `scoring.py:BASE_WEIGHTS` |
| EXTRA | Media quality validation: lighting, blur, resolution, clipping, noise, silence, face visibility, usable frames | `quality.py`; `validation.py`; analyzers |
| EXTRA | Optional spoken-phrase verification with local Whisper and normalized sequence matching | `audio.py:transcribe_audio`; `validation.py:phrase_match_score` |
| EXTRA | Optional local PyTorch multimodal classifier blended into readiness | `ml/*`; `scoring.py:compute_result` |
| EXTRA | Robust median/MAD personalization with quality eligibility | `baseline.py` |
| EXTRA | Observed/fatigue evidence score separate from readiness | `scoring.py:_fatigue_signal_context`, `compute_result` |
| EXTRA | Analyzer readiness, prewarm, process supervision, restart, structured timing | `analysis_runtime.py`; `analysis_worker.py`; `/ready` |
| EXTRA | High-risk alert creation and notification dispatch | `main.py:_dispatch_high_risk_notifications`, `_write_success` |
| EXTRA | Ownership/membership authorization and scan-state idempotency | `main.py:_authenticate_process_user`, `_authorize_scan_access`, `process_scan` |
| EXTRA | Schema-aware Directus payload filtering/choice aliasing | `main.py:_schema_aware_scan_result_payload`; `directus_client.py` |

No liveness/anti-spoofing model, anomaly detector, hosted LLM explanation, emotion recognition, or gaze model was found.

## 17. Dead / Legacy Code

| Status | Item | Evidence/reason |
|---|---|---|
| ⚪ DEAD / UNUSED | Legacy fixed weights `WEIGHT_CAMERA`, `WEIGHT_VIDEO`, `WEIGHT_VOICE`, `WEIGHT_TASK`, `ML_WEIGHT`, `MISSING_MEDIA_PENALTY` | Defined as “Legacy compatibility” in `config.py`; active scoring imports none of them and uses `scoring.py:BASE_WEIGHTS` |
| ⚪ DEAD / UNUSED | `READINESS_FACE_WEIGHT`, `READINESS_VOICE_WEIGHT`, `READINESS_REACTION_WEIGHT`, `READINESS_ML_BLEND` | Defined in `config.py`; no active scoring references |
| ⚪ DEAD / UNUSED | `BASELINE_PATH`, `BASELINE_ALPHA`, `BASELINE_DRIFT_THRESHOLD/PENALTY/STD_MULTIPLIER` | Directus + median/MAD implementation supersedes local JSON/older drift settings |
| ⚪ DEAD / UNUSED | `speech_rate` baseline slot | Explicitly retired and always `None` (`audio.py`, `baseline.py`, `scoring.py`) |
| ⚪ DEAD / UNUSED | Reaction baseline payload | `REACTION_FEATURES=[]`, always `{}` (`baseline.py`) |
| ⚪ DEAD / UNUSED | `/debug/scan` in production defaults | Conditional registration only (`main.py`) |
| ⚪ DEAD / UNUSED | `ProcessResponse`, `ScanResultResponse` as declared response contracts | No route uses them as `response_model`; actual `/process` envelope is built manually |
| ⚪ DEAD / UNUSED | `audio.analyze_audio_worker` legacy queue wrapper | Active architecture loads `analyze_audio` through `analysis_worker._load_analyzer_callable`; wrapper is not on that path |
| ⚪ DEAD / UNUSED | Training utilities as runtime functionality | `ml/train.py` is offline tooling and not imported by server path; do not describe it as online learning |
| CHANGED | SQL baseline count defaults | Migration’s 3/5 provisional/active values conflict with current Python 2/3 settings; migration is historical, not the current rule |

Some helpers in `main.py` (including older direct process-finalization functions) coexist with the newer `WarmAnalyzerRuntime`. Static reachability shows the production analysis call is through `get_analyzer_runtime().run_scan`; legacy helpers should not be treated as separate product endpoints (`main.py:_run_parallel_analysis/_analyze_media`; `analysis_runtime.py`).

## 18. PRD Feature Matrix

The counts below use each row as one auditable PRD/product claim. CHANGED is used instead of also double-counting the same row as implemented/partial.

| # | Claim | Status | Actual implementation/evidence |
|---:|---|---|---|
| 1 | Face analysis exists | ✅ IMPLEMENTED | Thumbnail + video analyzers (`vision.py`, `video.py`) |
| 2 | Eye movement | ❌ MISSING | No gaze/iris movement feature |
| 3 | Blink rate | ❌ MISSING | No blink event/rate logic |
| 4 | Head stability | PARTIAL | Whole-frame camera/motion stability, not head pose (`video.py`) |
| 5 | Alertness | PARTIAL | Rule-based fatigue inference, no alertness output (`scoring.py`) |
| 6 | Eye strain | ❌ MISSING | No implementation |
| 7 | Voice analysis exists | ✅ IMPLEMENTED | Acoustic pipeline (`audio.py`) |
| 8 | Speech clarity | ✅ IMPLEMENTED | Heuristic `voice_clarity_score` (`audio.py`) |
| 9 | Speech pace | ❌ MISSING | `speech_rate=None` |
| 10 | Voice tremor | ❌ MISSING | pitch skipped, stability `None` |
| 11 | Cognitive fatigue from voice | PARTIAL | Energy/presence/silence heuristic only (`scoring.py`) |
| 12 | Voice stress | ❌ MISSING | No stress output/model |
| 13 | Reaction time | ✅ IMPLEMENTED | Supplied scalar and piecewise score |
| 14 | Reaction accuracy | PARTIAL | Error count only |
| 15 | Processing speed | PARTIAL | Reaction-time proxy only |
| 16 | Coordination | ❌ MISSING | No metric/task engine |
| 17 | Readiness 0–100 | ✅ IMPLEMENTED | Integer clamped/ranged result (`scoring.py`) |
| 18 | Fixed PRD component formula | CHANGED | Adaptive four-channel fusion + optional ML/penalties |
| 19 | Stable class | ✅ IMPLEMENTED | `stable` rule outcome |
| 20 | Low Focus class | ✅ IMPLEMENTED | `low_focus` rule outcome |
| 21 | Elevated Fatigue class | ✅ IMPLEMENTED | `elevated_fatigue` rule outcome |
| 22 | High Risk class | ✅ IMPLEMENTED | `high_risk` rule outcome |
| 23 | Simple threshold bands/names | CHANGED | Snake_case rule tree, confidence/quality/fatigue conditions |
| 24 | Confidence | ✅ IMPLEMENTED | Reliability formula |
| 25 | Plain explanation | ✅ IMPLEMENTED | Deterministic template/rules |
| 26 | Suggested action | ✅ IMPLEMENTED | Deterministic action mapping |
| 27 | First 7 scans build baseline | CHANGED | Active after 3 eligible scans |
| 28 | Face baseline average | CHANGED | Median/MAD of EAR and asymmetry, not generic average |
| 29 | Voice baseline average | CHANGED | Median/MAD of RMS energy only; speech rate retired |
| 30 | Reaction baseline average | ❌ MISSING | Empty reaction feature set |
| 31 | Baseline scan count | ✅ IMPLEMENTED | Eligible/scan count in Directus |
| 32 | Scan 8+ personalized | CHANGED | Scan 4+ can personalize when all gates pass |
| 33 | Generic fallback before baseline | ✅ IMPLEMENTED | Deterministic/optional local ML scoring |
| 34 | Raw media not retained by AI service | PARTIAL | Normal `finally` cleanup; no crash/scheduled sweeper; Directus originals remain |
| 35 | Delete after processing/within 24h | PARTIAL | Immediate best-effort cleanup only; no 24-hour job |
| 36 | Three-modality processing | CHANGED | Video + voice + thumbnail + optional task (four fusion channels) |
| 37 | Synchronous scan response | CHANGED | 202 background ack; results written to Directus |
| 38 | LLM explanation/recommendation | ❌ MISSING | Not used; deterministic behavior instead |
| 39 | Separate quality validation | EXTRA | `quality.py`, `validation.py` |
| 40 | Phrase verification | EXTRA | Whisper + phrase match |
| 41 | Robust baseline eligibility/median-MAD | EXTRA | `baseline.py` |
| 42 | Optional local PyTorch fusion | EXTRA | `ml/*`, scoring blend |
| 43 | Analyzer supervision/readiness | EXTRA | runtime/worker and `/ready` |
| 44 | High-risk backend alerts | EXTRA | `main.py:_dispatch_high_risk_notifications` |
| 45 | Legacy fixed scoring/config paths | ⚪ DEAD / UNUSED | Compatibility constants and retired features |

Totals: Implemented 14 · Partial 7 · Missing 9 · Changed 8 · Extra 6 · Dead/Unused 1. These are matrix claim counts, not source-file counts.

## 19. Recommended PRD Changes

### MARK AS COMPLETED

- Document real thumbnail/video/audio analyzers, readiness 0–100, four risk outcomes, reliability confidence, deterministic explanation, deterministic suggested action, task summary scoring, and Directus-backed baseline count.
- Document `/health`, `/ready`, `/readiness`, `/process`, `/baseline`, and `/baseline/status` with their real authentication and async semantics.

### CHANGE DESCRIPTION

- Replace “Face Analysis” with two inputs: thumbnail image quality/face/EAR and sampled face video quality/motion/EAR/closure.
- Replace eye movement/blink/head stability claims with exact EAR, sustained-closure, visibility, and whole-frame motion behavior.
- Replace voice pace/tremor/stress claims with the actual acoustic feature list and state explicitly that speech rate and pitch stability are not calculated.
- Describe readiness as adaptive quality-weighted fusion with base weights 45/35/15/5, optional 16% ML readiness blend, penalties/caps, and personalization—not a fixed three-score average.
- Replace score-band risk definitions with the ordered rule table in section 9 and use snake_case API values.
- Replace “first 7 scans / scan 8+” with “three eligible samples to activate; personalization may start on the next qualifying scan; five yields high baseline confidence; up to nine samples retained per feature.”
- Replace average baseline language with median/MAD for thumbnail EAR, eye asymmetry, and voice RMS energy. Explicitly exclude reaction and speech rate.
- Describe `/process` as authenticated, scan-ID-only, asynchronous 202 plus Directus writeback.
- Describe privacy narrowly: temporary AI-server copies are deleted best effort after processing; original Directus asset retention is a separate backend policy.

### ADD TO PRD

- Media-quality and evidence gates, supported warnings, retake semantics, and mandatory default modalities.
- Thumbnail as a separate 15% base-weight channel and task as an optional 5% channel.
- Optional phrase verification with local Whisper, including when it is blocking.
- Optional local PyTorch model behavior, missing-model fallback, model version, and the distinction between ML readiness value and result confidence.
- Baseline eligibility requirements, robust statistics, duplicate-row handling, and per-scan baseline-use gates.
- Analyzer warmup/readiness, timeout behavior, background processing, Directus state transitions, and partial-write recovery.
- High-risk alert/notification integration and its failure-isolation behavior.
- Explicit non-diagnostic language: these handcrafted signals are not validated clinical measures of fatigue, stress, alertness, or cognitive impairment based on evidence in this repository.

### STILL MISSING

- Gaze/eye movement, blink rate, true head pose/stability, eye strain.
- Speech rate/pace, pitch/tremor, stress, validated cognitive-fatigue voice model.
- Coordination and a server-side focus/reaction task protocol.
- Reaction personalization.
- Multiple-face rejection/identity ambiguity handling.
- Durable job queue/cross-instance claim, documented latency benchmark, representative model validation/calibration, and crash-safe temp cleanup/sweeper.
- End-to-end proof that backend original face/voice assets meet the 24-hour retention requirement.

### REMOVE / OBSOLETE

- Seven-scan baseline and “scan 8+” language.
- Claims that simple face/voice/reaction averages are the active formula.
- Claims that every named PRD biometric is extracted merely because a broad face/voice module exists.
- Any claim that explanations/recommendations are LLM-generated.
- Any synchronous upload-and-result contract for `/process`.
- Legacy fixed weights and local JSON baseline behavior unless retained in a separately labeled historical appendix.

## 20. Evidence Appendix

### Primary execution and contracts

- `main.py:app`, route decorators, `ScanRequest`, `BaselineRequest`, `process_scan`, `process_scan_background`, `_process_scan_sync`, `_write_success`, `_build_scan_result_payload`.
- `directus_client.py:DirectusClient` for backend reads, asset authentication, schema inspection, and writes.
- `validation.py:ValidationPolicy.from_env`, `validate_scan_inputs`, `validate_video_result`, `validate_audio_result`, `validate_image_result`, `validate_phrase_result`.
- `quality.py:assess_quality` for modality usability, aggregate quality, confidence multiplier, warnings, retake and missing states.

### Face/video

- `vision.py:_eye_aspect_ratio`, `_analyze_image_array`, `analyze_face`.
- `video.py:_build_sample_plan`, `_motion_stability_from_diffs`, `_eye_landmark_reliability`, `_longest_temporal_eye_closure_streak`, `analyze_video`.

### Voice

- `audio.py:_decode_wav_slice`, `_decode_with_ffmpeg`, `_decode_audio_once`, `_feature_pipeline`, `_presence_from_features`, `_voice_clarity_from_features`, `_speech_state_and_warnings`, `_build_success_details`, `analyze_audio`, `transcribe_audio`.

### Scoring and classification

- `scoring.py:BASE_WEIGHTS`, `WARNING_SCORE_PENALTIES`, `compute_task_score`, `_degraded_signal_score`, `_adaptive_weights`, `_confidence_from_profiles`, `_video_fatigue_signal`, `_audio_fatigue_signal`, `_risk_level`, `_suggested_action`, `_explanation`, `compute_result`.
- `ml/features.py:FEATURE_ORDER`; `ml/runtime.py:MLRuntime.load`, `MLRuntime.predict`; `config.py:LABELS`, `LABEL_SCORES`, `MODEL_VERSION`, `REQUIRE_LOCAL_MODEL`.

### Baseline

- `config.py:BASELINE_PROVISIONAL_AFTER`, `BASELINE_ACTIVE_AFTER`, `BASELINE_USE_AFTER`, `BASELINE_HIGH_CONFIDENCE_AFTER`, quality/confidence/sample limits.
- `baseline.py:FACE_FEATURES`, `VOICE_FEATURES`, `REACTION_FEATURES`, `current_baseline_features`, `evaluate_baseline_eligibility`, `baseline_signal_payload`, `baseline_status_payload`, `baseline_ready_for_personalized_scoring`, `baseline_feature_reference`.
- `sql/2026_07_01_phase2_baseline_foundation.sql` for persistent schema and historical backfill rules.

### Runtime, privacy, and tests

- `analysis_runtime.py:WorkerSupervisor`, `WarmAnalyzerRuntime`; `analysis_worker.py:worker_main`.
- `utils.py:download_temp_file`, `remove_temp_file`; `main.py:_convert_audio_to_wav`, `_cleanup_download_future`, `_defer_temp_file_cleanup`, cleanup `finally` blocks.
- `logger.py:get_logger`; performance/error log calls in `main.py`, `analysis_runtime.py`, `analysis_worker.py`, `audio.py`.
- Root `test_validation.py`, `test_pipeline.py`; all modules under `tests/` listed in section 15.

AI SERVER AUDIT COMPLETE  
Implemented: 14  
Partial: 7  
Missing: 9  
Changed: 8  
Extra: 6  
Dead/Unused: 1
