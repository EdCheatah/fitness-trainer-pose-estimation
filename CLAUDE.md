# CLAUDE.md — Fitness Trainer Pose Estimation

Context for AI assistants working on this repo. Read this in full before making changes.

## Project

Fork of [yakupzengin/fitness-trainer-pose-estimation](https://github.com/yakupzengin/fitness-trainer-pose-estimation). Academic project (8th-semester Robotics, 3rd partial exam). Python 3.14, MediaPipe Tasks API (PoseLandmarker), OpenCV, Flask MJPEG streaming.

**Hardware target:** Ryzen 7 6800H, RTX 3050 Ti (MediaPipe is CPU-only on Windows), 720p webcam, frame size **640x480** (1280x720 drops to ~1 FPS).

## Repo and remotes

- Local path: `E:\Estudios\8VO_SEMESTRE\ROBOTICA\3erParcial\RAMA0\fitness-trainer-pose-estimation`
- User's fork (origin): `https://github.com/EdCheatah/fitness-trainer-pose-estimation`
- Original (upstream): `https://github.com/yakupzengin/fitness-trainer-pose-estimation`
- Branch: `master`

Pull updates from the original: `git fetch upstream && git merge upstream/master`
Push your work: `git push origin master`

## File map

```
app.py                              Flask server, MJPEG streaming, REST endpoints, video upload
exercises/
  engine.py                         ExerciseEngine: dispatches standard / bilateral / duration
  base_exercise.py                  BaseExercise + BilateralExercise + DurationExercise (FSM, scoring)
  loader.py                         Loads YAML configs and builds the correct class
  definitions/
    squat.yaml                      Currently shoulder-hip-knee angle, parallel = ascent state
    bicep_curl.yaml                 Bilateral, recently fixed (left/right keys, partial_rep condition)
    push_up.yaml                    Standard, has hip-sag / hip-too-high feedback
    plank.yaml                      Duration-based, 30 s default
    ... (12 other exercises)
pose_estimation/
  estimation.py                     PoseLandmarker wrapper, auto-downloads .task model on first run
  angle_calculation.py              Pure-math angle helper
utils/draw_text_with_background.py  OpenCV text rendering with background box
test_engine.py                      Standalone smoke test for the YAML pipeline
```

## Conventions for THIS repo (override defaults)

1. **Commits**: do NOT include `Co-Authored-By: Claude ...` or any reference to Claude in the body. The user is the sole author.
2. **No emojis** in code or commit messages unless the user asks for them.
3. **All comments / docstrings / user-facing strings are in English.** Turkish was fully removed on 2026-05-26.
4. **Don't touch `.task` model files** — they're gitignored and auto-downloaded by `PoseEstimator`.
5. **Model selection**: for mechanical work (translations, mass renames, bulk YAML edits) flag the cost and suggest switching to Haiku 4.5 or Sonnet 4.6 instead of running on Opus 4.7. The user has called this out explicitly.

## State of the bicep curl (DONE, do not redo)

- Migrated to MediaPipe Tasks API (`PoseLandmarker`) instead of the deprecated `mp.solutions.pose`. Model downloads on first run.
- Squeeze penalty fix in `engine.py::_process_bilateral` (lines ~167-271): per-side `_flex_min_left/right`, 0.2 s noise filter before evaluating, `_flex_penalty_done` flag to prevent double penalty, evaluation on leaving-flex transition.
- Ready grace period: `READY_GRACE_SECS = 1.5` in `ExerciseEngine`. The first 1.5 s after the body re-enters the frame doesn't count reps. Implemented in `engine.py::process_frame` lines ~76-86.
- `BilateralExercise.__init__` and `reset()` now manage `_flex_min_left/right`, `_flex_enter_left/right`, `_flex_penalty_done`.
- `bicep_curl.yaml` angle keys renamed: `primary` -> `left`, `right_arm` -> `right`. `partial_rep` condition tightened to `state == 'flex' and angle > 40`.

## Squat — next focus

The user confirmed the current `shoulder -> hip -> knee` angle is **good enough** to keep. Do NOT propose changing to a knee angle (`hip -> knee -> ankle`) — that ship has sailed. Iterate the feedback rules instead.

### Current `squat.yaml` summary

- Primary angle: `[left_shoulder, left_hip, left_knee]` (torso-hip-leg angle)
- States: `start` (>165), `descent` (90-165], `ascent` (<=90)
- Counter: trigger=`ascent`, from=`descent`, min_rep_duration=0.8 s
- Feedback rules:
  - `knees_caving`: `left_knee_x < left_ankle_x - 20` -> "Knees caving in!"
  - `leaning_forward`: `left_shoulder_x > left_hip_x + 50` -> "Leaning too far forward!" **(broken from frontal camera — see research below)**
  - `not_deep_enough`: `100 < angle < 130` -> "Go deeper!" **(only triggers mid-rep, not on shallow completion)**

### Biomechanics research (DONE — do not re-search)

Pulled from NSCA, NASM, ISSA, Brookbush, PubMed/PMC, Schoenfeld, Nerd Fitness on 2026-05-26.

**Squat depth classifications by interior knee angle (180° = straight):**

| Type | Interior knee angle |
|---|---|
| Quarter | 110-140° |
| Half | 80-100° |
| **Parallel** | **60-90°** (thigh parallel to floor) |
| Deep / full | < 70° (hip crease below knee) |

**A valid rep (consensus)**: hip crease drops to or below knee — i.e. at least parallel. NSCA and meta-analyses agree deeper squats give better strength gains and do NOT damage the knee.

**Common form errors (ranked):**
1. Knee valgus (knees caving in) — most common
2. Insufficient depth
3. Heels lifting
4. Excessive forward lean
5. Wrong stance / toe direction
6. Asymmetry / hip drop

**Detectability from frontal 640×480 camera:**

| Error | Detectable? | Method |
|---|---|---|
| Knee valgus | Yes, reliable | Frontal Plane Projection Angle (FPPA), >10° from baseline = problem |
| Depth | Yes | Any depth angle |
| Asymmetry | Yes | counter_left vs counter_right, or hip Y delta |
| Stance width | Yes | Ankle X distance |
| Heel lift | Partial | Y of foot relative to baseline |
| **Forward lean** | **No** — needs side camera | The current X-shift check is wrong |
| Toe direction | No | Needs side / top camera |

### Squat improvement plan (work 1×1, do NOT do all at once)

1. **Fix `not_deep_enough`** so it triggers when the rep ENDS without ever reaching the depth threshold, instead of triggering mid-rep at a specific angle window. (Track min-angle reached in the rep, compare on rep completion.)
2. **Replace `leaning_forward`** — current logic measures lateral X shift, not forward lean. Either remove it or replace with something detectable from the frontal plane (e.g., excessive hip-shoulder X distance change from baseline, which is still iffy). Recommendation: remove.
3. **Improve `knees_caving`** — current `-20` pixel offset is fragile. Normalize against stance width (e.g., distance between ankles) or use the FPPA approach (compare hip-knee-ankle projection angle vs baseline at start position; flag if increases >10°).
4. **Add asymmetry check** — if `counter_left` vs `counter_right` diverge significantly, or if hip Y values differ a lot at the bottom, flag a hip-drop warning. (Squat is currently standard, not bilateral — would need to compute right-side angle and track per-side, OR just hip-Y difference.)
5. **Form score weights for squat** — currently uses `calculate_form_score` with ideal_angles (empty), tempo, and feedback count. For squat specifically the user may want depth and knee tracking weighted more than tempo.

### Suggested workflow for the next session

- Pick ONE of the 5 items above with the user.
- Sketch the condition in plain English first ("rep counts as too shallow if min angle stayed above X for the entire rep").
- Implement, test against the live camera, iterate.
- Commit per fix. Use this commit format (no Claude attribution):

```
fix(squat): <one-line summary>

<body explaining the change>
```

### Sources (research-grade, already vetted)

- NSCA — Considerations for Squat Depth: https://www.nsca.com/education/articles/nsca-coach/considerations-for-squat-depth/
- NASM — Squat Biomechanics: https://blog.nasm.org/biomechanics-of-the-squat
- A Biomechanical Review of the Squat Exercise (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10987311/
- Schoenfeld — The Biomechanics of Squat Depth: https://www.lookgreatnaked.com/articles/the_biomechanics_of_squat_depth.pdf
- 2-D video knee-valgus reliability (PubMed): https://pubmed.ncbi.nlm.nih.gov/22104115/
- Brookbush — Squat Depth Recommendations: https://brookbushinstitute.com/articles/deep-squats-good-or-bad

## Pending across the project (not just squat)

- TTS feedback with `pyttsx3` so messages are spoken, not just drawn
- Calibration thresholds review for each exercise YAML
- Verify other exercises still work after the migration to PoseLandmarker (smoke-test each)
