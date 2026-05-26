"""
Exercise Engine - high-level API on top of BaseExercise.

This module wraps BaseExercise so it can be used directly inside a
real-time frame processing loop.
"""

import time
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from exercises.base_exercise import BaseExercise, BilateralExercise, DurationExercise
from exercises.loader import load_exercise, get_exercise_info, get_available_exercises
from utils.draw_text_with_background import draw_text_with_background


class ExerciseEngine:
    """
    Main engine for frame processing and rendering of an exercise.

    Usage:
        engine = ExerciseEngine()
        engine.set_exercise("squat")

        # Inside the frame loop:
        result = engine.process_frame(frame, landmarks)
    """

    READY_GRACE_SECS = 1.5  # wait this long after body appears before counting reps

    def __init__(self):
        self.exercise: Optional[BaseExercise] = None
        self.exercise_name: str = None
        self._exercise_info: Dict = {}
        self._body_visible_since: Optional[float] = None

    def set_exercise(self, exercise_name: str) -> bool:
        """
        Set the active exercise.

        Args:
            exercise_name: Exercise name (e.g. "squat")

        Returns:
            True on success.
        """
        try:
            self.exercise = load_exercise(exercise_name)
            self.exercise_name = exercise_name
            self._exercise_info = get_exercise_info(exercise_name)
            return True
        except Exception as e:
            print(f"Failed to load exercise '{exercise_name}': {e}")
            return False

    def reset(self):
        """Reset the current exercise state."""
        if self.exercise:
            self.exercise.reset()

    def process_frame(self, frame: np.ndarray, landmarks) -> Dict[str, Any]:
        """
        Process a single frame and update exercise state.

        Args:
            frame: OpenCV frame (BGR)
            landmarks: MediaPipe pose landmarks

        Returns:
            Dict with the results of this frame.
        """
        if not self.exercise or not landmarks:
            return {"success": False, "error": "No exercise or landmarks"}

        if not self.exercise.key_landmarks_visible(landmarks):
            self._body_visible_since = None  # reset grace timer when body leaves
            return {"success": False, "error": "Key landmarks not visible"}

        # Grace period: ignore first N seconds after body reappears to avoid ghost reps
        now = time.time()
        if self._body_visible_since is None:
            self._body_visible_since = now
        grace_remaining = self.READY_GRACE_SECS - (now - self._body_visible_since)
        if grace_remaining > 0:
            return {"success": False, "error": "Getting ready", "grace_remaining": grace_remaining}

        frame_shape = frame.shape[:2]  # (height, width)

        result = {
            "success": True,
            "exercise_name": self.exercise_name,
            "counter": 0,
            "state": None,
            "angles": {},
            "feedback": [],
            "counted": False
        }

        try:
            # Bilateral (two-sided) exercise?
            if isinstance(self.exercise, BilateralExercise):
                result = self._process_bilateral(frame, landmarks, frame_shape, result)

            # Duration-based exercise?
            elif isinstance(self.exercise, DurationExercise):
                result = self._process_duration(frame, landmarks, frame_shape, result)

            # Standard rep-based exercise
            else:
                result = self._process_standard(frame, landmarks, frame_shape, result)

            # Rendering
            self._draw_visualization(frame, landmarks, frame_shape)
            self._draw_feedback(frame, result["feedback"])

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"Exercise processing error: {e}")

        return result

    def _process_standard(self, frame, landmarks, frame_shape, result):
        """Process a single frame for a standard rep-based exercise."""
        # Compute every defined angle
        self.exercise.compute_all_angles(landmarks, frame_shape)

        # Build the evaluation context
        context = self.exercise.get_context(landmarks, frame_shape)

        # Update the FSM
        prev_state = self.exercise.current_state
        self.exercise.update_state(context)

        # Start rep tracking the moment we enter the descent state
        if prev_state == "start" and self.exercise.current_state == "descent":
            self.exercise.start_rep_tracking()

        # Update the rep counter
        counted = self.exercise.update_counter()

        # Evaluate form feedback rules
        feedback = self.exercise.check_feedback(context)

        # Compute the FORM SCORE
        form_score = self.exercise.calculate_form_score(context, feedback)

        # Close the rep timing window when a rep is counted
        if counted:
            self.exercise.end_rep_tracking()

        # Fill in the result
        result.update({
            "counter": self.exercise.counter,
            "state": self.exercise.current_state,
            "angles": self.exercise._computed_angles.copy(),
            "feedback": feedback,
            "counted": counted,
            "form_score": form_score,
            "avg_form_score": self.exercise.avg_form_score,
            "form_grade": self.exercise.get_form_score_grade()
        })

        return result

    def _process_bilateral(self, frame, landmarks, frame_shape, result):
        """Process a single frame for a bilateral exercise (left + right tracked separately)."""
        exercise: BilateralExercise = self.exercise

        # Compute angles for both sides
        exercise.compute_bilateral_angles(landmarks, frame_shape)

        # Build the evaluation context
        context = exercise.get_context(landmarks, frame_shape)
        context["left_angle"] = exercise._computed_angles.get("left_angle", 0)
        context["right_angle"] = exercise._computed_angles.get("right_angle", 0)

        # Update FSM state for both sides
        exercise.update_bilateral_state(context)

        trigger_state      = exercise.counter_rule.get("trigger_state")
        squeeze_threshold  = 40
        min_flex_hold_secs = 0.2  # ignore leaving_flex events shorter than this (noise filter)

        left_angle  = context.get("left_angle",  180)
        right_angle = context.get("right_angle", 180)
        now = time.time()

        # Per-side flex transitions
        entering_flex_left  = (exercise.prev_state_left  != trigger_state and exercise.current_state_left  == trigger_state)
        entering_flex_right = (exercise.prev_state_right != trigger_state and exercise.current_state_right == trigger_state)
        leaving_flex_left   = (exercise.prev_state_left  == trigger_state and exercise.current_state_left  != trigger_state)
        leaving_flex_right  = (exercise.prev_state_right == trigger_state and exercise.current_state_right != trigger_state)

        # On entry: reset per-side min tracker, record entry time, clear penalty flag
        if entering_flex_left:
            exercise._flex_min_left     = left_angle
            exercise._flex_enter_left   = now
            exercise._flex_penalty_done = False
        if entering_flex_right:
            exercise._flex_min_right    = right_angle
            exercise._flex_enter_right  = now
            exercise._flex_penalty_done = False

        # Track minimum angle per side independently while in flex
        if exercise.current_state_left == trigger_state:
            exercise._flex_min_left  = min(getattr(exercise, '_flex_min_left',  180), left_angle)
        if exercise.current_state_right == trigger_state:
            exercise._flex_min_right = min(getattr(exercise, '_flex_min_right', 180), right_angle)

        # Update counters
        left_counted, right_counted = exercise.update_bilateral_counter()

        # On leaving flex: evaluate squeeze (once per rep, only if held long enough to filter noise)
        leaving_flex = leaving_flex_left or leaving_flex_right
        if leaving_flex and not getattr(exercise, '_flex_penalty_done', False):
            hold_left  = now - getattr(exercise, '_flex_enter_left',  now)
            hold_right = now - getattr(exercise, '_flex_enter_right', now)
            long_enough = (leaving_flex_left  and hold_left  >= min_flex_hold_secs) or \
                          (leaving_flex_right and hold_right >= min_flex_hold_secs)

            if long_enough:
                # Check the worst-performing arm (highest min angle = least contracted)
                worst_min = 0
                if leaving_flex_left:
                    worst_min = max(worst_min, getattr(exercise, '_flex_min_left',  squeeze_threshold + 1))
                if leaving_flex_right:
                    worst_min = max(worst_min, getattr(exercise, '_flex_min_right', squeeze_threshold + 1))
                if worst_min > squeeze_threshold:
                    exercise.current_form_score = max(0, exercise.current_form_score - 10)
                exercise._flex_penalty_done = True

            exercise.start_rep_tracking()

        # End rep tracking when rep is counted
        if left_counted or right_counted:
            exercise.end_rep_tracking()

        # Evaluate feedback per side so the "angle" key is defined for each evaluation
        context["counter_left"] = exercise.counter_left
        context["counter_right"] = exercise.counter_right
        seen_fb = set()
        feedback = []
        for side in ["left", "right"]:
            side_ctx = context.copy()
            side_ctx["angle"] = context.get(f"{side}_angle", 0)
            side_ctx["state"] = exercise.current_state_left if side == "left" else exercise.current_state_right
            for fb in exercise.check_feedback(side_ctx):
                if fb["name"] not in seen_fb:
                    seen_fb.add(fb["name"])
                    feedback.append(fb)

        form_score = exercise.current_form_score

        # Fill in the result
        result.update({
            "counter": exercise.counter,
            "counter_left": exercise.counter_left,
            "counter_right": exercise.counter_right,
            "state_left": exercise.current_state_left,
            "state_right": exercise.current_state_right,
            "angles": exercise._computed_angles.copy(),
            "feedback": feedback,
            "counted": left_counted or right_counted,
            "form_score": form_score,
            "avg_form_score": exercise.avg_form_score,
            "form_grade": exercise.get_form_score_grade()
        })

        return result

    def _process_duration(self, frame, landmarks, frame_shape, result):
        """Process a single frame for a duration-based exercise (e.g. plank)."""
        exercise: DurationExercise = self.exercise

        # Compute angles
        exercise.compute_all_angles(landmarks, frame_shape)

        # Build the evaluation context
        context = exercise.get_context(landmarks, frame_shape)

        # Update the hold timer (this also updates the FSM state)
        current_duration = exercise.update_duration(context)

        # Evaluate form feedback rules
        feedback = exercise.check_feedback(context)

        # Fill in the result
        result.update({
            "counter": exercise.counter,
            "state": exercise.current_state,
            "current_duration": current_duration,
            "target_duration": exercise.target_duration,
            "is_holding": exercise.is_holding,
            "angles": exercise._computed_angles.copy(),
            "feedback": feedback
        })

        return result

    def _draw_visualization(self, frame, landmarks, frame_shape):
        """Render the exercise overlay (lines, joints, angle text)."""
        if not self.exercise:
            return

        viz_config = self.exercise.get_visualization_config()

        # Draw lines
        for line in viz_config.get("lines", []):
            points = line["points"]
            color = tuple(line.get("color", [0, 255, 0]))
            thickness = line.get("thickness", 2)

            try:
                p1 = self.exercise.get_landmark_coords(landmarks, points[0], frame_shape)
                p2 = self.exercise.get_landmark_coords(landmarks, points[1], frame_shape)
                cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)
            except:
                pass

        # Draw joint circles
        for circle in viz_config.get("circles", []):
            point = circle["point"]
            color = tuple(circle.get("color", [0, 255, 0]))
            radius = circle.get("radius", 5)

            try:
                center = self.exercise.get_landmark_coords(landmarks, point, frame_shape)
                cv2.circle(frame, center, radius, color, -1)
            except:
                pass

        # Draw angle labels
        for angle_display in viz_config.get("angle_display", []):
            angle_name = angle_display["angle"]
            position_point = angle_display["position"]
            offset = angle_display.get("offset", [10, -10])
            label = angle_display.get("label", "Angle")

            try:
                pos = self.exercise.get_landmark_coords(landmarks, position_point, frame_shape)
                angle_value = self.exercise._computed_angles.get(angle_name, 0)
                text = f"{label}: {int(angle_value)}"
                text_pos = (pos[0] + offset[0], pos[1] + offset[1])
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except:
                pass

    def _draw_feedback(self, frame, feedback_list):
        """Render feedback messages stacked from the bottom of the frame."""
        y_offset = frame.shape[0] - 100  # start from the bottom

        for fb in feedback_list:
            message = fb["message"]
            severity = fb.get("severity", "warning")

            # Color by severity
            if severity == "error":
                bg_color = (0, 0, 200)
            elif severity == "warning":
                bg_color = (0, 165, 255)
            else:  # info
                bg_color = (200, 200, 0)

            draw_text_with_background(
                frame, message, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), bg_color, 1
            )
            y_offset -= 35

    def draw_status_overlay(self, frame, exercise_goal: int = 10, sets_goal: int = 3,
                           sets_completed: int = 0):
        """
        Render the status overlay (rep count, current set, goals, stage).

        Args:
            frame: OpenCV frame
            exercise_goal: Target number of reps
            sets_goal: Target number of sets
            sets_completed: Sets completed so far
        """
        if not self.exercise:
            return

        info = self._exercise_info

        # Exercise name
        draw_text_with_background(
            frame, f"Exercise: {info.get('name', self.exercise_name)}",
            (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255), (118, 29, 14), 1
        )

        # Reps goal
        draw_text_with_background(
            frame, f"Reps Goal: {exercise_goal}",
            (40, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255), (118, 29, 14), 1
        )

        # Sets goal
        draw_text_with_background(
            frame, f"Sets Goal: {sets_goal}",
            (40, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255), (118, 29, 14), 1
        )

        # Current set
        draw_text_with_background(
            frame, f"Current Set: {sets_completed + 1}",
            (40, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (255, 255, 255), (118, 29, 14), 1
        )

        # Rep counter
        counter = self.exercise.counter
        draw_text_with_background(
            frame, f"Count: {counter}",
            (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 0), (192, 192, 192), 2
        )

        # Stage
        state = self.exercise.current_state or "Ready"
        draw_text_with_background(
            frame, f"Stage: {state.title() if state else 'Ready'}",
            (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), (192, 192, 192), 1
        )

        # Extra info for bilateral exercises
        if isinstance(self.exercise, BilateralExercise):
            draw_text_with_background(
                frame, f"Left: {self.exercise.counter_left} | Right: {self.exercise.counter_right}",
                (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), (192, 192, 192), 1
            )

        # Extra info for duration exercises
        if isinstance(self.exercise, DurationExercise):
            duration = int(self.exercise.current_duration)
            target = self.exercise.target_duration
            draw_text_with_background(
                frame, f"Hold: {duration}/{target}s",
                (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), (192, 192, 192), 1
            )

    def draw_form_score(self, frame):
        """
        Render the Form Score indicator (large score with letter grade and progress bar)
        in the top-right corner of the frame.
        """
        if not self.exercise:
            return

        score = self.exercise.current_form_score
        grade = self.exercise.get_form_score_grade()
        color = self.exercise.get_form_score_color()
        avg_score = self.exercise.avg_form_score

        # Frame dimensions
        h, w = frame.shape[:2]

        # Top-right anchor
        x_pos = w - 180
        y_pos = 50

        # Background box
        cv2.rectangle(frame, (x_pos - 10, y_pos - 40), (w - 10, y_pos + 100), (50, 50, 50), -1)
        cv2.rectangle(frame, (x_pos - 10, y_pos - 40), (w - 10, y_pos + 100), color, 2)

        # "FORM SCORE" header
        cv2.putText(frame, "FORM SCORE", (x_pos, y_pos - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Big score number
        cv2.putText(frame, f"{score}", (x_pos + 20, y_pos + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

        # Letter grade
        cv2.putText(frame, grade, (x_pos + 110, y_pos + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        # Average score
        cv2.putText(frame, f"Avg: {avg_score}", (x_pos, y_pos + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Progress bar
        bar_width = 150
        bar_height = 8
        bar_x = x_pos
        bar_y = y_pos + 90

        # Bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

        # Filled portion
        fill_width = int((score / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

    def get_counter(self) -> int:
        """Return the current rep counter."""
        if self.exercise:
            return self.exercise.counter
        return 0

    def get_status(self) -> Dict[str, Any]:
        """Return the current exercise status."""
        if self.exercise:
            return self.exercise.get_status()
        return {}

    @staticmethod
    def list_exercises() -> List[str]:
        """List all available exercises."""
        return get_available_exercises()

    @staticmethod
    def get_info(exercise_name: str) -> Dict:
        """Return metadata for a single exercise."""
        return get_exercise_info(exercise_name)
