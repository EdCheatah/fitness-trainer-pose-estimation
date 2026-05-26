"""
BaseExercise - Core Exercise Engine.

Base class for every exercise. Movements are loaded from YAML files
and executed by this engine.

Features:
- Finite State Machine (FSM) for movement phases
- Automatic rep counting
- Real-time form feedback
- Calibration support
- Time-based filtering (prevents miscounts)
"""

import re
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any


class BaseExercise:
    """
    Base class for every exercise.

    Processes movement definitions loaded from a YAML configuration.
    """

    # MediaPipe landmark indices
    LANDMARK_MAP = {
        # Face
        "nose": 0,
        # Shoulders
        "left_shoulder": 11,
        "right_shoulder": 12,
        # Elbows
        "left_elbow": 13,
        "right_elbow": 14,
        # Wrists
        "left_wrist": 15,
        "right_wrist": 16,
        # Hips
        "left_hip": 23,
        "right_hip": 24,
        # Knees
        "left_knee": 25,
        "right_knee": 26,
        # Ankles
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the exercise engine.

        Args:
            config: Exercise configuration loaded from YAML
        """
        # Basic metadata
        self.name = config["name"]
        self.display_name = config.get("display_name", self.name.replace("_", " ").title())
        self.type = config.get("type", "repetition")  # repetition | duration

        # Angle definitions
        self.angles = config.get("angles", {})

        # Finite State Machine (FSM)
        self.states = config.get("states", {})
        self.state_order = config.get("state_order", list(self.states.keys()))

        # Counter rules
        self.counter_rule = config.get("counter", {})

        # Feedback rules
        self.feedback_rules = config.get("feedback", {})

        # Drawing / visualization settings
        self.visualization = config.get("visualization", {})

        # FSM state variables
        self.current_state = None
        self.prev_state = None
        self.counter = 0
        self.counter_left = 0  # For bilateral movements
        self.counter_right = 0

        # Time-based filter
        self.last_count_time = 0
        self.min_rep_duration = config.get("min_rep_duration", 0.5)  # minimum seconds

        # Calibration
        self.calibration_enabled = config.get("calibration", {}).get("enabled", False)
        self.calibration_reps = config.get("calibration", {}).get("reps", 3)
        self.calibration_data = {"max_angles": [], "min_angles": []}
        self.is_calibrated = False

        # Smoothing
        self.smoothing_enabled = config.get("smoothing", {}).get("enabled", False)
        self.smoothing_window = config.get("smoothing", {}).get("window", 5)
        self.angle_history = []

        # Computed angles cache
        self._computed_angles = {}

        # Calibration runtime
        self.calibration_scale = 1.0
        self.calibration_offset_val = 0.0
        self.calibration_angles_all = []

        # Cycle timing (#2 – rep speed detection)
        self.last_left_trigger_time = 0.0
        self.last_left_trigger_left = 0.0
        self.last_left_trigger_right = 0.0

        # ==================== FORM SCORE SYSTEM ====================
        # Variables for Form Score (0-100) computation
        self.form_score_config = config.get("form_score", {})
        self.ideal_angles = self.form_score_config.get("ideal_angles", {})
        self.tempo_range = self.form_score_config.get("tempo_range", {"min": 1.0, "max": 3.0})

        # Rep tracking for form score
        self.rep_start_time = None
        self.rep_durations = []
        self.rep_form_scores = []
        self.current_form_score = 100
        self.avg_form_score = 100

        # Feedback penalty tracking
        self.active_feedback_count = 0

    def get_landmark_coords(self, landmarks, point_name: str, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get pixel coordinates from a landmark name.

        Args:
            landmarks: MediaPipe pose landmarks
            point_name: Landmark name (e.g. "left_hip")
            frame_shape: Frame dimensions (height, width)

        Returns:
            (x, y) pixel coordinates
        """
        idx = self.LANDMARK_MAP.get(point_name)
        if idx is None:
            raise ValueError(f"Unknown landmark: {point_name}")

        landmark = landmarks[idx]
        x = int(landmark.x * frame_shape[1])
        y = int(landmark.y * frame_shape[0])
        return (x, y)

    def compute_angle(self, landmarks, angle_name: str, frame_shape: Tuple[int, int]) -> float:
        """
        Compute the specified angle.

        Args:
            landmarks: MediaPipe pose landmarks
            angle_name: Angle name (defined under 'angles' in the config)
            frame_shape: Frame dimensions

        Returns:
            Angle in degrees
        """
        angle_def = self.angles.get(angle_name)
        if not angle_def:
            raise ValueError(f"Undefined angle: {angle_name}")

        points = angle_def["points"]
        p1 = self.get_landmark_coords(landmarks, points[0], frame_shape)
        p2 = self.get_landmark_coords(landmarks, points[1], frame_shape)
        p3 = self.get_landmark_coords(landmarks, points[2], frame_shape)

        angle = self._angle_between(p1, p2, p3)

        # Apply smoothing
        if self.smoothing_enabled:
            angle = self._smooth_angle(angle)

        # Collect calibration samples (#3)
        if self.calibration_enabled and not self.is_calibrated:
            self.calibration_angles_all.append(angle)

        # Apply calibration scale+offset if calibrated
        if self.is_calibrated and self.calibration_scale != 1.0:
            angle = max(0.0, min(180.0, self.calibration_scale * angle + self.calibration_offset_val))

        # Cache the result
        self._computed_angles[angle_name] = angle

        return angle

    def compute_all_angles(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Compute every defined angle."""
        self._computed_angles = {}
        for angle_name in self.angles.keys():
            self.compute_angle(landmarks, angle_name, frame_shape)
        return self._computed_angles

    def get_context(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Build the context dict used for state evaluation.

        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Frame dimensions

        Returns:
            Dict containing every angle and landmark coordinate.
        """
        context = {}

        # Add every computed angle
        for angle_name, angle_value in self._computed_angles.items():
            context[f"{angle_name}_angle"] = angle_value
            # Also expose a short-form "angle" key for the primary angle
            if angle_name == "primary":
                context["angle"] = angle_value

        # Add landmark coordinates
        for point_name, idx in self.LANDMARK_MAP.items():
            try:
                coords = self.get_landmark_coords(landmarks, point_name, frame_shape)
                context[f"{point_name}_x"] = coords[0]
                context[f"{point_name}_y"] = coords[1]
            except:
                pass

        return context

    def key_landmarks_visible(self, landmarks, min_visibility: float = 0.5) -> bool:
        """Return False if any landmark used in angle calculations is not visible enough."""
        for angle_def in self.angles.values():
            for point_name in angle_def.get("points", []):
                idx = self.LANDMARK_MAP.get(point_name)
                if idx is None:
                    continue
                try:
                    if landmarks[idx].visibility < min_visibility:
                        return False
                except (IndexError, AttributeError):
                    return False
        return True

    def update_state(self, context: Dict[str, Any]) -> str:
        """
        Update the current FSM state.

        Args:
            context: Dict containing angles and coordinates

        Returns:
            New state name
        """
        self.prev_state = self.current_state

        # Check states in priority order
        for state_name in self.state_order:
            state_def = self.states.get(state_name, {})
            condition = state_def.get("condition", "False")

            # Safe eval
            try:
                if self._safe_eval(condition, context):
                    self.current_state = state_name
                    break
            except Exception as e:
                print(f"State condition error ({state_name}): {e}")

        # Track when we leave the trigger state (rep cycle start)
        trigger_state = self.counter_rule.get("trigger_state")
        if trigger_state and self.prev_state == trigger_state and self.current_state != trigger_state:
            self.last_left_trigger_time = time.time()

        return self.current_state

    def update_counter(self) -> bool:
        """
        Update the rep counter.

        Returns:
            True if the counter was incremented.
        """
        trigger_state = self.counter_rule.get("trigger_state")
        from_state = self.counter_rule.get("from_state")  # Optional: required previous state

        # State transition check
        state_changed = self.prev_state != self.current_state
        reached_trigger = self.current_state == trigger_state

        # Validate from_state if provided
        from_valid = True
        if from_state:
            from_valid = self.prev_state == from_state

        # Time-based filter
        current_time = time.time()
        time_valid = (current_time - self.last_count_time) >= self.min_rep_duration

        # Cycle duration check (#2): full rep cycle must take >= min_rep_duration
        cycle_valid = (
            self.last_left_trigger_time == 0.0 or
            (current_time - self.last_left_trigger_time) >= self.min_rep_duration
        )

        if state_changed and reached_trigger and from_valid and time_valid and cycle_valid:
            self.counter += 1
            self.last_count_time = current_time

            # Trigger calibration after N reps (#3)
            if self.calibration_enabled and not self.is_calibrated and self.counter >= self.calibration_reps:
                self._apply_calibration()

            return True

        return False

    def check_feedback(self, context: Dict[str, Any]) -> List[str]:
        """
        Evaluate form feedback rules.

        Args:
            context: Dict containing angles and coordinates

        Returns:
            List of warning messages
        """
        messages = []

        for feedback_name, feedback_def in self.feedback_rules.items():
            condition = feedback_def.get("condition", "False")
            message = feedback_def.get("message", "Form warning")
            severity = feedback_def.get("severity", "warning")  # warning | error | info

            try:
                if self._safe_eval(condition, context):
                    messages.append({
                        "name": feedback_name,
                        "message": message,
                        "severity": severity
                    })
            except Exception as e:
                print(f"Feedback condition error ({feedback_name}): {e}")

        return messages

    def get_visualization_config(self) -> Dict[str, Any]:
        """Return the visualization configuration."""
        return self.visualization

    # ==================== FORM SCORE METHODS ====================

    def calculate_form_score(self, context: Dict[str, Any], feedback_list: List[Dict]) -> int:
        """
        Compute the form score (0-100).

        Score components:
        - Angle accuracy (40 points max penalty)
        - Tempo/speed (30 points max penalty)
        - Form errors (30 points max penalty)

        Args:
            context: Current angles and coordinates
            feedback_list: Active feedback messages

        Returns:
            Form score in the 0-100 range.
        """
        score = 100

        # 1. ANGLE ACCURACY (40 points max penalty)
        angle_penalty = self._calculate_angle_penalty(context)
        score -= min(angle_penalty, 40)

        # 2. TEMPO PENALTY (30 points max penalty)
        tempo_penalty = self._calculate_tempo_penalty()
        score -= min(tempo_penalty, 30)

        # 3. FORM ERRORS (30 points max penalty)
        # Each feedback message costs -10 points
        feedback_penalty = len(feedback_list) * 10
        score -= min(feedback_penalty, 30)

        # Clamp to 0-100
        score = max(0, min(100, score))

        self.current_form_score = score
        self.active_feedback_count = len(feedback_list)

        return score

    def _calculate_angle_penalty(self, context: Dict[str, Any]) -> int:
        """Compute penalty based on angle deviation from ideal."""
        if not self.ideal_angles:
            return 0

        total_deviation = 0
        count = 0

        for angle_name, ideal_value in self.ideal_angles.items():
            current_value = context.get(f"{angle_name}_angle") or context.get("angle", 0)
            if current_value:
                deviation = abs(current_value - ideal_value)
                # Each 10 degrees of deviation = 5 points penalty
                total_deviation += (deviation / 10) * 5
                count += 1

        if count > 0:
            return int(total_deviation / count)
        return 0

    def _calculate_tempo_penalty(self) -> int:
        """Compute the tempo/speed penalty."""
        if not self.rep_durations:
            return 0

        last_duration = self.rep_durations[-1] if self.rep_durations else 0
        min_tempo = self.tempo_range.get("min", 1.0)
        max_tempo = self.tempo_range.get("max", 3.0)

        if last_duration < min_tempo:
            # Too fast - each 0.5s = 15 points penalty
            return int((min_tempo - last_duration) / 0.5 * 15)
        elif last_duration > max_tempo:
            # Too slow - each 1s = 10 points penalty
            return int((last_duration - max_tempo) * 10)

        return 0

    def start_rep_tracking(self):
        """Store the rep start timestamp."""
        self.rep_start_time = time.time()

    def end_rep_tracking(self):
        """Record the rep duration and update the form score history."""
        if self.rep_start_time:
            duration = time.time() - self.rep_start_time
            self.rep_durations.append(duration)
            self.rep_start_time = None

            # Store this rep's form score
            self.rep_form_scores.append(self.current_form_score)

            # Update the running average form score
            if self.rep_form_scores:
                self.avg_form_score = int(sum(self.rep_form_scores) / len(self.rep_form_scores))

    def get_form_score_grade(self, score: int = None) -> str:
        """Return the letter grade for the form score."""
        if score is None:
            score = self.current_form_score

        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def get_form_score_color(self, score: int = None) -> Tuple[int, int, int]:
        """Return a BGR color matching the form score band."""
        if score is None:
            score = self.current_form_score

        if score >= 90:
            return (0, 255, 0)      # Green
        elif score >= 80:
            return (0, 255, 255)    # Yellow
        elif score >= 70:
            return (0, 165, 255)    # Orange
        elif score >= 60:
            return (0, 100, 255)    # Dark orange
        else:
            return (0, 0, 255)      # Red

    def reset(self):
        """Reset the exercise state."""
        self.current_state = None
        self.prev_state = None
        self.counter = 0
        self.counter_left = 0
        self.counter_right = 0
        self.last_count_time = 0
        self.angle_history = []
        self._computed_angles = {}
        self.last_left_trigger_time = 0.0
        self.last_left_trigger_left = 0.0
        self.last_left_trigger_right = 0.0
        # Form score reset
        self.rep_start_time = None
        self.rep_durations = []
        self.rep_form_scores = []
        self.current_form_score = 100
        self.avg_form_score = 100
        self.active_feedback_count = 0

    def get_status(self) -> Dict[str, Any]:
        """Return the current exercise status."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "counter": self.counter,
            "current_state": self.current_state,
            "angles": self._computed_angles.copy(),
            "is_calibrated": self.is_calibrated,
            "form_score": self.current_form_score,
            "avg_form_score": self.avg_form_score,
            "form_grade": self.get_form_score_grade()
        }

    # ==================== Private Methods ====================

    @staticmethod
    def _angle_between(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        """
        Compute the angle formed at vertex b by points a-b-c.

        Args:
            a, b, c: (x, y) coordinates

        Returns:
            Angle in degrees
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def _smooth_angle(self, angle: float) -> float:
        """Smooth the angle with a moving average."""
        self.angle_history.append(angle)
        if len(self.angle_history) > self.smoothing_window:
            self.angle_history.pop(0)
        return np.mean(self.angle_history)

    def _safe_eval(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a boolean condition string.

        Only a whitelisted set of names is exposed.
        """
        # Whitelisted names
        allowed_names = {
            "True": True,
            "False": False,
            "abs": abs,
            "min": min,
            "max": max,
        }
        allowed_names.update(context)

        # Reject dangerous constructs
        dangerous = ["import", "exec", "eval", "__", "open", "file", "os", "sys"]
        for d in dangerous:
            if d in condition:
                raise ValueError(f"Unsafe condition: {condition}")

        return eval(condition, {"__builtins__": {}}, allowed_names)

    def _collect_calibration_data(self):
        """Deprecated - data is now collected inline in compute_angle."""
        pass

    def _apply_calibration(self):
        """Compute scale+offset from collected angle samples so the user's ROM maps to YAML thresholds."""
        if len(self.calibration_angles_all) < 10:
            return

        arr = sorted(self.calibration_angles_all)
        n = len(arr)
        user_flex = arr[max(0, int(n * 0.05))]    # 5th-percentile  = most-flexed angle
        user_extend = arr[min(n - 1, int(n * 0.95))]  # 95th-percentile = most-extended angle

        if user_extend - user_flex < 5:            # not enough range to calibrate
            self.is_calibrated = True
            return

        # Parse YAML thresholds from state conditions
        trigger_state = self.counter_rule.get("trigger_state")
        yaml_flex, yaml_extend = user_flex, user_extend

        for state_name, state_def in self.states.items():
            condition = state_def.get("condition", "")
            if state_name == trigger_state:
                m = re.search(r'angle\s*<=?\s*(\d+(?:\.\d+)?)', condition)
                if m:
                    yaml_flex = float(m.group(1))
            else:
                # Look for a pure lower-bound condition (extended position)
                m = re.search(r'angle\s*>(?!=)\s*(\d+(?:\.\d+)?)\s*$', condition)
                if m:
                    val = float(m.group(1))
                    if val > yaml_flex:
                        yaml_extend = max(yaml_extend, val)

        yaml_range = yaml_extend - yaml_flex
        user_range = user_extend - user_flex

        if yaml_range > 5 and user_range > 5:
            self.calibration_scale = yaml_range / user_range
            self.calibration_offset_val = yaml_flex - self.calibration_scale * user_flex
        else:
            self.calibration_scale = 1.0
            self.calibration_offset_val = 0.0

        self.is_calibrated = True
        print(f"[Calibration] {self.name}: user=[{user_flex:.0f}°,{user_extend:.0f}°] "
              f"yaml=[{yaml_flex:.0f}°,{yaml_extend:.0f}°] "
              f"scale={self.calibration_scale:.2f} offset={self.calibration_offset_val:.1f}")


class BilateralExercise(BaseExercise):
    """
    Subclass for bilateral (two-sided) exercises.

    Example: Hammer Curl, where the left and right arms are counted
    independently.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Bilateral settings
        self.bilateral = config.get("bilateral", False)
        self.sides = config.get("sides", ["left", "right"])

        # Independent state per side
        self.current_state_left = None
        self.current_state_right = None
        self.prev_state_left = None
        self.prev_state_right = None

        self.last_count_time_left = 0
        self.last_count_time_right = 0

        # Squeeze tracking (set by engine per-rep)
        self._flex_min_left     = 180.0
        self._flex_min_right    = 180.0
        self._flex_enter_left   = 0.0
        self._flex_enter_right  = 0.0
        self._flex_penalty_done = False

    def compute_bilateral_angles(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Compute angles for both the left and right sides."""
        angles = {}

        for side in self.sides:
            angle_key = f"{side}_angle"
            angle_def = self.angles.get(side)

            if angle_def:
                points = angle_def["points"]
                p1 = self.get_landmark_coords(landmarks, points[0], frame_shape)
                p2 = self.get_landmark_coords(landmarks, points[1], frame_shape)
                p3 = self.get_landmark_coords(landmarks, points[2], frame_shape)

                angles[angle_key] = self._angle_between(p1, p2, p3)

        self._computed_angles.update(angles)
        return angles

    def update_bilateral_state(self, context: Dict[str, Any]) -> Tuple[str, str]:
        """Update the FSM state for both sides."""
        self.prev_state_left = self.current_state_left
        self.prev_state_right = self.current_state_right

        trigger_state = self.counter_rule.get("trigger_state")

        # Left side
        left_context = context.copy()
        left_context["angle"] = context.get("left_angle", 0)
        for state_name in self.state_order:
            state_def = self.states.get(state_name, {})
            condition = state_def.get("condition", "False")
            try:
                if self._safe_eval(condition, left_context):
                    self.current_state_left = state_name
                    break
            except:
                pass

        if trigger_state and self.prev_state_left == trigger_state and self.current_state_left != trigger_state:
            self.last_left_trigger_left = time.time()

        # Right side
        right_context = context.copy()
        right_context["angle"] = context.get("right_angle", 0)
        for state_name in self.state_order:
            state_def = self.states.get(state_name, {})
            condition = state_def.get("condition", "False")
            try:
                if self._safe_eval(condition, right_context):
                    self.current_state_right = state_name
                    break
            except:
                pass

        if trigger_state and self.prev_state_right == trigger_state and self.current_state_right != trigger_state:
            self.last_left_trigger_right = time.time()

        return self.current_state_left, self.current_state_right

    def update_bilateral_counter(self) -> Tuple[bool, bool]:
        """Update the counter for both sides."""
        trigger_state = self.counter_rule.get("trigger_state")
        current_time = time.time()

        left_counted = False
        right_counted = False

        # Left
        left_cycle_valid = (
            self.last_left_trigger_left == 0.0 or
            (current_time - self.last_left_trigger_left) >= self.min_rep_duration
        )
        if (self.prev_state_left != self.current_state_left and
                self.current_state_left == trigger_state and
                (current_time - self.last_count_time_left) >= self.min_rep_duration and
                left_cycle_valid):
            self.counter_left += 1
            self.last_count_time_left = current_time
            left_counted = True

        # Right
        right_cycle_valid = (
            self.last_left_trigger_right == 0.0 or
            (current_time - self.last_left_trigger_right) >= self.min_rep_duration
        )
        if (self.prev_state_right != self.current_state_right and
                self.current_state_right == trigger_state and
                (current_time - self.last_count_time_right) >= self.min_rep_duration and
                right_cycle_valid):
            self.counter_right += 1
            self.last_count_time_right = current_time
            right_counted = True

        # Combined counter
        self.counter = self.counter_left + self.counter_right

        # Trigger calibration after N reps (#3)
        if self.calibration_enabled and not self.is_calibrated and self.counter >= self.calibration_reps:
            self._apply_calibration()

        return left_counted, right_counted

    def reset(self):
        """Reset all bilateral state."""
        super().reset()
        self.current_state_left = None
        self.current_state_right = None
        self.prev_state_left = None
        self.prev_state_right = None
        self.last_count_time_left = 0
        self.last_count_time_right = 0
        self._flex_min_left     = 180.0
        self._flex_min_right    = 180.0
        self._flex_enter_left   = 0.0
        self._flex_enter_right  = 0.0
        self._flex_penalty_done = False

    def get_status(self) -> Dict[str, Any]:
        """Bilateral status info."""
        status = super().get_status()
        status.update({
            "counter_left": self.counter_left,
            "counter_right": self.counter_right,
            "state_left": self.current_state_left,
            "state_right": self.current_state_right
        })
        return status


class DurationExercise(BaseExercise):
    """
    Subclass for duration-based exercises.

    Example: Plank (hold the target position for N seconds).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.target_duration = config.get("target_duration", 30)  # seconds
        self.current_duration = 0
        self.hold_start_time = None
        self.is_holding = False
        self.hold_state = config.get("hold_state", "hold")

    def update_duration(self, context: Dict[str, Any]) -> float:
        """Update the hold duration."""
        # Update the FSM state
        self.update_state(context)

        if self.current_state == self.hold_state:
            if not self.is_holding:
                self.hold_start_time = time.time()
                self.is_holding = True
            else:
                self.current_duration = time.time() - self.hold_start_time
        else:
            self.is_holding = False
            # Bump the counter if the target duration was reached
            if self.current_duration >= self.target_duration:
                self.counter += 1
            self.current_duration = 0

        return self.current_duration

    def reset(self):
        """Reset all duration state."""
        super().reset()
        self.current_duration = 0
        self.hold_start_time = None
        self.is_holding = False

    def get_status(self) -> Dict[str, Any]:
        """Duration status info."""
        status = super().get_status()
        status.update({
            "current_duration": self.current_duration,
            "target_duration": self.target_duration,
            "is_holding": self.is_holding
        })
        return status
