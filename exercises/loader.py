"""
Exercise Loader - YAML-based exercise loader.

This module loads exercise configurations from YAML files and
instantiates the appropriate Exercise class.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Optional, List

from exercises.base_exercise import BaseExercise, BilateralExercise, DurationExercise


# Path to the exercise definitions folder
DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def load_exercise(exercise_name: str) -> BaseExercise:
    """
    Load a YAML definition by exercise name and build an Exercise instance.

    Args:
        exercise_name: Exercise name (e.g. "squat", "push_up")

    Returns:
        BaseExercise (or subclass) instance
    """
    yaml_path = DEFINITIONS_DIR / f"{exercise_name}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Exercise definition not found: {yaml_path}")

    return load_exercise_from_file(str(yaml_path))


def load_exercise_from_file(yaml_path: str) -> BaseExercise:
    """
    Load an exercise from a YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        BaseExercise (or subclass) instance
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Pick the right class based on exercise type
    exercise_type = config.get("type", "repetition")
    bilateral = config.get("bilateral", False)

    if exercise_type == "duration":
        return DurationExercise(config)
    elif bilateral:
        return BilateralExercise(config)
    else:
        return BaseExercise(config)


def get_available_exercises() -> List[str]:
    """
    List every available exercise definition.

    Returns:
        List of exercise names
    """
    if not DEFINITIONS_DIR.exists():
        return []

    exercises = []
    for file in DEFINITIONS_DIR.glob("*.yaml"):
        exercises.append(file.stem)

    return sorted(exercises)


def get_exercise_info(exercise_name: str) -> Dict:
    """
    Get UI-facing exercise metadata.

    Args:
        exercise_name: Exercise name

    Returns:
        Exercise metadata dictionary
    """
    yaml_path = DEFINITIONS_DIR / f"{exercise_name}.yaml"

    if not yaml_path.exists():
        return {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return {
        "name": config.get("display_name", exercise_name.replace("_", " ").title()),
        "type": config.get("type", "repetition"),
        "target_muscles": config.get("target_muscles", []),
        "equipment": config.get("equipment", "Bodyweight"),
        "reps": config.get("default_reps", 10),
        "sets": config.get("default_sets", 3),
        "rest_time": config.get("rest_time", "60 seconds"),
        "benefits": config.get("benefits", []),
        "difficulty": config.get("difficulty", "beginner"),
        "description": config.get("description", ""),
    }


def get_all_exercises_info() -> Dict[str, Dict]:
    """
    Get metadata for every available exercise.

    Returns:
        Dict shaped as {exercise_name: info_dict}
    """
    exercises = {}
    for name in get_available_exercises():
        exercises[name] = get_exercise_info(name)
    return exercises


def validate_exercise_config(config: Dict) -> List[str]:
    """
    Validate an exercise configuration.

    Args:
        config: Configuration loaded from YAML

    Returns:
        List of error messages (empty list means the config is valid)
    """
    errors = []

    # Required top-level fields
    required = ["name", "angles", "states", "counter"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate angles
    if "angles" in config:
        for angle_name, angle_def in config["angles"].items():
            if "points" not in angle_def:
                errors.append(f"Angle '{angle_name}' missing 'points' field")
            elif len(angle_def["points"]) != 3:
                errors.append(f"Angle '{angle_name}' must have exactly 3 points")

    # Validate states
    if "states" in config:
        for state_name, state_def in config["states"].items():
            if "condition" not in state_def:
                errors.append(f"State '{state_name}' missing 'condition' field")

    # Validate counter
    if "counter" in config:
        if "trigger_state" not in config["counter"]:
            errors.append("Counter missing 'trigger_state' field")

    return errors


# Make sure the definitions folder exists when the module loads
DEFINITIONS_DIR.mkdir(exist_ok=True)
