import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

class _LandmarksWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list

class _ResultsWrapper:
    def __init__(self, pose_landmarks_list):
        self.pose_landmarks = _LandmarksWrapper(pose_landmarks_list[0]) if pose_landmarks_list else None

class PoseEstimator:
    def __init__(self, static_mode=False, model_complexity=1):
        model_dir = os.path.dirname(__file__)
        model_name = _MODEL_NAMES.get(model_complexity, _MODEL_NAMES[1])
        model_path = os.path.join(model_dir, model_name)

        if not os.path.exists(model_path):
            print(f"[INFO] Downloading {model_name}...")
            urllib.request.urlretrieve(_MODEL_URLS.get(model_complexity, _MODEL_URLS[1]), model_path)
            print("[INFO] Model downloaded.")

        running_mode = vision.RunningMode.IMAGE if static_mode else vision.RunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._static_mode = static_mode
        self._start_time = time.time()

    def close(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def estimate_pose(self, frame, exercise_type):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._static_mode:
            detection_result = self._landmarker.detect(mp_image)
        else:
            timestamp_ms = int((time.time() - self._start_time) * 1000)
            detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        results = _ResultsWrapper(detection_result.pose_landmarks)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if exercise_type == "squat":
                self.draw_squat_lines(frame, lm)
            elif exercise_type == "push_up":
                self.draw_push_up_lines(frame, lm)
            elif exercise_type == "hammer_curl":
                self.draw_hammerl_curl_lines(frame, lm)

        return results

    def draw_hammerl_curl_lines(self, frame, landmarks):
        shoulder_right = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_right = [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        wrist_right = [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]

        shoulder_left = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_left = [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        wrist_left = [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]

        cv2.line(frame, shoulder_left, elbow_left, (0, 0, 255), 4, 2)
        cv2.line(frame, elbow_left, wrist_left, (0, 0, 255), 4, 2)
        cv2.line(frame, shoulder_right, elbow_right, (0, 0, 255), 4, 2)
        cv2.line(frame, elbow_right, wrist_right, (0, 0, 255), 4, 2)

    def draw_squat_lines(self, frame, landmarks):
        hip = [int(landmarks[23].x * frame.shape[1]), int(landmarks[23].y * frame.shape[0])]
        knee = [int(landmarks[25].x * frame.shape[1]), int(landmarks[25].y * frame.shape[0])]
        shoulder = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]

        hip_right = [int(landmarks[24].x * frame.shape[1]), int(landmarks[24].y * frame.shape[0])]
        knee_right = [int(landmarks[26].x * frame.shape[1]), int(landmarks[26].y * frame.shape[0])]
        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]

        cv2.line(frame, shoulder, hip, (178, 102, 255), 2)
        cv2.line(frame, hip, knee, (178, 102, 255), 2)
        cv2.line(frame, shoulder_right, hip_right, (51, 153, 255), 2)
        cv2.line(frame, hip_right, knee_right, (51, 153, 255), 2)

    def draw_push_up_lines(self, frame, landmarks):
        shoulder_left = [int(landmarks[11].x * frame.shape[1]), int(landmarks[11].y * frame.shape[0])]
        elbow_left = [int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0])]
        wrist_left = [int(landmarks[15].x * frame.shape[1]), int(landmarks[15].y * frame.shape[0])]

        shoulder_right = [int(landmarks[12].x * frame.shape[1]), int(landmarks[12].y * frame.shape[0])]
        elbow_right = [int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0])]
        wrist_right = [int(landmarks[16].x * frame.shape[1]), int(landmarks[16].y * frame.shape[0])]

        cv2.line(frame, shoulder_left, elbow_left, (0, 0, 255), 2)
        cv2.line(frame, elbow_left, wrist_left, (0, 0, 255), 2)
        cv2.line(frame, shoulder_right, elbow_right, (102, 0, 0), 2)
        cv2.line(frame, elbow_right, wrist_right, (102, 0, 0), 2)
