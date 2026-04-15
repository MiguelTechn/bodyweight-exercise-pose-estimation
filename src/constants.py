MEDIAPIPE_POSE_MAP = {
    "L_shoulder": 11,
    "R_shoulder": 12,
    "L_elbow": 13,
    "R_elbow": 14,
    "L_wrist": 15,
    "R_wrist": 16,
    "L_hip": 23,
    "R_hip": 24,
    "L_knee": 25,
    "R_knee": 26,
    "L_ankle": 27,
    "R_ankle": 28
}

YOLO_POSE_MAP = {
    "L_shoulder": 5,
    "R_shoulder": 6,
    "L_elbow": 7,
    "R_elbow": 8,
    "L_wrist": 9,
    "R_wrist": 10,
    "L_hip": 11,
    "R_hip": 12,
    "L_knee": 13,
    "R_knee": 14,
    "L_ankle": 15,
    "R_ankle": 16
}

POSE_CONNECTIONS = [
    ("L_shoulder", "R_shoulder"), ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"), ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"), # Arms
    ("L_shoulder", "L_hip"), ("R_shoulder", "R_hip"), ("L_hip", "R_hip"), # Upper body
    ("L_hip", "L_knee"), ("L_knee", "L_ankle"), ("R_hip", "R_knee"), ("R_knee", "R_ankle") # Legs
]