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
    "L_shoulder": 6,
    "R_shoulder": 7,
    "L_elbow": 8,
    "R_elbow": 9,
    "L_wrist": 10,
    "R_wrist": 11,
    "L_hip": 12,
    "R_hip": 13,
    "L_knee": 14,
    "R_knee": 15,
    "L_ankle": 16,
    "R_ankle": 17
}

POSE_CONNECTIONS = [
    ("L_shoulder", "R_shoulder"), ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"), ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"), # Arms
    ("L_shoulder", "L_hip"), ("R_shoulder", "R_hip"), ("L_hip", "R_hip"), # Upper body
    ("L_hip", "L_knee"), ("L_knee", "L_ankle"), ("R_hip", "R_knee"), ("R_knee", "R_ankle") # Legs
]