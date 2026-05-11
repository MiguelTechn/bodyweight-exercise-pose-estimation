MEDIAPIPE_POSE_MAP = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}

YOLO_POSE_MAP = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

COCO_POSE_MAP = {
    "left_shoulder": 6,
    "right_shoulder": 7,
    "left_elbow": 8,
    "right_elbow": 9,
    "left_wrist": 10,
    "right_wrist": 11,
    "left_hip": 12,
    "right_hip": 13,
    "left_knee": 14,
    "right_knee": 15,
    "left_ankle": 16,
    "right_ankle": 17
}

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), # Arms
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), # Upper body
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle") # leftegs
]

ADJACENCY_MAP = {
    "left_shoulder": ["left_hip", "left_elbow"],
    "right_shoulder": ["right_hip", "right_elbow"],
    "left_elbow": ["left_shoulder", "left_wrist"],
    "right_elbow": ["right_shoulder", "right_wrist"],
    "left_hip": ["left_shoulder", "left_knee"],
    "right_hip": ["right_shoulder", "right_knee"],
    "left_knee": ["left_hip", "left_ankle"],
    "right_knee": ["right_hip", "right_ankle"]
}