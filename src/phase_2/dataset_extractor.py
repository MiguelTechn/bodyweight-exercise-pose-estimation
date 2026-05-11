import cv2 as cv
import numpy as np
import pandas as pd

from pathlib import Path
from src.utils.streamer import VideoStreamer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
import src.constants as c

def angle_calculation(landmark, k_name, k_value):
    # Vector calculation (Vert - AdjacentJoint)
    k_point1 = c.MEDIAPIPE_POSE_MAP[c.ADJACENCY_MAP[k_name][0]]
    vert_point1 = np.array([landmark[k_value].x, landmark[k_value].y, landmark[k_value].z]) - \
        np.array([landmark[k_point1].x, landmark[k_point1].y, landmark[k_point1].z])
    
    k_point2 = c.MEDIAPIPE_POSE_MAP[c.ADJACENCY_MAP[k_name][1]]
    vert_point2 = np.array([landmark[k_value].x, landmark[k_value].y, landmark[k_value].z]) - \
        np.array([landmark[k_point2].x, landmark[k_point2].y, landmark[k_point2].z])
    
    # Scalar product 
    dot_product = np.dot(vert_point1, vert_point2)

    # Vector magnitudes
    magnitude1 = np.linalg.norm(vert_point1)
    magnitude2 = np.linalg.norm(vert_point2)

    # Cos vertex calculation
    cos = dot_product / (magnitude1 * magnitude2)
    cos = np.clip(cos, -1.0, 1.0)

    # Angle calculation
    angle_rad = np.arccos(cos)
    return np.degrees(angle_rad)

def geometric_normalization(results):
    if not results.pose_world_landmarks:
        print("No pose landmarks detected.")
        return None
    
    normalized_landmarks = []   
    
    # Shoulders center calculations
    left_shoulder = np.array([
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['left_shoulder']].x,
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['left_shoulder']].y,
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['left_shoulder']].z
    ])
    
    right_shoulder = np.array([
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['right_shoulder']].x,
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['right_shoulder']].y,
        results.pose_world_landmarks[0][c.MEDIAPIPE_POSE_MAP['right_shoulder']].z
    ])

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0

    # Torso len (MidShoulder(x,y,z) - MidHip(0,0,0))
    torso_len = np.linalg.norm(mid_shoulder)

    landmark = results.pose_world_landmarks[0]

    for k_name, k_value in c.MEDIAPIPE_POSE_MAP.items():
        # Keypoint presence threshold. If it is less than 0.5, discard the frame
        if landmark[k_value].presence < 0.5:
            return None

        # Angles calculations        
        if k_name in c.ADJACENCY_MAP.keys():
            normalized_landmarks.append(angle_calculation(landmark, k_name, k_value))

        # Keypoint normalization
        xn = landmark[k_value].x / torso_len
        yn = landmark[k_value].y / torso_len
        zn = landmark[k_value].z / torso_len

        normalized_landmarks.extend([xn, yn, zn, landmark[k_value].visibility])
    
    return normalized_landmarks

def main():
    
    # ___________Column names____________
    columm_names = ["video_id", "frame_index"]
    for k_name, k_value in c.MEDIAPIPE_POSE_MAP.items():
        if k_name in c.ADJACENCY_MAP.keys():
            columm_names.append(f"angle_{k_name}")
        columm_names.extend([f"x_{k_value}", f"y_{k_value}", f"z_{k_value}", f"v_{k_value}"])
    columm_names.append("class")
    #______________________________________

    dataset_data = []

    # Path to storage results
    final_results_path = Path(__file__).parent.parent.parent / 'data' / 'processed'
    final_results_path.mkdir(parents=True, exist_ok=True)

    #Path to videos
    video_path = Path(__file__).parent.parent.parent / 'data' / 'raw'
    folder = ['PushUps', 'BodyWeightSquats', 'Other']
    folder_video_path = [video_path / folder[0], video_path / folder[1], video_path / folder[2]]

    for path in folder_video_path:
        print(f"|-Folder {path.parent.name}")
        for video in path.iterdir():
            print(f"|----Processing {video.name}")
            if video.suffix.lower() not in ['.mp4', '.avi']: continue

            # Video inicialization
            videoStreamer = VideoStreamer(video)

            # _____________FPS management_______________
            video_fps = videoStreamer.get_fps()

            # Calculation of how many frames to skip. The target is 30 fps
            frames_jump = round(video_fps / 30)

            # If the original video is 30fps it doesn't jump any frame
            if frames_jump < 1:
                frames_jump = 1 

            #___________________________________________

            # Model inicialization
            with MediaPipePoseEstimator(video_processing = 'video') as ia:
                prev_timestamp = -1
                while True:
                    frame = videoStreamer.get_frame()
                    if frame is None: break
                    
                    # Frame counter and fps management
                    frame_index = videoStreamer.get_frame_position()
                    if frame_index % frames_jump != 0: continue

                    # Set color format to RGB
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    
                    # Timestamp for mediapipe  
                    timestamp_ms = int(videoStreamer.get_timestamp())
                    if timestamp_ms <= prev_timestamp:
                        timestamp_ms = prev_timestamp + 1
                    
                    prev_timestamp = timestamp_ms
                    
                    # Frame processing
                    results = ia.process_frame(rgb_frame, timestamp_ms)

                    # Geometric normalization and angles calculations                    
                    keypoints_list = geometric_normalization(results)
                    if keypoints_list is None: continue

                    line = [video.stem, frame_index] + keypoints_list + [video.parent.name]

                    dataset_data.append(line)
                    
            videoStreamer.close()
    
    df = pd.DataFrame(dataset_data, columns=columm_names)
    df.to_csv(final_results_path / 'dataset.csv', index=False)
                    
if __name__ == "__main__":
    main()