import cv2 as cv
import numpy as np
import pandas as pd
import re

from pathlib import Path
from src.utils.streamer import VideoStreamer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_normalizer import PoseNormalizer
import src.constants as c

def main():
    # ___________Column names____________
    columm_names = ["subject_id", "video_id", "window_id", "frame_window_id"]
    for k_name, k_value in c.MEDIAPIPE_POSE_MAP.items():
        if k_name in c.ADJACENCY_MAP.keys():
            columm_names.append(f"angle_{k_name}")
        columm_names.extend([f"x_{k_value}", f"y_{k_value}", f"z_{k_value}", f"v_{k_value}"])
    columm_names.append("class")
    #______________________________________

    dataset_data = [] # Final dataset list of lists

    # Path to storage results
    final_results_path = Path(__file__).parent.parent.parent / 'data' / 'processed'
    final_results_path.mkdir(parents=True, exist_ok=True)

    # Path to videos
    video_path = Path(__file__).parent.parent.parent / 'data' / 'raw'
    folder = ['PushUps', 'BodyWeightSquats', 'Other']
    folder_video_path = [video_path / folder[0], video_path / folder[1], video_path / folder[2]]

    # None counter (Invalid Window)
    none_counter = 0

    for path in folder_video_path:
        for video in path.iterdir():
            print(f"|-Folder {path.parent.name}")
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
            
            # Sliding window with overlap variables structure 
            sliding_window = []
            subject_id_match = re.search(r"([a-zA-Z_]*g\d\d)|(addsub_\d)", video.stem)
            if subject_id_match:
                subject_id = subject_id_match.group()
                action_name = video.stem[subject_id_match.end():]
            else:
                subject_id = "NA"
                action_name = "NA"
            video_name = video.name
            window_id_num = 1
            wind_id = subject_id + action_name + "_" + str(window_id_num)
            frame_window_id = 1

            # Landmarks normalizer
            pose_normalizer = PoseNormalizer()

            # For frame imputation
            last_valid_keypoints = None

            # Model inicialization
            with MediaPipePoseEstimator(video_processing = 'video') as ia:
                prev_timestamp = -1
                while True:
                    frame = videoStreamer.get_frame()
                    if frame is None: break
                    
                    # Frame counter and fps management
                    if videoStreamer.get_frame_position() % frames_jump != 0: continue

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
                    keypoints_list = pose_normalizer.process_and_normalize(results)
                    if keypoints_list is None: 
                        none_counter += 1
                        print(f"|--------Returned None Frame: {videoStreamer.get_frame_position()}. None Counter: {none_counter}")
                        #If 0.10sec between valid frames discard window
                        if none_counter >= 5:
                            print(f"|--------Discarding window: {wind_id}")
                            none_counter = 0
                            sliding_window = []
                            frame_window_id = 1
                            last_valid_keypoints = None
                            pose_normalizer = PoseNormalizer() # Reseteo de memoria EMA y Oclusiones
                            continue
                            
                        # Fill with the last known valid frame.
                        elif last_valid_keypoints is not None:
                            print(f"|--------Imputating last frame: {frame_window_id}")
                            keypoints_list = last_valid_keypoints
                        else:
                            continue # If fails in the first frame
                    else:
                        none_counter = 0
                        last_valid_keypoints = keypoints_list
                    
                    # Line structure
                    line = [subject_id, video_name, wind_id, frame_window_id] + keypoints_list + [video.parent.name]
                    
                    # Sliding window management
                    sliding_window.append(line)
                    
                    # Sliding window management. If len is 60 add sliding window to dataset and overlap 1 second (30fps)
                    if len(sliding_window) >= 60:
                        dataset_data.extend(sliding_window)

                        window_id_num += 1
                        wind_id = subject_id + action_name + "_" + str(window_id_num)
        
                        sliding_window = [[line[0], line[1], wind_id, line[3] - 30] + line[4:] for line in sliding_window[30:]]
                    
                        frame_window_id = 31
                        continue

                    frame_window_id += 1
                    
            videoStreamer.close()
    
    df = pd.DataFrame(dataset_data, columns=columm_names)
    df.to_csv(final_results_path / 'dataset.csv', index=False)
                    
if __name__ == "__main__":
    main()