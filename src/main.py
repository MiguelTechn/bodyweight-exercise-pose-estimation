import cv2 as cv
import pandas as pd
from src.utils.streamer import VideoStreamer
from src.utils.visualizer import PoseVisualizer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_classifier.classificator import ExcersisesClassifier
from src.pose_normalizer import PoseNormalizer
from src.pose_evaluator import PoseEvaluator
import src.constants as c

def main():
    # Pose estimator inicialization
    with MediaPipePoseEstimator(video_processing='video') as pose_estimator:
        # ___________Column names____________
        columm_names = []
        for k_name, k_value in c.MEDIAPIPE_POSE_MAP.items():
            if k_name in c.ADJACENCY_MAP.keys():
                columm_names.append(f"angle_{k_name}")
            columm_names.extend([f"x_{k_value}", f"y_{k_value}", f"z_{k_value}", f"v_{k_value}"])
        #______________________________________

        # Classifier inicialization
        classifier = ExcersisesClassifier()

        # Video filming inicialization
        videoStreamer = VideoStreamer(0)

        # Visualizer inicialization
        videoVisualizer = PoseVisualizer()

        # Landmarks normalizer
        pose_normalizer = PoseNormalizer()
        
        # Pose evaluator inicialization
        pose_evaluator = PoseEvaluator(columm_names)

        # _____________FPS management_______________
        cam_fps = videoStreamer.get_fps()

        # Calculation of how many frames to skip. The target is 30 fps
        frames_jump = round(cam_fps / 30)

        # If the original video is 30fps it doesn't jump any frame
        if frames_jump < 1:
            frames_jump = 1 
        #___________________________________________

        # Sliding window list
        sliding_window = []

        f_counter = 0
        none_counter = 0
        last_valid_keypoints = None
        prev_timestamp = -1
        class_prediction = None
        feedback = None

        # Streaming window behavior
        cv.namedWindow(f"Visualizer", cv.WINDOW_NORMAL)

        while True:
            frame = videoStreamer.get_frame()
            if frame is None: break
            
            # Frame counter and fps management
            f_counter += 1
            if f_counter % frames_jump != 0: continue

            # Set color format to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Timestamp for mediapipe
            timestamp_ms = int(videoStreamer.get_timestamp())
            if timestamp_ms <= prev_timestamp:
                timestamp_ms = prev_timestamp + 1
            prev_timestamp = timestamp_ms
            
            # Frame processing
            pe_results = pose_estimator.process_frame(frame=rgb_frame, timestamp_ms=timestamp_ms) 

            # Frame drawing
            final_frame = videoVisualizer.draw_on_image("mediapipe", rgb_frame, pe_results)

            # Set color format to BGR
            procesed_frame = cv.cvtColor(final_frame, cv.COLOR_RGB2BGR)

            # Feedback
            if class_prediction is not None:
                if class_prediction.lower() != "other":
                    cv.putText(procesed_frame, class_prediction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (3, 186, 252), 2)

                if feedback is not None:
                    cv.putText(procesed_frame, feedback, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (252, 186, 3), 1)
            
            if feedback is None:
                cv.putText(procesed_frame, "No feedback - Do PushUps or BodyWeightSquats", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (252, 186, 3), 1)


            # Show frame
            cv.imshow(f"Visualizer", procesed_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            # Geometric normalization and angles calculations                    
            keypoints_list = pose_normalizer.process_and_normalize(pe_results)
            if keypoints_list is None: 
                none_counter += 1
                print(f"|--------Returned None Frame: {videoStreamer.get_frame_position()}. None Counter: {none_counter}")
                #If 0.10sec between valid frames discard window
                if none_counter >= 5:
                    print(f"|--------Discarding window")
                    none_counter = 0
                    sliding_window = []
                    last_valid_keypoints = None
                    pose_normalizer = PoseNormalizer() # EMA and oclusions reset
                    continue
                    
                # Fill with the last known valid frame.
                elif last_valid_keypoints is not None:
                    print(f"|--------Imputating last frame")
                    keypoints_list = last_valid_keypoints
                else:
                    continue # If fails in the first frame
            else:
                if class_prediction != None:
                    if class_prediction.lower() != "other":
                        feedback = pose_evaluator.evaluate(keypoints_list, class_prediction)
                    else:
                        feedback = None
                none_counter = 0
                last_valid_keypoints = keypoints_list

            
            # Sliding window management
            sliding_window.append(keypoints_list)
            
            # Sliding window management. If len is 60 reset window and classify exercise
            if len(sliding_window) >= 60:
                df_window = pd.DataFrame(sliding_window, columns=columm_names)
                class_prediction = classifier.predict(df_window)[0]
                print(class_prediction)
                sliding_window = []

        cv.destroyAllWindows()    

if __name__ == "__main__":
    main()
