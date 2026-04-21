import cv2 as cv
from pathlib import Path
from src.utils.streamer import VideoStreamer
from src.utils.visualizer import PoseVisualizer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_estimators.yolo_pose.yolo_estimator import YoloPoseEstimator

def model_init(model_name):
    if model_name.lower() == "mediapipe":
        return MediaPipePoseEstimator(video_processing = 'video')
    elif model_name.lower() == "yolo":
        return YoloPoseEstimator()
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented yet.")

def main():

    models_test = ["mediapipe", "yolo"]

    # Visualizer inicialization
    visualizer = PoseVisualizer()

    #To get the path to video test
    test_video_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'fase_1_test'

    for video in test_video_path.iterdir():        
        if video.suffix.lower() in ['.mp4', '.avi']:
            # Video inicialization
            videoStreamer = VideoStreamer(video)
            for model_name in models_test:
                # Model inicialization
                with model_init(model_name) as ia:
                    print(f"Oclusion test of {model_name} model")
                    # Streaming window behavior
                    cv.namedWindow(f"frame {model_name}", cv.WINDOW_NORMAL)
                    
                    while True:
                        frame = videoStreamer.get_frame()
                        if frame is None: break

                        # Set color format to RGB
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        
                        # Timestamp for mediapipe
                        timestamp_ms = int(videoStreamer.get_timestamp())
                        
                        # Frame processing
                        results = ia.process_frame(frame=rgb_frame, timestamp_ms=timestamp_ms) 
                        
                        # Frame drawing
                        final_frame = visualizer.draw_on_image(model_name, rgb_frame, results)

                        # Set color format to BGR
                        procesed_frame = cv.cvtColor(final_frame, cv.COLOR_RGB2BGR)

                        # Show frame
                        cv.imshow(f"frame {model_name}", procesed_frame)
                        
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
                cv.destroyAllWindows()

                # Rewind to the beginning of the video
                videoStreamer.video_rewind()
            videoStreamer.close()

if __name__ == "__main__":
    main()