import cv2 as cv
from pathlib import Path
from src.utils.streamer import VideoStreamer
from src.utils.visualizer import PoseVisualizer
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_estimators.yolo_pose.yolo_estimator import YoloPoseEstimator
#from mediapipe.- import 
#from yolo.- import 
#from openpose.- import 

def extract_metrics():

    models_test = ["mediapipe"]

    for model_name in models_test:

        print(f"Performance test of {model_name} model")

        # TODO: Iterar sobre un dataset de pruebas completo estructurado por vistas (frontal, lateral, oclusiones)
        #To get the path to video test
        test_video_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'fase_1_test' / 'squat_test1.mp4'

        # Video inicialization
        video = VideoStreamer(test_video_path)

        # Visualizer inicialization
        visualizer = PoseVisualizer()

        # Model inicialization
        if model_name.lower() == "mediapipe":
            ia = MediaPipePoseEstimator()
        elif model_name.lower() == "yolo":
            ia = YoloPoseEstimator()
        else:
            raise NotImplementedError(f"The model {model_name} is not implemented yet.")
        
        

        #Original video fps
        video_fps = video.get_fps()
        print("Original video fps:", video_fps)

        # Streaming window behavior
        cv.namedWindow(f"frame {model_name}", cv.WINDOW_NORMAL)
        
        while True:
            frame = video.get_frame()
            if frame is None: break

            # Set color format to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Timestamp for mediapipe
            timestamp_ms = int(video.get_timestamp())
            
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

        try:
            ia.close()
        except RuntimeError as e:
            print(f"ERROR: Error {e} closing resources {model_name}")
            
        video.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    extract_metrics()