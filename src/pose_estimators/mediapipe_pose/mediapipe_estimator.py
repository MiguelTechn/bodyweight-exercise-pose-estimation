from src.pose_estimators.base_estimator import BasePoseEstimator

from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipePoseEstimator(BasePoseEstimator):
    """Class for pose estimation using the MediaPipe (BlazePose) model.

    It acts as a wrapper on top of the MediaPipe Tasks Vision API. It is configured 
    to run in VIDEO mode, processing sequential frames and maintaining 
    Time tracking for greater accuracy and stability in landmark detection.

    Attributes:
        model (mediapipe.tasks.python.vision.PoseLandmarker): Model instance for videoframe processing.
    """
    def __init__(self, model_path: str | Path = Path(__file__).parent.parent.parent.parent / 'models' / 'mediapipe' / 'pose_landmarker_heavy.task'):
        """MediaPipePoseEstimator constructor

        Args:
            model_path (str | Path, optional): Path to the MediaPipe model's weights file (.task).
            By default, it points to 'pose_landmarker_heavy.task' in the models directory.
        """
        super().__init__()

        base_options = python.BaseOptions(model_asset_path=str(model_path))

        landmarker_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False
        )

        self.model = vision.PoseLandmarker.create_from_options(landmarker_options)

    def process_frame(self, frame, timestamp_ms: int = 1, **kwargs):
        """Mediapipe frame processor

        Args:
            frame (numpy.ndarray): Matrix of the frame image, preferably in RGB format.
            timestamp_ms (int, optional): Frame timestamp in milliseconds. 
                It should be monotonically increasing in video mode for temporal coherence. By default it is 1.

        Returns:
            mp.tasks.vision.PoseLandmarkerResult: Object with detected landmarks 
                (2D coordinates in image and 3D in the real world).
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  
        return self.model.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """Close and release resources
        It is essential to use this method to free up system resources.
        """
        self.model.close()
        del self.model