from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipePoseEstimator:

    def __init__(self, model_path: str | Path = Path(__file__).parent.parent.parent / 'models' / 'mediapipe' / 'pose_landmarker_heavy.task'):
        """MediaPîpePoseEstimator constructor

        Args:
            model_path (str | Path, optional): _description_. Defaults to Path(__file__).parent.parent.parent/'models'/'pose_landmarker_full.task'.
        """
        # Convertimos model_path a string para evitar errores de codificación en el backend de C++ de MediaPipe
        base_options = python.BaseOptions(model_asset_path=str(model_path))

        landmarker_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False
        )

        self.model = vision.PoseLandmarker.create_from_options(landmarker_options)

    def process_frame(self, frame, timestamp_ms: int = 1):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  
        return self.model.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.model.close()