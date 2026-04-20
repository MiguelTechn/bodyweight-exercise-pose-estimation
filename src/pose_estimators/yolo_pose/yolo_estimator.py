from src.pose_estimators.base_estimator import BasePoseEstimator
import torch

from pathlib import Path
from ultralytics import YOLO

class YoloPoseEstimator(BasePoseEstimator):
    """Class for pose estimation using the Ultralytics YOLOv8-pose model.

    It acts as a wrapper on top of the Ultralytics YOLO API. It leverages PyTorch
    under the hood to perform fast inference on video frames, extracting 2D keypoints
    for anatomical landmarks.

    Attributes:
        model (ultralytics.YOLO): The loaded YOLO pose estimation model.
    """

    def __init__(self, model_path: str | Path = Path(__file__).parent.parent.parent.parent / 'models' / 'yolo' / 'yolo26n-pose.pt'):
        """YoloPoseEstimator constructor.

        Args:
            model_path (str | Path, optional): Path to the YOLO model's weights file (.pt).
                By default, it points to 'yolo26n-pose.pt' in the models directory.
        """
        super().__init__()
        self.model = YOLO(str(model_path))

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'model'):
            del self.model
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if exc_type is not None:
            print(f"ERROR: {exc_type} {exc_value} {traceback}")

    def process_frame(self, frame, stream: bool = False, **kwargs):
        """Processes a single frame to extract pose landmarks.

        Args:
            frame (numpy.ndarray): The image frame matrix (typically in RGB/BGR format).
            **kwargs: Additional keyword arguments (ignored in this implementation).

        Returns:
            list[ultralytics.engine.results.Results]: Results objects containing the detected 
            bounding boxes and keypoints (due to stream=True).
        """
        # If stream is True YOLO manages video more efficiently internally, but it reduces our control.
        # Enable this feature when you want to automatically paint over video or frames.
        if stream is True:
            return self.model.predict(frame, stream = True, show = True)
        else:
            return self.model.predict(frame, stream= False)

    def close(self):
        """
        Releases the model's resources.
        
        Although Python uses Garbage Collection, PyTorch retains VRAM on the GPU.
        We explicitly empty the CUDA cache to prevent OutOfMemory (OOM) errors 
        when switching between models during testing.
        """
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()