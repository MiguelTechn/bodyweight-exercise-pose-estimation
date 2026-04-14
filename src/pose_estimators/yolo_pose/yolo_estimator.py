from src.pose_estimators.base_estimator import BasePoseEstimator

import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YoloPoseEstimator(BasePoseEstimator):
    def __init__(self, model_path: str | Path = Path(__file__).parent.parent.parent.parent / 'models' / 'yolo' / 'yolo26n-pose.pt'):
        super().__init__()
        self.model = YOLO(str(model_path))

    def process_frame(self, frame, **kwargs):
        results =  self.model.predict(frame, stream= True)
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                return result.keypoints.xyn
        return None

    def close(self):
        pass