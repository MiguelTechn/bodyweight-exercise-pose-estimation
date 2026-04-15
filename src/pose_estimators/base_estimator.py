from abc import ABC, abstractmethod
from typing import Any

class BasePoseEstimator(ABC):
    """
    Abstract base class for all pose estimation models.
    It guarantees a unified interface for the processing pipeline.
    """
    def __init__(self):
        pass

    @abstractmethod
    def process_frame(self, frame, **kwargs) -> Any:
        """
        Frame processor
        
        Args:
            frame (numpy.ndarray): Frame image matrix.

        Returns:
            This method returns the raw results of the model used.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close and release resources
        It is essential to use this method to free up system resources.
        """
        pass
