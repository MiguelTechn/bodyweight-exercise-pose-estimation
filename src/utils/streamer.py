import sys #Sirve para comunicarse con el intérprete de python. Por ejemplo, para ejecutar el script desde el terminal
import os #Para hablar con el sistema operativo. Sirve para abrir carpetas, eliminarlas, modificar ficheros (añadir archivos...), etc.
from pathlib import Path
import cv2 as cv
import numpy as np

class VideoStreamer:
    """Opens and process videos
    
    Atributtes:
        cap (VideoCapture)
    """
    
    def __init__(self, source: int | str | Path):
        """VideoStreamer constructor

        Args:
            source (int | Path): VideoStreamer video source

        Raises:
            ValueError: The vídeo can't be opened Path can't be oppened
        """
        self.cap = cv.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self.cap.isOpened():
            raise ValueError(f"The vídeo can't be oppened: {source}")

    def get_frame(self):
        """getter of current frame image

        Returns:
            image: frame image
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            return None
        return frame

    def get_fps(self):
        """fps getter

        Returns:
            float: video fps
        """
        return self.cap.get(cv.CAP_PROP_FPS)
    
    def get_timestamp(self):
        """Timestamp of video capture getter

        Returns:
            float: video timestamp
        """
        return self.cap.get(cv.CAP_PROP_POS_MSEC)


    def close(self):
        """Close and release of resources
        """
        self.cap.release()