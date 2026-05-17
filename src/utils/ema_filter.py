import numpy as np

class EMASmoother:
    def __init__(self, frames_number = 5):
        self.alpha = 2 / (frames_number + 1)
        self.ema = None 

    def smooth_frame(self, keypoints):
        if self.ema is None:
            self.ema = np.copy(keypoints)
            return self.ema
        
        self.ema = (self.alpha * keypoints) + ((1 - self.alpha) * self.ema)
        return self.ema
        
    def reset(self):
        self.ema = None