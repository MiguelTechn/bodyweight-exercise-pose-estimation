import numpy as np

from src.utils.ema_filter import EMASmoother

import src.constants as c

class PoseNormalizer:
    def __init__(self, apply_smoothing=True):
        self.smoother = EMASmoother() if apply_smoothing else None
        self.torso_smoother = EMASmoother(frames_number=39) if apply_smoothing else None # alpha ≈ 0.05
    
    def _calculate_angle(self, landmarks_matrix, k_name, k_value):
        k_name_list = list(c.MEDIAPIPE_POSE_MAP.keys())
        # Vector calculation (Vert - AdjacentJoint)
        k_point1 = k_name_list.index(c.ADJACENCY_MAP[k_name][0])
        vert_point1 = landmarks_matrix[k_value, :3] - landmarks_matrix[k_point1, :3]

        k_point2 = k_name_list.index(c.ADJACENCY_MAP[k_name][1])
        vert_point2 = landmarks_matrix[k_value, :3] - landmarks_matrix[k_point2, :3]

        # Scalar product 
        dot_product = np.dot(vert_point1, vert_point2)

        # Vector magnitudes
        magnitude1 = np.linalg.norm(vert_point1)
        magnitude2 = np.linalg.norm(vert_point2)

        # Cos vertex calculation
        cos = dot_product / ((magnitude1 * magnitude2) + 1e-6)
        cos = np.clip(cos, -1.0, 1.0)

        # Angle calculation
        angle_rad = np.arccos(cos)
        return np.degrees(angle_rad)

    def process_and_normalize(self, results):
        if not results.pose_world_landmarks:
            return None
            
        mp_landmarks = results.pose_world_landmarks[0]
        
        # Numpy Matrix [x,y,z,visibility,presence]
        landmarks_matrix = np.zeros((len(c.MEDIAPIPE_POSE_MAP.values()) + 1, 5))
        # Filling Numpy Matriz
        for i, k_value in enumerate(c.MEDIAPIPE_POSE_MAP.values()):
            lm = mp_landmarks[k_value]
            landmarks_matrix[i] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

        #_________Invalid frame management________  
        # Keypoints indexes
        l_hip_index = list(c.MEDIAPIPE_POSE_MAP.keys()).index("left_hip")
        r_hip_index = list(c.MEDIAPIPE_POSE_MAP.keys()).index("right_hip")
        
        # Visibility threshold. If left_hip or right_hip visibility is less than 0.3 return None
        if (landmarks_matrix[l_hip_index, 3] < 0.3) or (landmarks_matrix[r_hip_index, 3] < 0.3):
            return None

        
        # Presence threshold. If more than 3 keypoints are out the frame (less than 0.5), it discard the actual frame and assign the previous if it has.
        if (np.sum(landmarks_matrix[:, 4] < 0.5) > 3):
            return None
        #__________________________________________

        # EMA Smoothing [x,y,z]
        if self.smoother is not None:
            landmarks_matrix[:, :3] = self.smoother.smooth_frame(landmarks_matrix[:, :3])

        # ________________Spatial normalization___________________
        l_shoulder_index = list(c.MEDIAPIPE_POSE_MAP.keys()).index("left_shoulder")
        r_shoulder_index = list(c.MEDIAPIPE_POSE_MAP.keys()).index("right_shoulder")

        mid_shoulder = (landmarks_matrix[l_shoulder_index, :3] + landmarks_matrix[r_shoulder_index, :3]) / 2.0
        mid_hip = (landmarks_matrix[l_hip_index, :3] + landmarks_matrix[r_hip_index, :3]) / 2.0
        
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        
        # Temporal stabilization of the scale factor (Dedicated EMA filter)
        # Avoids "Skeleton Breathing". We use 39 frames for a smooth alpha (0.05)
        if self.torso_smoother is not None:
            torso_length = float(self.torso_smoother.smooth_frame(torso_length))
            
        if torso_length > 1e-6: # div by 0 protection
            landmarks_matrix[:, :3] = landmarks_matrix[:, :3] / torso_length
        #_________________________________________________________

        # ________________Angles calculations____________________
        world_landmarks = []
        for k_index, k_name in enumerate(c.MEDIAPIPE_POSE_MAP.keys()):
            if k_name in c.ADJACENCY_MAP.keys():
                world_landmarks.append(self._calculate_angle(landmarks_matrix, k_name, k_index))
                
            world_landmarks.extend(landmarks_matrix[k_index, :4].tolist()) # [Add x, y, z, visibility]
        #_________________________________________________________

        return world_landmarks
