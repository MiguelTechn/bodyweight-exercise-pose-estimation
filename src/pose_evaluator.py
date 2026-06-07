import src.constants as c
import numpy as np

class PoseEvaluator:
    def __init__(self, col_names):
        self.col_names = col_names
        self.angle_index_map = {clave: i for i, clave in enumerate(c.ADJACENCY_MAP.keys())}
        self._dispatch_table = {
            "pushups": self._pushup_evaluator,
            "bodyweightsquats": self._bodyweightsquat_evaluator
        }

    def evaluate(self, keypoints_list, class_name):
        func = self._dispatch_table.get(class_name.lower())
        
        if func:
            col_list = []
            k_list = []

            for i, col_name in enumerate(self.col_names):
                if col_name.startswith("angle_"):
                    col_list.append(keypoints_list[i])
                else:
                    k_list.append(keypoints_list[i])

            return func(col_list, k_list)
        else:
            print(f"Error: Exercise '{class_name}' is not supported.")

    def _pushup_evaluator(self, col_list, k_list):
        # Angles evaluator
        if col_list[self.angle_index_map["left_shoulder"]] > 70 or col_list[self.angle_index_map["right_shoulder"]] > 70:
            return "Shoulder angle exceeded. Bring your elbows close to your shoulders"
        elif col_list[self.angle_index_map["left_hip"]] > 195 or col_list[self.angle_index_map["left_hip"]] < 165:
            return "Hip angle too high or too low. Flat your body"
        
        # Keypoints x,y,z Based on c.MEDIAPIPE_POSE_MAP
        l_wrist = np.array([k_list[4*4], k_list[4*4 + 1], k_list[4*4 + 2]])
        r_wrist = np.array([k_list[4*5], k_list[4*5 + 1], k_list[4*5 + 2]])
        l_shoulder = np.array([k_list[0], k_list[1], k_list[2]])
        r_shoulder = np.array([k_list[4], k_list[4 + 1], k_list[4 + 2]])

        wrist_dist = np.linalg.norm(l_wrist - r_wrist)
        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)

        # Keypoints evaluator
        if wrist_dist < (shoulder_dist * 2) * (1-0.2):
            return "Hands too close each other"
        elif wrist_dist > (shoulder_dist * 2) * 1.20:
            return "Hands too far each other"
        
        return "Good execution"

    def _bodyweightsquat_evaluator(self, col_list, k_list):
        # Angles evaluator
        if col_list[self.angle_index_map["left_knee"]] < 50 or col_list[self.angle_index_map["right_knee"]] < 50:
            return "Knee angle exceeded. Don't squat down too low"
        elif col_list[self.angle_index_map["left_hip"]] < 50 or col_list[self.angle_index_map["right_hip"]] < 50:
            return "Hip angle exceeded. Don't go down so steeply"

        # Keypoints x,y,z Based on c.MEDIAPIPE_POSE_MAP
        l_knee = np.array([k_list[4*8], k_list[4*8 + 1], k_list[4*8 + 2]])
        r_knee = np.array([k_list[4*9], k_list[4*9 + 1], k_list[4*9 + 2]])
        l_shoulder = np.array([k_list[0], k_list[1], k_list[2]])
        r_shoulder = np.array([k_list[4], k_list[4 + 1], k_list[4 + 2]])

        knee_dist = np.linalg.norm(l_knee - r_knee)
        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)

        # Keypoints evaluator
        if knee_dist < shoulder_dist * (1-0.15):
            return "Knees too close each other, in valgus position"
        elif knee_dist > shoulder_dist * 1.15:
            return "Knees too far each other"
        
        return "Good execution"