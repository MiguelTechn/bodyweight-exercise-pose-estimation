import src.constants as c
import numpy as np

class PoseEvaluator:
    """
    Evaluates the quality of an exercise performance based on pose data.

    This class uses a set of rule-based heuristics to provide real-time feedback
    on the user's form for specific exercises like push-ups and squats.
    It takes a list of normalized keypoints and the name of the classified exercise,
    then returns a feedback string.
    """

    def __init__(self, col_names):
        """
        Initializes the PoseEvaluator.

        Args:
            col_names (list[str]): A list of column names that corresponds to the
                                   structure of the input keypoints list. This is
                                   used to parse the incoming data correctly.
        """
        self.col_names = col_names
        self.angle_index_map = {clave: i for i, clave in enumerate(c.ADJACENCY_MAP.keys())}

        # Dispatch table to map exercise names to their corresponding evaluation function.
        # This makes the system easily extensible with new exercises.
        self._dispatch_table = {
            "pushups": self._pushup_evaluator,
            "bodyweightsquats": self._bodyweightsquat_evaluator
        }

    def evaluate(self, keypoints_list, class_name):
        """
        Main evaluation method that dispatches to the correct exercise evaluator.

        Args:
            keypoints_list (list[float]): A flat list containing angle and coordinate
                                          data for a single frame.
            class_name (str): The name of the exercise to evaluate (e.g., "PushUps").

        Returns:
            str | None: A string with feedback on the user's form, or None if the
                        exercise is not supported.
        """        
        func = self._dispatch_table.get(class_name.lower())
        
        if func:
            angle_list = []
            k_list = []

            for i, col_name in enumerate(self.col_names):
                if col_name.startswith("angle_"):
                    angle_list.append(keypoints_list[i])
                else:
                    k_list.append(keypoints_list[i])

            return func(angle_list, k_list)
        else:
            print(f"Error: Exercise '{class_name}' is not supported.")

    def _pushup_evaluator(self, angle_list, k_list):
        """Evaluates push-up form based on angles and keypoint distances."""
        # Angles evaluator
        if angle_list[self.angle_index_map["left_shoulder"]] > 70 or angle_list[self.angle_index_map["right_shoulder"]] > 70:
            return "Shoulder angle exceeded. Bring your elbows close to your shoulders"
        elif angle_list[self.angle_index_map["left_hip"]] > 195 or angle_list[self.angle_index_map["left_hip"]] < 165:
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

    def _bodyweightsquat_evaluator(self, angle_list, k_list):
        """Evaluates _bodyweightsquat_evaluator form based on angles and keypoint distances."""
        # Angles evaluator
        if angle_list[self.angle_index_map["left_knee"]] < 50 or angle_list[self.angle_index_map["right_knee"]] < 50:
            return "Knee angle exceeded. Don't squat down too low"
        elif angle_list[self.angle_index_map["left_hip"]] < 50 or angle_list[self.angle_index_map["right_hip"]] < 50:
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