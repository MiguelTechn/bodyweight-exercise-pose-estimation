import cv2 as cv
import src.constants as c

class PoseVisualizer:
    def __init__(self, dot_color=(255, 0, 0), line_color=(0, 255, 0), radius = 5, dot_thickness = -1, line_thickness=2):
        self.dot_color = dot_color
        self.line_color = line_color
        self.radius = radius
        self.dot_thickness = dot_thickness
        self.line_thickness = line_thickness

        self._dispatch_table = {
            "mediapipe": self._draw_on_image_mediapipe,
            "yolo": self._draw_on_image_yolo
        }

    def draw_on_image (self, model_name, image, results):

        func = self._dispatch_table.get(model_name.lower())
        if func:
            return func(image, results)
        else:
            print(f"Error: Model '{model_name}' is not supported.")
            return image
    
    def _draw_on_image_mediapipe(self, image, results): 
        
        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return image

        height, width, _ = image.shape

        pose_points = c.MEDIAPIPE_POSE_MAP

        for landmark in results.pose_landmarks:     
            # Drawing lines
            for connection in c.POSE_CONNECTIONS:
                pose_1 = pose_points.get(connection[0])
                pose_2 =pose_points.get(connection[1])
                
                px_x1 = int(landmark[pose_1].x * width)
                px_y1 = int(landmark[pose_1].y * height)
                
                px_x2 = int(landmark[pose_2].x * width)
                px_y2 = int(landmark[pose_2].y * height)

                cv.line(image, (px_x1, px_y1), (px_x2, px_y2), self.line_color, self.line_thickness)

            # Drawing dots
            for dot_pose in pose_points.values():
                px_x = int(landmark[dot_pose].x * width)
                px_y = int(landmark[dot_pose].y * height)

                cv.circle(image, (px_x, px_y), self.radius, self.dot_color, -1)

        return image

    def _draw_on_image_yolo(self, image, results):
        pass