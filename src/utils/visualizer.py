import cv2 as cv
import src.constants as c

class PoseVisualizer:
    """A utility class to visualize pose estimation landmarks on images.

    This class handles the drawing of anatomical landmarks (dots) and connections (lines) 
    for different pose estimation models using OpenCV. It uses a dispatch table to 
    support multiple models dynamically.

        Attributes:
        dot_color (tuple): Color for the landmark dots RGB format.
        line_color (tuple): Color for the connection lines RGB format.
        radius (int): Radius of the landmark dots.
        dot_thickness (int): Thickness of the dots (-1 for filled).
        line_thickness (int): Thickness of the connection lines.
    """

    def __init__(self, dot_color=(255, 0, 0), line_color=(0, 255, 0), radius = 5, dot_thickness = -1, line_thickness=2):
        """Initializes the PoseVisualizer with customizable drawing parameters.

        Args:
            dot_color (tuple, optional): Color for the landmark dots. Defaults to (255, 0, 0).
            line_color (tuple, optional): Color for the connection lines. Defaults to (0, 255, 0).
            radius (int, optional): Radius of the landmark dots. Defaults to 5.
            dot_thickness (int, optional): Thickness of the dots (-1 for filled). Defaults to -1.
            line_thickness (int, optional): Thickness of the connection lines. Defaults to 2.
        """
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
        """Dispatches the drawing process to the corresponding model's drawing function.

        Args:
            model_name (str): The name of the pose estimation model (e.g., 'mediapipe', 'yolo').
            image (array): The image frame on which to draw.
            results (Any): The output results from the pose estimation model.

        Returns:
            numpy.ndarray: The processed image with drawn landmarks, or the original image if the model is not supported.
        """

        func = self._dispatch_table.get(model_name.lower())
        if func:
            return func(image, results)
        else:
            print(f"Error: Model '{model_name}' is not supported.")
            return image
    
    def _draw_on_image_mediapipe(self, image, results): 
        """Draws MediaPipe pose landmarks and connections on the image.

        Args:
            image (numpy.ndarray): The image frame to draw on.
            results (mediapipe.tasks.vision.PoseLandmarkerResult): The results from MediaPipe Pose model.

        Returns:
            numpy.ndarray: The modified image with MediaPipe landmarks drawn.
        """
        
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
        """Draws YOLO pose landmarks and connections on the image.
        
        Args:
            image (numpy.ndarray): The image frame to draw on.
            results (Any): The results from YOLOv8-pose model.

        Returns:
            numpy.ndarray: The modified image with YOLO landmarks (Implementation pending).
        """
        """
        if results is None or len(results.xyn) == 0:
            print("No YOLO pose landmarks detected.")
            return image
        """
        height, width, _ = image.shape

        pose_points = c.YOLO_POSE_MAP


        for result in results:
            landmarks = result.keypoints.xyn.cpu().numpy()
            #It shows only one person detected
            x = landmarks[0, :, 0]
            y = landmarks[0, :, 1]

            # Drawing lines
            for connection in c.POSE_CONNECTIONS:
                pose_1 = pose_points.get(connection[0])
                pose_2 =pose_points.get(connection[1])
                
                px_x1 = int(x[pose_1] * width)
                px_y1 = int(y[pose_1] * height)
                
                px_x2 = int(x[pose_2] * width)
                px_y2 = int(y[pose_2] * height)

                cv.line(image, (px_x1, px_y1), (px_x2, px_y2), self.line_color, self.line_thickness)

            # Drawing dots
            for dot_pose in pose_points.values():
                px_x = int(x[dot_pose] * width)
                px_y = int(y[dot_pose] * height)

                cv.circle(image, (px_x, px_y), self.radius, self.dot_color, -1)
        return image