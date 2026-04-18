import cv2 as cv
import json
import numpy as np
from pathlib import Path

import src.constants as c
from src.pose_estimators.mediapipe_pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_estimators.yolo_pose.yolo_estimator import YoloPoseEstimator

def extract_metrics():

    # Path to storage results
    results_ann_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations'
    results_ann_path.mkdir(parents=True, exist_ok=True)

    results_ann_path = results_ann_path / 'results_pose_models.json'
    #results_ann_path.write_text('[]')

    final_json = []

    # Path to coco test images
    test_image_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'images_coco'

    # Path to coco test annotations
    test_image_annotations_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'filtered_person_keypoints_val2017.json'
    # Json coco test
    test_image_annotations_json = json.loads(test_image_annotations_path.read_text())

    # Images and annotations fields from json to iterate through test images
    images = test_image_annotations_json['images']

    # Models inicialization
    mediapipe = model_init('mediapipe')
    yolo = model_init('yolo')
    
    for image_it in range(len(images)):
        # Extract image .jpg from test_image_path
        image_file = test_image_path / images[image_it]['file_name']

        # Image reading
        image_readed = cv.imread(str(image_file))
        if image_readed is None: continue

        # Set color format to RGB
        rgb_image = cv.cvtColor(image_readed, cv.COLOR_BGR2RGB)  
        
        # Mediapipa image processing
        print(f"Precision data extraction of [mediapipe] model, for image {images[image_it]['file_name']}")
        mediapipe_results = mediapipe.process_frame(frame=rgb_image, )

        print(f"Precision data extraction of [yolo] model, for image {images[image_it]['file_name']}")
        yolo_results = yolo.process_frame(frame=rgb_image)
        
        if mediapipe_results and yolo_results:
            image_details = [images[image_it]['id'], images[image_it]['width'], images[image_it]['height']]
            final_json.append(generate_results_json(mediapipe_results, yolo_results, image_details))
        else:
            break
    
    try:
        mediapipe.close()
        yolo.close()
    except RuntimeError as e:
        print(f"ERROR: Error {e} closing model resources")

    cv.destroyAllWindows()

    # json creation
    results_ann_path.write_text(json.dumps(final_json, indent=4))


def model_init(model_name):
    if model_name.lower() == "mediapipe":
        return MediaPipePoseEstimator(video_processing = 'image')
    elif model_name.lower() == "yolo":
        return YoloPoseEstimator(video_processing = 'image')
    else:
        raise NotImplementedError(f"The model {model_name} is not implemented yet.")
    
def generate_results_json(mediapipe_results, yolo_results, image_details):

    mediapipe_keypoints = []
    # List creation [x,y,visibility/confidence]
    for landmark in mediapipe_results.pose_landmarks:
        for keypoint in c.MEDIAPIPE_POSE_MAP.values():
            mediapipe_keypoints.append([landmark[keypoint].x * image_details[1], landmark[keypoint].y * image_details[2], landmark[keypoint].visibility])
    
    yolo_keypoints = []
    for result in yolo_results:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            landmarks = result.keypoints.xy.cpu().numpy()
            confidence = result.keypoints.conf.cpu().numpy()
            #It filter only one person detected
            x = landmarks[0, :, 0].astype(float)
            y = landmarks[0, :, 1].astype(float)
            conf = confidence[0, :].astype(float)

            for keypoint in c.YOLO_POSE_MAP.values():
                yolo_keypoints.append([x[keypoint] , y[keypoint], conf[keypoint]])

    # data
    data = {
            "image_id": image_details[0],
            "mediapipe": {
                "keypoints": [
                    mediapipe_keypoints
                ]
            },
            "yolo": {
                "keypoints": [
                    yolo_keypoints
                ]
            }
        }
    return data

if __name__ == "__main__":
    extract_metrics()