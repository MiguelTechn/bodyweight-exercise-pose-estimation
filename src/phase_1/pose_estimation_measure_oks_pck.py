from pathlib import Path
import json
import numpy as np
import pandas as pd

import src.constants as c

def calculate_oks(euclidean_distances, scale, visibilities):
    # Sigmas extracted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    sigmas = np.array([.79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    # You can find the OKS formula in README.md
    return (np.exp((-np.square(euclidean_distances))/(2 * np.square(scale) * np.square(sigmas))) * visibilities) / visibilities

    

def calculate_pck(euclidean_distances, gt_keypoints, tolerance = 0.2):
    # -6 to get the first element of gt_keypoints -6 = - c.COCO_POSE_MAP.values()[0]
    left_shoulder_index = c.COCO_POSE_MAP["left_shoulder"] - 6
    right_hip_index = c.COCO_POSE_MAP["right_hip"] - 6

    d_norm = np.linalg.norm(gt_keypoints[left_shoulder_index] - gt_keypoints[right_hip_index])

    # You can find the PCK formula in README.md
    return (euclidean_distances / d_norm <= tolerance).astype(int) 

def main():
    # List to storage results
    results = []

    # Path to storage results
    final_results_path = Path(__file__).parent.parent.parent / 'results' / 'phase_1'
    final_results_path.mkdir(parents=True, exist_ok=True)

    # Path to coco test annotations
    coco_annotations_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'filtered_person_keypoints_val2017.json'
    
    # Path to pose results
    pose_results_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'results_pose_models.json'

    # Json coco test
    coco_annotations = json.loads(coco_annotations_path.read_text())
    coco_annotations_ann = coco_annotations['annotations']

    # Json pose results
    pose_results = json.loads(pose_results_path.read_text())

    # To get COCO_POSE_MAP keypoints names
    coco_pose_map_inverted = [k for k, v in c.COCO_POSE_MAP.items()]

    for image_it in range(len(coco_annotations_ann)):
        # Extracting and filtering keypoints
        xyv_mediapipe = np.array(pose_results[image_it]['mediapipe']['keypoints']) \
            if len(pose_results[image_it]['mediapipe']['keypoints']) == 12 else np.full((12, 3), np.nan)
        xyv_yolo = np.array(pose_results[image_it]['yolo']['keypoints']) \
            if len(pose_results[image_it]['yolo']['keypoints']) == 12 else np.full((12, 3), np.nan)
        
        # Euclidean distance calculations
        pred_mediapipe_xy = np.array(xyv_mediapipe)[:, :2]
        pred_yolo_xy = np.array(xyv_yolo)[:, :2]
        gt_xy = np.array(coco_annotations_ann[image_it]['keypoints'])[:, :2]

        euclidean_distance_mediapipe = np.linalg.norm(pred_mediapipe_xy - gt_xy, axis=1)
        euclidean_distance_yolo = np.linalg.norm(pred_yolo_xy - gt_xy, axis=1)

        # Scale calculation
        scale = np.sqrt(coco_annotations['images'][image_it]['width'] * coco_annotations['images'][image_it]['height'])

        # Visibility filter calculation
        pred_mediapipe_v = xyv_mediapipe[:, 2:3]
        pred_yolo_v = xyv_yolo[:, 2:3]

        pred_mediapipe_v = [1 if k > 0.5 else 0 for k in pred_mediapipe_v]
        pred_yolo_v = [1 if k > 0.5 else 0 for k in pred_yolo_v]

        #OKS calculations
        mediapipe_oks = calculate_oks(euclidean_distance_mediapipe, scale, pred_mediapipe_v)
        yolo_oks = calculate_oks(euclidean_distance_yolo, scale, pred_yolo_v)

        #PCK calculations
        mediapipe_pck = calculate_pck(euclidean_distance_mediapipe, gt_xy, 0.2)
        yolo_pck = calculate_pck(euclidean_distance_yolo, gt_xy, 0.2)

        for keypoint_it in range(len(coco_annotations_ann[image_it]['keypoints'])):
            image_id = coco_annotations_ann[image_it]['image_id']
            
            # Element 6 is the first element of COCO_POSE_MAP
            keypoint_name = coco_pose_map_inverted[keypoint_it]
            
            #Mediapipe line
            results.append({
                "image_id": image_id,
                "model_name": "mediapipe",
                "keypoint_name": keypoint_name,
                "visibility":pred_mediapipe_v[keypoint_it],
                "distance": euclidean_distance_mediapipe[keypoint_it],
                "oks_score": mediapipe_oks[keypoint_it],
                "pck": mediapipe_pck[keypoint_it]
            })
            #Yolo line
            results.append({
                "image_id": image_id,
                "model_name": "yolo",
                "keypoint_name": keypoint_name,
                "visibility":pred_yolo_v[keypoint_it],
                "distance": euclidean_distance_yolo[keypoint_it],
                "oks_score": yolo_oks[keypoint_it],
                "pck": yolo_pck[keypoint_it]
            })

    results_dataframe = pd.DataFrame(results)
    print(results_dataframe[results_dataframe['image_id'] == 492758])
            # image_id, model_name, keypoint_name, visibility, distance, oks_score, pck
    results_dataframe.to_csv(final_results_path / 'results_oks_pck.csv', index=False)



if __name__ == "__main__":
    main()
