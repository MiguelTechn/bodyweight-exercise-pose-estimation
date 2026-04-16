from pycocotools.coco import COCO
import json
import requests
from pathlib import Path
from typing import Any, Dict
import src.constants as c


# You should download before person_keypoints_val2017.json from https://cocodataset.org/#download oficial webpage
annFile = str(Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'person_keypoints_val2017.json')

ann_filtered_path = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'filtered_person_keypoints_val2017.json'

#Create dir for downloaded images
image_dir = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'images_coco'
image_dir.mkdir(parents=True, exist_ok=True)

# COCO
coco = COCO(annFile)

# Person images ids
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

final_images_metadata = []
final_annotations = []

counter = 0
for imgId in imgIds:
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # Only one person and keypoints
    if len(anns) == 1 and anns[0]['num_keypoints'] == 17: # type: ignore
        img_info = coco.loadImgs(imgId)
        img_ann = coco.loadAnns(annIds)

        url = img_info[0]['coco_url'] # type: ignore
        file_name = img_info[0]['file_name']
        image_route = image_dir / file_name

        try:
            if not image_route.exists():
                image_request = requests.get(url)
                if image_request.status_code == 200:
                    image_route.write_bytes(image_request.content)
                    print(f"Downloaded {url}")
            
            final_images_metadata.extend(img_info)
            final_annotations.extend(img_ann)
            
        except Exception as e:
            print(f"{e}: Error downloading {url}")

        counter += 1
        if counter > 100:
            break
        
# Filtering annotations dataset and category cleaning 
categories: Dict[str, Any] = {k: v for k, v in coco.dataset['categories'][0].items() if k not in ['supercategory', 'id', 'name']}

# Filtering keypoints
categories['keypoints'] = [keypoint for keypoint in categories['keypoints'] if keypoint in c.COCO_POSE_MAP]

# Filtering keypoints connections
inverse_pose_map = {v: k for k, v in c.COCO_POSE_MAP.items()}
categories['skeleton'] = [
    conn for conn in categories['skeleton']
    if all(point in inverse_pose_map for point in conn)
]

# New filtered dataset
filtered_person_keypoints_val2017 = {
    "categories": categories,
    "images": final_images_metadata,
    "annotations": final_annotations
}

# Saving the file
with ann_filtered_path.open("w") as f:
    json.dump(filtered_person_keypoints_val2017, f, indent=4)