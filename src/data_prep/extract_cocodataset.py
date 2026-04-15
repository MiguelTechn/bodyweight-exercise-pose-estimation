from pycocotools.coco import COCO
import requests
from pathlib import Path
import os

annFile = str(Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'annotations' / 'person_keypoints_val2017.json')

#Create dir for downloaded images
image_dir = Path(__file__).parent.parent.parent / 'data' / 'coco_dataset' / 'imagenes_coco'
os.makedirs(str(image_dir), exist_ok=True)
# COCO
coco = COCO(annFile)

# Person images ids
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

counter = 0
for imgId in imgIds:
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # Only one person and keypoints
    if len(anns) == 1 and anns[0]['num_keypoints'] == 17:
        img_info = coco.loadImgs(imgId)
        url = img_info[0]['coco_url']
        file_name = img_info[0]['file_name']
        image_route = image_dir / file_name

        counter += 1
        if counter > 100:
            break

        image_request = requests.get(url)

        if image_request.status_code == 200:
            
            image_route.write_bytes(image_request.content)

        
    
