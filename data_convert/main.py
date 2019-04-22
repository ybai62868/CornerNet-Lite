import json
import os
import os.path as osp
import cv2


root_path = '/home/yunfanlu/project/CornerNet-Lite-master/data/dac'

phase = 'train'


# dataset has 5 keys, info, licenses, images, annotations, categories.
dataset = {}
dataset['images'] = []
dataset['categories'] = []
dataset['annotations'] = []

with open(osp.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()


for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})


indexes = [f for f in os.listdir(osp.join(root_path, 'images'))]





with open(osp.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

for k, index in enumerate(indexes):
    img = cv2.imread(osp.join(root_path, 'images/') + index)
    height, width, _ = img.shape

    dataset['images'].append({'file_name': index, 
                             'id': k,
                             'width': width,
                             'height:' height})
    
    for ii, anno in enumerate(annos):
        parts = anno.strip().split()
        if parts[0] == index:
            cls_id = parts[1]
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': i,
                'image_id': k,
                'is_crowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })








folder = os.path.join(root_path, 'annotations')
if not osp.exists(folder):
    os.makedirs(folder)

json_name = osp.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
