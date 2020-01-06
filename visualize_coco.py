#!/usr/bin/env python3
# coding=utf-8
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

image_directory = '/home/harman-jx/work/landslide/test/jpeg/'
annotation_file = '/home/harman-jx/work/landslide/test/landslide_val_google_20191115.json'

example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

#exit()

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

#exit()

category_ids = example_coco.getCatIds(catNms=['landslide'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
#for image_id in image_ids:
#    image_data = example_coco.loadImgs(image_id)[0]
#    print(image_data)

#exit()

# load and display instance annotations
image = io.imread(image_directory + image_data['file_name'])
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
for annotation in annotations:
    print(annotation)
    segmentation = annotation['segmentation']
    n = len(segmentation[0])
    x = []
    y = []
    for i in range(int(n/2)):
        x.append(segmentation[0][i*2])
        y.append(segmentation[0][i*2+1])
    plt.scatter(x, y, color = 'red')
example_coco.showAnns(annotations)
plt.show()
