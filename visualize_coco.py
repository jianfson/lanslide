#!/usr/bin/env python3
# coding=utf-8
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import ipdb
import pycocotools.mask as maskutil
from skimage import measure,draw,data


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

#image_directory = '/home/harman-jx/work/landslide/test/jpeg/'
#annotation_file = '/home/harman-jx/work/landslide/test/landslide_val_google_20191115.json'
#image_directory = '/media/jiangxin/data/lanslide/train/jpeg/'
#annotation_file = '/media/jiangxin/data/lanslide/train/landslide_train_google_20191115.json'
image_directory = '/media/jiangxin/data/lanslide/test/jpeg/'
annotation_file = '/media/jiangxin/data/lanslide/annotations/landslide_val_google_20191115.json'

bbox_test_file = '/media/jiangxin/data/lanslide/models/inference/coco_2014_minival/bbox.json'
seg_test_file = '/media/jiangxin/data/lanslide/models/inference/coco_2014_minival/segm.json'

example_coco = COCO(annotation_file)
bbox_coco=example_coco.loadRes(bbox_test_file)
seg_coco = example_coco.loadRes(seg_test_file)
#bbox_coco = COCO(bbox_test_file)
#seg_coco = COCO(seg_test_file)
#ipdb.set_trace()
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
#img_size=2000
#sum_r=0
#sum_g=0
#sum_b=0
#count=0
#for image_id in image_ids:
#    image_data = example_coco.loadImgs(image_id)[0]
#    #image = io.imread(image_directory + image_data['file_name'])
#    img=cv2.imread(image_directory + image_data['file_name'])
#    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    img=cv2.resize(img,(img_size,img_size))
#    sum_r=sum_r+img[:,:,0].mean()
#    sum_g=sum_g+img[:,:,1].mean()
#    sum_b=sum_b+img[:,:,2].mean()
#    count=count+1
##    annotation_ids = example_coco.getAnnIds(imgIds=image_id, catIds=category_ids, iscrowd=None)
##    annotations = example_coco.loadAnns(annotation_ids)
##    for annotation in annotations:
##        #if annotation['iscrowd'] == 1:
##        print('iscrowd: ' + str(annotation['iscrowd']))
#sum_r=sum_r/count
#sum_g=sum_g/count
#sum_b=sum_b/count
#img_mean=[sum_r,sum_g,sum_b]
#print (img_mean)
#exit()

# load and display instance annotations
#image = io.imread(image_directory + 'J48E022011_wow_818_zx.jpeg')
plt.figure()
image = io.imread(image_directory + image_data['file_name'])
#plt.imshow(image); plt.axis('off')
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)

bbox_annotation_ids = bbox_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
bbox_annotations = bbox_coco.loadAnns(bbox_annotation_ids)

seg_annotation_ids = seg_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
seg_annotations = seg_coco.loadAnns(seg_annotation_ids)
ipdb.set_trace()
masks = []
for annotation in annotations:
    bbox = annotation['bbox']
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 2)
    print(annotation)
    mask = example_coco.annToMask(annotation)
#area = np.sum(np.greator(mask,0))
    masks.append(mask)
    segmentation = annotation['segmentation']
    n = len(segmentation[0])
    x = []
    y = []
    for i in range(int(n/2)):
        x.append(segmentation[0][i*2])
        y.append(segmentation[0][i*2+1])
#plt.scatter(x, y, color = 'red')
plt.subplot(1,2,1)
plt.imshow(image); plt.axis('off')
plt.title('Ground Truth')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
example_coco.showAnns(annotations)
#seg_coco.showAnns(seg_annotations)
#plt.show()


image2 = io.imread(image_directory + image_data['file_name'])
for annotation in bbox_annotations:
    bbox = annotation['bbox']
    cv2.rectangle(image2, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 2)
plt.subplot(1,2,2)
plt.imshow(image2); plt.axis('off')
plt.title('prediction')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#example_coco.showAnns(annotations)
seg_coco.showAnns(seg_annotations)
plt.show()
exit()
#label = np.zeros_like(image, dtype='uint8')
label = masks[0]
for mask in masks:
    #cv2.imwrite("temp.jpeg", mask)
    #temp = cv2.imread("temp.jpeg", cv2.IMREAD_GRAYSCALE)
#label = np.zeros_like(im, dtype='uint8')
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#ret, im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)
    masked = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask = mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                label[i, j] = 1
plt.imshow(label)
plt.show()
