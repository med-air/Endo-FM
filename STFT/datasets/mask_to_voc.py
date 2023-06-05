#!/usr/bin/python
# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
from pascal_voc_io import PascalVocWriter
import os
import sys
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import cv2
import numpy as np


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def create_sub_mask_annotation(sub_mask, width, height, offset_x=0, offset_y=0):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. 
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []

    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, 2)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1) #n*[w,h]
        for temp in contour:
            temp[0] += offset_x
            if temp[0]<=0:
                temp[0]=1
            if temp[0]>=width:
                temp[0]=width-1
            temp[1] += offset_y
            if temp[1]<=0:
                temp[1]=1
            if temp[1]>=height:
                temp[1]=height-1

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    min_x, min_y, max_x, max_y = multi_poly.bounds

    return (min_x, min_y, max_x, max_y)


if __name__ == "__main__":
    dataname = 'asuvideo'
    category = 'polyp'

    img_name = 'xxxx.png'
    mask_name = 'xxxx_mask.png'
    read_path = 'ASUVideo/images/subdir'
    save_path = 'ASUVideo/Annotations/subdir-annos'

    # Read image
    image = cv2.imread(os.path.join(read_path, img_name))
    imageShape = [image.shape[0], image.shape[1], image.shape[2]]
    writer = PascalVocWriter('subdir-annos', img_name, imageShape, databaseSrc=dataname)

    # Read mask
    gt_mask = cv2.imread(os.path.join(read_path, mask_name))
    gt_mask = gt_mask[:,:,0]
    gt_mask[gt_mask==255]=1
    assert image.shape[0]==gt_mask.shape[0]
    assert image.shape[1]==gt_mask.shape[1]
    if gt_mask.sum()==0:
        writer.addBndBox(0, 0, 0, 0, category, 0)
    else:
        label_mask = measure.label(gt_mask)
        for i in range(1, label_mask.max()+1):
            if (label_mask==i).sum() < 50:
                # skip the small connected domains, they may be label noise.
                continue
            sub_mask = (label_mask==i)*1
            min_x, min_y, max_x, max_y = create_sub_mask_annotation(sub_mask, width, height)
            writer.addBndBox(min_x, min_y, max_x, max_y, category, 1)

    writer.save(targetFile=os.path.join(save_path, img_name.replace('.png', '.xml')))