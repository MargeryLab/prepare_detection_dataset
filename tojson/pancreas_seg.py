import datetime
import json
import os
import re
import fnmatch

import cv2
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = './'
DATA_DIR = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Pancreas-CT/PNG/test/ct'
ANNOTATION_TUMOR_DIR = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Pancreas-CT/PNG/test/seg_whole'

INFO = {
    "description": "Pancreas Segmentation Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "PING MENG",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 根据自己的需要添加种类
CATEGORIES = [
    {
        'id': 0,
        'name': 'pancreas',
        'supercategory': 'pancreas',
    },
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png','*.tif']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    # file_name_prefix = basename_no_extension + '.*'
    file_name_prefix = basename_no_extension
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    data_list = os.listdir(DATA_DIR)
    # data_list = [l.strip('\n') for l in open(os.path.join(DATA_DIR,'val.txt')).readlines()]
    for i in range(len(data_list)):
        image = Image.open(os.path.join(DATA_DIR,data_list[i]))
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(data_list[i]), image.size)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        for (root, _, files) in os.walk(ANNOTATION_TUMOR_DIR):
            tumor_anno_files = filter_for_annotations(root, files, data_list[i])

            # go through each associated annotation
            for tumor_anno_filename in tumor_anno_files:
                # if tumor_anno_filename.split('/')[-1] == 'PANCREAS27-001.png':
                #     print('error')
                class_id = [x['id'] for x in CATEGORIES]
                t_category_info = {'id': class_id[0], 'is_crowd': 0}
                t_binary_mask = np.asarray(Image.open(tumor_anno_filename)
                                         .convert('1')).astype(np.uint8)
                t_anno_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, t_category_info, t_binary_mask,
                    image.size, tolerance=2)
                if t_anno_info is not None:
                    coco_output["annotations"].append(t_anno_info)
                    segmentation_id = segmentation_id + 1
                else:
                    print(tumor_anno_filename.split('/')[-1])

        image_id = image_id + 1

    with open('{}/rectal_seg_test.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()