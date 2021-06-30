from sklearn.model_selection import train_test_split
import os, csv

imgs_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/images'
masks_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/masks'

# total_files = [file for file in os.listdir(imgs_path)]
# train_files, val_files = train_test_split(total_files, test_size=0.2, random_state=42)
# #train
train_txt_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/train.txt'
# with open(train_txt_path, 'w', newline='') as file:
#     for name in train_files:
#         img_path = os.path.join(imgs_path, name)
#         # mask_path = os.path.join(masks_path, name)
#         file.write(img_path + '\n')
#     file.close()
#
test_txt_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/test.txt'
# with open(test_txt_path, 'w', newline='') as file:
#     for name in val_files:
#         img_path = os.path.join(imgs_path, name)
#         # mask_path = os.path.join(masks_path, name)
#         file.write(img_path + '\n')
#
#     file.close()

import shutil
train_des_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/MSBC-Net/datasets/rectalTumor/rectal_tumor_train'
test_des_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/MSBC-Net/datasets/rectalTumor/rectal_tumor_val'
with open(test_txt_path, 'r') as file:
    while True:
        lines = file.readline()
        if not lines:
            break

        shutil.copy(lines[:-1], os.path.join(test_des_path, lines[:-1].split('/')[-1]))


