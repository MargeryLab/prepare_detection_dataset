import os, cv2, shutil

def jpg_to_png():
    imgs_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/MSBC-Net/datasets/rectalTumor/rectal_tumor_val_jpg'
    imgs_png_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/MSBC-Net/datasets/rectalTumor/rectal_tumor_val'

    masks_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/masks_raw_jpg'
    masks_png_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/masks_raw'

    for file in os.listdir(masks_path):
        arr = cv2.imread(os.path.join(masks_path, file), flags=0)
        arr[arr<200] = 0
        arr[arr>200] = 255
        for i in range(len(arr)-1):
            for j in range(len(arr[0])-1):
                if arr[i][j] == 255 and arr[i][j+1] != 255 and arr[i+1][j] != 255:
                    arr[i][j] = 0
        cv2.imwrite(os.path.join(masks_png_path, file[:-4]+'.png'), arr)

def find_imgs():
    imgs_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/medical_image_segmentation/Data/data_png_png/imgs'
    des_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/tools/prepare_detection_dataset/imgs_rectal'
    by_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/tools/prepare_detection_dataset/test_tumor_whole'
    for name in os.listdir(by_path):
        shutil.copy(os.path.join(imgs_path, name), os.path.join(des_path, name))


if __name__ == '__main__':
    jpg_to_png()
    # find_imgs()