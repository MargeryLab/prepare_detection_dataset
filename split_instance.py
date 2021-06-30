from skimage.measure import label,regionprops
import cv2
import os
import numpy as np

# gt_path = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/tools/prepare_detection_dataset/test_tumor_whole'
# gt_crop_save = '/media/margery/4ABB9B07DF30B9DB/pythonDemo/tools/prepare_detection_dataset/test_tumor_mask'
gt_path = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/masks_raw'
gt_crop_save = '/media/margery/4ABB9B07DF30B9DB/MedicalImagingDataset/Kvasir-SEG/masks'


def extract_gt_bbox(gt):
    gt_arr = cv2.imread(os.path.join(gt_path,gt),flags=0)
    # gt_arr = gt_arr.transpose(2,0,1)
    if gt_arr.max() == 0:
        return gt_arr, None
    ConnectMap = label(gt_arr, connectivity=2)
    Props = regionprops(ConnectMap)
    Area = []
    Bbox = []
    for j in range(len(Props)):
        Area.append(Props[j]['area'])
        Bbox.append(Props[j]['bbox'])
    Area = np.array(Area)
    Bbox = np.array(Bbox)
    argsort = np.argsort(Area)
    Area = Area[argsort]
    Bbox = Bbox[argsort]
    Area=Area[::-1]
    Bbox = Bbox[::-1,:]
    Area = Area[Area>200]
    l = int(Area.shape[0])
    Bbox = Bbox[:l,:]
    # MaximumBbox = Bbox[0]
    # return gt_arr, MaximumBbox
    return gt_arr, Bbox


def crop_imgs_gt():
    x_ls = []
    y_ls = []
    for gt,img in zip(os.listdir(gt_path),os.listdir(imgs_path)):
        # 边界外接框(min_row, min_col, max_row, max_col)
        gt_arr, MaxiumBbox = extract_gt_bbox(gt)
        if MaxiumBbox is None:
            continue
        img_arr = cv2.imread(os.path.join(imgs_path,img),flags=0)
        img_crop_arr = img_arr[MaxiumBbox[0]:MaxiumBbox[2],MaxiumBbox[1]:MaxiumBbox[3]]
        x_ls.append(img_crop_arr.shape[0])
        y_ls.append(img_crop_arr.shape[1])
        gt_crop_arr = gt_arr[MaxiumBbox[0]:MaxiumBbox[2],MaxiumBbox[1]:MaxiumBbox[3]]
        cv2.imwrite(os.path.join(gt_crop_save, gt),gt_crop_arr)
        cv2.imwrite(os.path.join(img_crop_save, img),img_crop_arr)
    print(np.mean(x_ls))
    print(np.mean(y_ls))

def split_bbox5(gt_arr, Bbox):
    gt_arr1 = np.copy(gt_arr)
    gt_arr2 = np.copy(gt_arr)
    gt_arr3 = np.copy(gt_arr)
    gt_arr4 = np.copy(gt_arr)
    gt_arr5 = np.copy(gt_arr)
    gt_arr1[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr1[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr1[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr1[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0

    gt_arr2[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr2[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr2[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr2[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0

    gt_arr3[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr3[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr3[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr3[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0

    gt_arr4[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr4[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr4[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr4[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0

    gt_arr5[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr5[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr5[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr5[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0

    return gt_arr1,gt_arr2,gt_arr3,gt_arr4,gt_arr5

def split_bbox6(gt_arr,Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5 = split_bbox5(gt_arr, Bbox)
    gt_arr1[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr2[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr3[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr4[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr5[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0

    gt_arr6 = np.copy(gt_arr)
    gt_arr6[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr6[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr6[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr6[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr6[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0

    return gt_arr1,gt_arr2,gt_arr3,gt_arr4,gt_arr5,gt_arr6


def split_bbox7(gt_arr, Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6 = split_bbox6(gt_arr, Bbox)
    gt_arr1[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr2[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr3[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr4[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr5[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr6[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0


    gt_arr7 = np.copy(gt_arr)
    gt_arr7[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr7[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr7[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr7[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr7[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0
    gt_arr7[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0

    return gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7


def split_bbox8(gt_arr, Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7 = split_bbox7(gt_arr, Bbox)
    gt_arr1[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr2[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr3[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr4[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr5[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr6[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr7[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0

    gt_arr8 = np.copy(gt_arr)
    gt_arr8[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr8[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr8[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr8[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr8[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0
    gt_arr8[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr8[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0

    return gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8

def split_bbox9(gt_arr, Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8 = split_bbox8(gt_arr, Bbox)
    gt_arr1[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr2[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr3[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr4[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr5[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr6[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr7[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr8[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0

    gt_arr9 = np.copy(gt_arr)
    gt_arr9[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr9[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr9[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr9[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr9[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0
    gt_arr9[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr9[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr9[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0

    return gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9

def split_bbox10(gt_arr, Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9 = split_bbox9(gt_arr, Bbox)
    gt_arr1[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr2[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr3[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr4[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr5[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr6[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr7[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr8[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0
    gt_arr9[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0

    gt_arr10 = np.copy(gt_arr)
    gt_arr10[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr10[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr10[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr10[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr10[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0
    gt_arr10[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr10[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr10[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr10[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0

    return gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9, gt_arr10

def split_bbox11(gt_arr, Bbox):
    gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9, gt_arr10 = split_bbox10(gt_arr, Bbox)
    gt_arr1[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr2[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr3[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr4[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr5[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr6[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr7[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr8[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr9[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0
    gt_arr10[Bbox[10][0]:Bbox[10][2], Bbox[10][1]:Bbox[10][3]] = 0

    gt_arr11 = np.copy(gt_arr)
    gt_arr11[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
    gt_arr11[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
    gt_arr11[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
    gt_arr11[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0
    gt_arr11[Bbox[4][0]:Bbox[4][2], Bbox[4][1]:Bbox[4][3]] = 0
    gt_arr11[Bbox[5][0]:Bbox[5][2], Bbox[5][1]:Bbox[5][3]] = 0
    gt_arr11[Bbox[6][0]:Bbox[6][2], Bbox[6][1]:Bbox[6][3]] = 0
    gt_arr11[Bbox[7][0]:Bbox[7][2], Bbox[7][1]:Bbox[7][3]] = 0
    gt_arr11[Bbox[8][0]:Bbox[8][2], Bbox[8][1]:Bbox[8][3]] = 0
    gt_arr11[Bbox[9][0]:Bbox[9][2], Bbox[9][1]:Bbox[9][3]] = 0

    return gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9, gt_arr10, gt_arr11

def wall_bbox():
    for gt in os.listdir(gt_path):
        # 边界外接框(min_row, min_col, max_row, max_col)
        gt_arr, Bbox = extract_gt_bbox(gt)
        if Bbox is None or len(Bbox) == 1 or len(Bbox) == 0:
            cv2.imwrite(os.path.join(gt_crop_save, gt), gt_arr)
            continue
        elif len(Bbox) == 2:
            gt_arr1 = np.copy(gt_arr)
            gt_arr2 = np.copy(gt_arr)
            gt_arr1[Bbox[1][0]:Bbox[1][2],Bbox[1][1]:Bbox[1][3]] = 0
            gt_arr2[Bbox[0][0]:Bbox[0][2],Bbox[0][1]:Bbox[0][3]] = 0

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-a.png'),gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-b.png'),gt_arr2)
        elif len(Bbox) == 3:
            gt_arr1 = np.copy(gt_arr)
            gt_arr2 = np.copy(gt_arr)
            gt_arr3 = np.copy(gt_arr)
            gt_arr1[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
            gt_arr1[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0

            gt_arr2[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
            gt_arr2[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0

            gt_arr3[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
            gt_arr3[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-c.png'), gt_arr3)
        elif len(Bbox) == 4:
            gt_arr1 = np.copy(gt_arr)
            gt_arr2 = np.copy(gt_arr)
            gt_arr3 = np.copy(gt_arr)
            gt_arr4 = np.copy(gt_arr)
            gt_arr1[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
            gt_arr1[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
            gt_arr1[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0

            gt_arr2[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
            gt_arr2[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0
            gt_arr2[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0

            gt_arr3[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
            gt_arr3[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
            gt_arr3[Bbox[3][0]:Bbox[3][2], Bbox[3][1]:Bbox[3][3]] = 0

            gt_arr4[Bbox[0][0]:Bbox[0][2], Bbox[0][1]:Bbox[0][3]] = 0
            gt_arr4[Bbox[1][0]:Bbox[1][2], Bbox[1][1]:Bbox[1][3]] = 0
            gt_arr4[Bbox[2][0]:Bbox[2][2], Bbox[2][1]:Bbox[2][3]] = 0

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4]+'-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
        elif len(Bbox) == 5:
            gt_arr1,gt_arr2,gt_arr3,gt_arr4,gt_arr5 = split_bbox5(gt_arr, Bbox)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
        elif len(Bbox) == 6:
            gt_arr1,gt_arr2,gt_arr3,gt_arr4,gt_arr5,gt_arr6 = split_bbox6(gt_arr, Bbox)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-f.png'), gt_arr6)
        elif len(Bbox) == 7:
            gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7 = split_bbox7(gt_arr, Bbox)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-f.png'), gt_arr6)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-g.png'), gt_arr7)
        elif len(Bbox) == 8:
            gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8 = split_bbox8(gt_arr, Bbox)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-f.png'), gt_arr6)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-g.png'), gt_arr7)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-h.png'), gt_arr8)
        elif len(Bbox) == 9:
            gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9 = split_bbox9(gt_arr, Bbox)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-f.png'), gt_arr6)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-g.png'), gt_arr7)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-h.png'), gt_arr8)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-i.png'), gt_arr9)
        else:
            print(gt, len(Bbox))
            gt_arr1, gt_arr2, gt_arr3, gt_arr4, gt_arr5, gt_arr6, gt_arr7, gt_arr8, gt_arr9, gt_arr10 = split_bbox10(gt_arr, Bbox)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-a.png'), gt_arr1)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-b.png'), gt_arr2)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-c.png'), gt_arr3)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-d.png'), gt_arr4)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-e.png'), gt_arr5)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-f.png'), gt_arr6)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-g.png'), gt_arr7)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-h.png'), gt_arr8)
            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-i.png'), gt_arr9)

            cv2.imwrite(os.path.join(gt_crop_save, gt[:-4] + '-j.png'), gt_arr10)

if __name__ == '__main__':
    # crop_imgs_gt()
    wall_bbox()#注意png还是png


