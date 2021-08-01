from utils import mhd
from utils.slice_op import hist_match
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
import platform
from os.path import join
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle
from skimage.filters.rank import median
import matplotlib.pyplot as plt
from nutsml import *
from nutsflow import *

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/'
else:
    DATA_ROOT = 'E:\\PycharmProject\\mutilclass_fluid_segmentation\\data\\retouch'

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3

CHULL = False


def preprocess_oct_images():
    # Prepare reference image for histogram matching (randomly selected from Spectralis dataset)
    filepath_r = 'E:\\PycharmProject\\mutilclass_fluid_segmentation\\data\\retouch\\Spectralis\\TRAIN030\\oct.mhd'
    oct_r, _, _ = mhd.load_oct_image(filepath_r)

    image_names = list()
    for subdir, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith("reference.mhd"):
                image_name = filepath.split('\\')[-2]
                vendor = filepath.split('\\')[-3]
                print('======== ' + vendor + ', ' + image_name + ', reference.mhd ========')
                img, _, _ = mhd.load_oct_seg(filepath)
                num_slices = img.shape[0]
                for slice_num in range(0, num_slices):
                    im_slice = img[slice_num, :, :]
                    image_names.append([image_name, vendor, subdir, slice_num, int(np.any(im_slice == IRF_CODE)),
                                        int(np.any(im_slice == SRF_CODE)), int(np.any(im_slice == PED_CODE))])
                    im_slice = im_slice.astype(np.uint8)
                    im_slice[im_slice == 1] = 85
                    im_slice[im_slice == 2] = 170
                    im_slice[im_slice == 3] = 255
                    im_slice = Image.fromarray(im_slice, mode='L')
                    im_slice.save(join(DATA_ROOT,'pre_processed','oct_masks',vendor + '_' + image_name + '_' + str(
                        slice_num).zfill(3) + '.png'))
                    # save_name = DATA_ROOT + 'pre_processed/oct_masks/' + vendor + '_' + image_name + '_' + str(
                    #     slice_num).zfill(
                    #     3) + '.tiff'
                    # im_slice.save(save_name)


            elif filepath.endswith("oct.mhd"):
                image_name = filepath.split('\\')[-2]
                vendor = filepath.split('\\')[-3]
                print('======== ' + vendor + ', ' + image_name + ', oct.mhd ========')
                img, _, _ = mhd.load_oct_image(filepath)
                if 'Cirrus' in vendor:
                    img = hist_match(img, oct_r)
                elif 'Topcon' in vendor:
                    img = hist_match(img, oct_r)
                num_slices = img.shape[0]
                for slice_num in range(0, num_slices):
                    if 'Cirrus' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
                        im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                            np.uint8)
                    if 'Topcon' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
                        im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                            np.uint8)
                    else:
                        im_slice = img[slice_num, :, :].astype(np.uint8)
                    im_slice = Image.fromarray(im_slice, mode='L')
                    # TODO : check if this second filtering is useful
                    if 'Cirrus' in vendor:
                        im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))
                    elif 'Topcon' in vendor:
                        im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))

                    im_slice.save(join(DATA_ROOT, 'pre_processed', 'oct_imgs', vendor + '_' + image_name + '_' + str(
                        slice_num).zfill(3) + '.jpg'))

                    # save_name = DATA_ROOT + 'pre_processed/oct_imgs/' + vendor + '_' + image_name + '_' + str(
                    #     slice_num).zfill(3) + '.tiff'
                    # im_slice.save(save_name)


    col_names = ['image_name', 'vendor', 'root', 'slice', 'is_IRF', 'is_SRF', 'is_PED']
    df = pd.DataFrame(image_names, columns=col_names)
    df.to_csv(DATA_ROOT + '\\pre_processed\\slice_gt.csv', index=False)


def create_test_train_set():
    print('generating new test train SPLIT')
    # reading training data
    train_file = DATA_ROOT + '/pre_processed/slice_gt.csv'
    data = ReadPandas(train_file, dropnan=True)
    data = data >> Shuffle(7000) >> Collect()

    case_names = data >> GetCols(0, 1) >> Collect(set)
    # case_names >> Print() >> Consume()

    # is_topcon = lambda s: s[1] == 'Topcon'
    # is_cirrus = lambda s: s[1] == 'Cirrus'
    # is_spectralis = lambda s: s[1] == 'Spectralis'

    train_cases, test_cases = case_names >> Shuffle(70) >> SplitRandom(ratio=0.75)

    print(train_cases >> GetCols(1) >> CountValues())
    print(test_cases >> GetCols(1) >> CountValues())

    train_cases = train_cases >> GetCols(0) >> Collect()
    test_cases = test_cases >> GetCols(0) >> Collect()

    is_in_train = lambda sample: (sample[0],) in list(train_cases)

    writer = WriteCSV('./outputs/train_data.csv')
    data >> Filter(is_in_train) >> writer
    writer = WriteCSV('./outputs/test_data.csv')
    data >> FilterFalse(is_in_train) >> writer

    case_names = data >> Filter(is_in_train) >> GetCols(0, 1) >> Collect(set)
    print(case_names >> GetCols(1) >> CountValues())
    case_names = data >> FilterFalse(is_in_train) >> GetCols(0, 1) >> Collect(set)
    print(case_names >> GetCols(1) >> CountValues())


def create_roi_masks(tresh=1e-2):
    import skimage.io as sio
    import skimage
    from skimage.morphology import disk, rectangle, closing, opening, binary_closing, convex_hull_image
    from skimage.filters.rank import entropy

    MASK_PATH = '/home/truwan/DATA/retouch/pre_processed/oct_masks/'

    image_names = list()
    for subdir, dirs, files in os.walk(DATA_ROOT + 'pre_processed/oct_imgs/'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tiff"):
                image_name = filepath.split('/')[-1]
                image_names.append([filepath, image_name])

    for filepath, image_name in image_names:
        # print filepath, image_name
        img = sio.imread(filepath)
        im_mask = sio.imread(MASK_PATH + image_name)
        im_mask = im_mask.astype(np.int8)
        im_slice = skimage.img_as_float(img.astype(np.float32) / 128. - 1.)

        im_slice_ = entropy(im_slice, disk(15))
        im_slice_ = im_slice_ / (np.max(im_slice_) + 1e-16)
        im_slice_ = np.asarray(im_slice_ > tresh, dtype=np.int8)
        im_slice_ = np.bitwise_or(im_slice_, im_mask)
        selem = disk(55)
        im_slice_ = binary_closing(im_slice_, selem=selem)

        h, w = im_slice_.shape
        rnge = list()
        for x in range(0, w):
            col = im_slice_[:, x]
            col = np.nonzero(col)[0]
            # print col, col.shape
            if len(col) > 0:
                y_min = np.min(col)
                y_max = np.max(col)
                rnge.append(int((float(y_max) - y_min) / h * 100.))
                im_slice_[y_min:y_max, x] = 1
        if len(rnge) > 0:
            print(image_name, np.max(rnge))
        else:
            print(image_name, "**************")

        if CHULL:
            im_slice_ = convex_hull_image(im_slice_)

        # plt.imshow(im_slice, cmap='gray')
        # plt.imshow(im_slice_, cmap='jet', alpha=0.5)
        # plt.pause(.1)

        im_slice_ = Image.fromarray(im_slice_, mode='L')
        save_name = DATA_ROOT + 'pre_processed/roi_masks_chull/' + image_name
        im_slice_.save(save_name)

def get_kmeans_segment():
    out_path='E:\\PycharmProject\\mutilclass_fluid_segmentation\\data\\retouch\\proposed'
    in_path='E:\\PycharmProject\\mutilclass_fluid_segmentation\\data\\retouch'
    crop_out_path='E:\\PycharmProject\\mutilclass_fluid_segmentation\\data\\retouch\\croped'
    img_files=os.listdir(join(in_path,'pre_processed\\oct_imgs'))
    for img_file in img_files:
        mask=cv2.imread(join(in_path,'pre_processed\\oct_masks',img_file[:-len('jpg')]+'png'), cv2.IMREAD_GRAYSCALE)
        mask[mask!=0]=1
        if mask.sum()>=50:
            print('###### '+img_file+' #####')
            img = cv2.imread(join(in_path,'pre_processed\\oct_imgs',img_file), cv2.IMREAD_GRAYSCALE)
            # change img(2D) to 1D
            img1 = img.reshape((img.shape[0] * img.shape[1], 1))
            img1 = np.float32(img1)

            # define criteria = (type,max_iter,epsilon)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # set flags: hou to choose the initial center
            # ---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
            flags = cv2.KMEANS_RANDOM_CENTERS
            # apply kmenas
            compactness, labels, centers = cv2.kmeans(img1, 2, None, criteria, 10, flags)

            img2 = labels.reshape((img.shape[0], img.shape[1]))
            if img2[0][0]==1:
                img_t=np.zeros((img2.shape[0],img2.shape[1]),np.uint8)
                img_t[img2==0]=1
                img2=img_t
            # median filter
            img2=cv2.medianBlur(img2.astype(np.float32),3).astype(np.uint8)

            step=60
            for i in range(img2.shape[0]):
                if img2[i,:].sum()>=10:
                    if i-step<0:
                        top_line=0
                    else:
                        top_line=i-step
                    break
            for j in range(img2.shape[0]-1,-1,-1):
                if img2[j,:].sum()>=10:
                    if j+step>img2.shape[0]-1:
                        down_line=img2.shape[0]-1
                    else:
                        down_line=j+step
                    break
            img = cv2.imread(join(in_path, 'pre_processed\\oct_imgs', img_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(join(in_path, 'pre_processed\\oct_masks', img_file[:-len('jpg')] + 'png'),cv2.IMREAD_GRAYSCALE)
            img=img[top_line:down_line,:]
            mask=mask[top_line:down_line,:]
            mask[mask == 1] = 85
            mask[mask == 2] = 170
            mask[mask == 3] = 255

            img=cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask=cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(join(crop_out_path,img_file),img)
            cv2.imwrite(join(crop_out_path,img_file[:-len('jpg')]+'png'),mask)
            img2[img2==1]=255
            cv2.imwrite(join(out_path,img_file),img2)

if __name__ == "__main__":
    # crop_out_path = 'D:\\pycharmproject\\mutilclass_fluid_segmentation\\data\\retouch\\croped'
    # img_files = os.listdir(crop_out_path)
    # for img_file in img_files:
    #     if img_file.endswith('.png'):
    #         img=cv2.imread(join(crop_out_path,img_file),cv2.IMREAD_GRAYSCALE)
    #         print(np.unique(img))

    # preprocess_oct_images()
    get_kmeans_segment()
    # create_roi_masks()
    # create_test_train_set()
