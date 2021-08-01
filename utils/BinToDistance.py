import os
from tqdm import tqdm
import cv2
from glob import glob
from os.path import dirname, join, basename
from shutil import copy
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_cdt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F


def DistanceWithoutNormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

def DistanceBinNormalise(bin_image):
    bin_image = label(bin_image)
    result = np.zeros_like(bin_image, dtype="float")
    for k in range(1, bin_image.max() + 1):
        tmp = np.zeros_like(result, dtype="float")
        tmp[bin_image == k] = 1
        dist = distance_transform_cdt(tmp)
        MAX = dist.max()
        dist = dist.astype(float) / MAX
        result[bin_image == k] = dist[bin_image == k]
    result = result.astype('float32')*255
    result=result.astype(np.uint8)
    return result

def DistanceBinNormaliseNpy(bin_image):
    bin_image = label(bin_image)
    result = np.zeros_like(bin_image, dtype=np.float32)
    for k in range(1, bin_image.max() + 1):
        tmp = np.zeros_like(result, dtype=np.float32)
        tmp[bin_image == k] = 1
        dist = distance_transform_cdt(tmp)
        MAX = dist.max()
        dist = dist.astype(float) / MAX
        result[bin_image == k] = dist[bin_image == k]
    return result

def get_distancemap(in_path,out_path):
    file_names=os.listdir(in_path)
    # mask_names=[]
    for file_name in tqdm(file_names):
        if file_name.endswith('.png'):
            # mask = cv2.imread(join(in_path, file_name), cv2.IMREAD_GRAYSCALE)
            mask=np.array(Image.open(join(in_path,file_name)))
            mask[mask == 85] = 1
            mask[mask == 170] = 2
            mask[mask == 255] = 3
            result=np.zeros(mask.shape+(3,),dtype=np.uint8)

            mask=mask.astype(np.int64)
            mask=F.one_hot(torch.from_numpy(mask),4)
            for i in range(1,4):
                ms=mask[:,:,i].numpy()
                ms=DistanceBinNormalise(ms)
                result[:,:,i-1]=ms
            # print(result)
            # result=result[:, :, ::-1]
            # result[result==np.array([0,0,0])]=np.array([255,255,255])
            index_x, index_y = np.where((result[:,:,0]==0) & (result[:,:,1]==0) & (result[:,:,2]==0))
            result[index_x,index_y,:]=np.array([[255,255,255]])
            result=Image.fromarray(result)
            result.save(join(out_path,file_name))
            # cv2.imwrite(join(out_path, file_name), result)



if __name__ == "__main__":
    in_path = '../data/retouch/croped'
    out_path = '../data/retouch/distance_contour_npy/distance'




