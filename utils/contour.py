#!/usr/bin/env Python
# coding=utf-8

import cv2
import os
from os.path import join
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def find_contour(in_path,out_path):
    file_names = os.listdir(in_path)
    # mask_names=[]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    colors=[(0,0,255),(0,255,0),(255,0,0)]
    for file_name in tqdm(file_names):
        if file_name.endswith('TRAIN004_057.png'):
            # mask = cv2.imread(join(in_path, file_name), cv2.IMREAD_GRAYSCALE)
            mask=np.array(Image.open(join(in_path,file_name)))
            mask[mask == 85] = 1
            mask[mask == 170] = 2
            mask[mask == 255] = 3
            result = np.zeros(mask.shape + (3,), dtype=np.uint8)
            mask = mask.astype(np.int64)
            mask = F.one_hot(torch.from_numpy(mask), 4)
            for i in range(1, 4):
                ms = mask[:, :, i].numpy()
                ms[ms==1]=255
                contours, hierarchy = cv2.findContours(ms.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result,contours,-1,colors[i-1],1)

            # print(result)
            result = result[:, :, ::-1]
            # index_x, index_y = np.where((result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0))
            # result[index_x, index_y, :] = np.array([[255, 255, 255]])
            result = Image.fromarray(result)
            result=result.resize((512,512),Image.NEAREST)
            result.save(join(out_path, file_name))
            # cv2.imwrite(join(out_path, file_name), result)


if __name__ == "__main__":
    in_path = '../data/retouch/croped'
    out_path = '../data/retouch/distance_contour_npy/contour'


