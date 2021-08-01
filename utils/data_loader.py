from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
from torch.utils import data
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch.nn.functional as F
from os.path import join
import itertools
from torch.utils.data.sampler import Sampler
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_cdt
import cv2

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

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
    result=result.astype(np.float32)
    return result

class ImageFolder(torch.utils.data.Dataset):

    def __init__(self, file_dir, files,mode='train'):
        self.files_dir=file_dir
        self.files = files
        self.mode=mode

    def __len__(self):

        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(join(self.files_dir, self.files[i]+'.jpg'))
        label = Image.open(join(self.files_dir, self.files[i]+'.png'))

        image_name=self.files[i]
        if self.mode=='train':
            if random.random()<0.5:
                if random.random()<0.5:
                    image=transforms.RandomVerticalFlip(p=1)(image)
                    label=transforms.RandomVerticalFlip(p=1)(label)

                if random.random()<0.5:
                    image=transforms.RandomHorizontalFlip(p=1)(image)
                    label=transforms.RandomHorizontalFlip(p=1)(label)
            image=transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(image)

        image=transforms.ToTensor()(image)

        label=torch.from_numpy(np.array(label)).long()
        label[label==85]=1
        label[label==170]=2
        label[label==255]=3
        label=F.one_hot(label,num_classes=4).permute((2,0,1)).float()

        #distance_map
        distance_map=torch.zeros((3,label.shape[1],label.shape[2]),dtype=torch.float32)
        for i in range(1, 4):
            ms = label[i, :, :].numpy()
            ms = DistanceBinNormalise(ms)
            distance_map[i-1, :, :] = torch.from_numpy(ms)
        # distance_map=torch.from_numpy(np.transpose(distance_map,(2,0,1)))
        distance_map=distance_map.float()
        #counter_map
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        contour_map=np.zeros((label.shape[1],label.shape[2],3),dtype=np.uint8)
        for i in range(1, 4):
            ms = label[i, :, :].numpy()
            # ms=ms.permute((1,2,0))
            ms[ms == 1] = 255
            _, contours, hierarchy = cv2.findContours(ms.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_map, contours, -1, colors[i - 1], 1)
        contour_map=torch.from_numpy(contour_map).permute((2,0,1)).float()
        # contour_map=torch.from_numpy(np.transpose(contour_map,(2,0,1))).float()

        return image, label,distance_map,contour_map,image_name

seed=130000
def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

def get_semi_loader(root,distance_path,contour_path, imgfiles,mode='trian', batch_size=4, labeled_bs=2,num_workers=2,labeled_ratio=0.5):
    # files = os.listdir(root)
    # imgfiles = [file[:-len('.jpg')] for file in files if file.endswith('.jpg')]
    start_idx=int(len(imgfiles)*labeled_ratio)
    labeled_idxs = list(range(start_idx))
    unlabeled_idxs = list(range(start_idx, len(imgfiles)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    dataset = ImageFolder(root, distance_path,contour_path,imgfiles,mode)
    trainloader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True,
                                  worker_init_fn=worker_init_fn)
    return trainloader

def get_loader(root,files, mode='trian', batch_size=4, num_workers=2):
    dataset = ImageFolder(root,files,mode)
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return loader

if __name__ == '__main__':
    root = '../data/retouch/croped'
    contour_path = "../data/retouch/distance_contour_npy/contour"
    distance_path='../data/retouch/distance_contour_npy/distance'
    files = os.listdir(root)
    files = [file[:-len('.jpg')] for file in files if file.endswith('.jpg')]
    print(files)
    # dataset = ImageFolder(root,'train')
    # print(dataset.__len__())
    # train_loader = data.DataLoader(dataset,batch_size=2,shuffle=True)
    train_loader=get_semi_loader(root,distance_path,contour_path,files)
    for i,data in enumerate(train_loader):
        image,label,distance_map,contour_map,file_name = data
        print(i)
        print(image.shape)
        print(label.shape)
        print(distance_map.dtype)
        print(contour_map.dtype)
        print(file_name)
        print('=================')

