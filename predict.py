import torch.nn as nn
import torch
from torch import optim
from torch.nn import init
import torch.nn.functional as F
import os
import  skimage.io as io
import scipy.io as io2
from skimage import transform
import cv2
from os.path import join
from PIL import Image
from networks.UncertaintyNet import UncertaintyNet
from utils.metrics import get_dc
from utils.losses import CombinedLoss
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.get_color import gradient_rgb_color
from utils.data_loader import get_loader

model_save_path='./results/models/UncertaintyNet.pth'
batch_size=4
test_data_path='./data/test'
test_save_path='./results/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
uncertainty_colors=gradient_rgb_color(["#02007D", "#01CDFF", "FEEA02", "#840000"], color_sum=100)

def test():
    network=UncertaintyNet(in_ch=1, numclasses=4).to(device)
    if os.path.exists(model_save_path):
        network.load_state_dict(torch.load(model_save_path))
        print('loading pretrained model from %s' % model_save_path)
    else:
        raise Exception("The model pretrained file does not exist")

    with torch.no_grad():
        network.eval()
        test_files = os.listdir(test_data_path)
        test_images = [file[:-len('.jpg')] for file in test_files if file.endswith('.jpg')]
        test_loader = get_loader(test_data_path, test_images, 'test', batch_size, num_workers=8)
        for i, data in enumerate(test_loader):
            image, label, distance, contour, file_name = data
            image, label = image.to(device), label.to(device)
            pred, distance_map, contour_map = network(image)

            pred_s = F.softmax(pred, dim=1)
            pred_s = torch.argmax(pred_s, dim=1)

            T = 8
            image_batch_r = image.repeat(2, 1, 1, 1)
            stride = image_batch_r.shape[0] // 2
            pred_uncertainty = torch.zeros([stride * T, 4, image.shape[2], image.shape[3]]).to(device)
            for ui in range(T // 2):
                ema_inputs = image_batch_r + torch.clamp(torch.randn_like(image_batch_r) * 0.1, -0.2, 0.2)
                ema_output, distance_map, contour_map = network(ema_inputs)
                pred_uncertainty[2 * stride * ui:2 * stride * (ui + 1)] = ema_output
            pred_uncertainty = F.softmax(pred_uncertainty, dim=1)
            pred_uncertainty = pred_uncertainty.reshape(T, stride, 4, image.shape[2], image.shape[3])
            pred_uncertainty = torch.mean(pred_uncertainty, dim=0)  # (batch, 4, 512,512)
            uncertainty = -1.0 * torch.sum(pred_uncertainty * torch.log(pred_uncertainty + 1e-6), dim=1,
                                           keepdim=True)  # (batch, 1, 256,256)

            for k in range(pred.size(0)):
                img = image[k, :, :, :].squeeze(0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                ms = label[k, :, :, :].cpu().numpy().astype(np.uint8)
                ms = np.transpose(ms[1:, :, :], (1, 2, 0))
                ms[ms == 1] = 255
                pred_ss = np.zeros((3,) + (pred_s.shape[1], pred_s.shape[2]), dtype=np.uint8)
                for j in range(3):
                    pred_ss[j, :, :][pred_s.cpu().numpy()[k] == (j + 1)] = 255
                pred_ss = np.transpose(pred_ss, (1, 2, 0))

                img = Image.fromarray(img, mode='L')
                ms = Image.fromarray(ms)
                pred_ss = Image.fromarray(pred_ss)
                img.save(join(test_save_path, file_name[k] + "_test.jpg"))
                ms.save(join(test_save_path, file_name[k] + "_test_mask.png"))
                pred_ss.save(join(test_save_path, file_name[k] + "_test_pred.png"))

                # labeled uncertainty map
                un_image = uncertainty[k, 0:1, :, :].squeeze(0).cpu().detach().numpy()
                un_image = (un_image - un_image.min()) / (un_image.max() - un_image.min())
                un_img = np.zeros((3,) + un_image.shape, dtype=np.uint8)
                pre_color_index = 0
                for color_index in range(1, len(uncertainty_colors) + 1):
                    if (color_index < len(uncertainty_colors)):
                        index_x, index_y = np.where(
                            (un_image >= (pre_color_index / 100)) & (un_image < (color_index / 100)))
                    else:
                        index_x, index_y = np.where(
                            (un_image >= (pre_color_index / 100)) & (un_image <= 1))

                    pre_color_index = color_index
                    un_img[:, index_x, index_y] = np.array([[uncertainty_colors[color_index - 1][0]],
                                                            [uncertainty_colors[color_index - 1][1]],
                                                            [uncertainty_colors[color_index - 1][2]]])
                un_img = np.transpose(un_img, (1, 2, 0))
                un_img = Image.fromarray(un_img)
                un_img.save(join(test_save_path, file_name[k] + "_test_uncertainty.png"))

if __name__ == "__main__":
    test()




