import os
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy.io as scio
import numpy as np


def norm(img):
    if isinstance(img, np.ndarray):
        min = np.min(img)
        max = np.max(img)
        img = (img - min) / (max - min)
    elif isinstance(img, list):
        for i in range(len(img)):
            c = img[i]
            min = np.min(c)
            max = np.max(c)
            img[i] = (c - min) / (max - min)

    return img


class DataSetImage(Dataset):
    def __init__(self,
                 dataset: Dict
                 ):
        super(DataSetImage, self).__init__()
        self.root = dataset["root"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.lmsi_dir = dataset["lmsi_dir"]
        self.data_index = dataset["data_index"]

        self.GT = torch.from_numpy(scio.loadmat(self.root + '/' + self.gt_dir + '/' + self.data_index)['data']) \
            .to(torch.float32)
        self.height, self.width, self.bands = self.GT.shape
        self.HSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.hsi_dir + '/' + self.data_index)['data'])\
            .to(torch.float32)
        self.height_hsi, self.width_hsi = self.HSI.shape[0:2]
        self.MSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.msi_dir + '/' + self.data_index)['data'])\
            .to(torch.float32)
        self.lMSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.lmsi_dir + '/' + self.data_index)['data']) \
            .to(torch.float32)
        self.bands_msi = self.MSI.shape[2]
        self.scale_spatial = self.height // self.height_hsi
        self.bHSI = F.interpolate(self.HSI.permute(2, 0, 1).unsqueeze(dim=0),
                                  scale_factor=self.scale_spatial, mode='bilinear', align_corners=True)\
            .squeeze(dim=0).permute(1, 2, 0)

        self.gt = self.GT.reshape(self.height*self.width, self.bands)
        self.Hsi = self.HSI.reshape(self.height_hsi*self.width_hsi, self.bands)
        self.Msi = self.MSI.reshape(self.height*self.width, self.bands_msi)
        self.bHsi = self.bHSI.reshape(self.height * self.width, self.bands)

        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]
        self.scale_spatial = self.height // self.height_hsi

    def __len__(self):
        return 1

    def __getitem__(self, item):
        data = {
            "gt": self.gt,
            "hsi": self.Hsi,
            "msi": self.Msi,
            "bhsi": self.bHsi
        }

        return data


class DataSetBlock(Dataset):
    def __init__(self,
                 dataset: Dict,
                 data_index: int,
                 # blocksize: List
                 ):
        super(DataSetBlock, self).__init__()
        self.root = dataset["root"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.scale_factor = int(dataset["scale_factor"])
        self.data_index = data_index
        # edited in 11/8/22
        # self.block_height, self.block_width = blocksize

        self.GT = torch.from_numpy(scio.loadmat(self.root + '/' + self.gt_dir + '/' + str(self.data_index))['data']) \
            .to(torch.float32)
        self.height, self.width, self.bands = self.GT.shape
        self.block_height, self.block_width = self.height // self.scale_factor, self.width // self.scale_factor
        self.HSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.hsi_dir + '/' + str(self.data_index))['data'])\
            .to(torch.float32)
        self.height_hsi, self.width_hsi = self.HSI.shape[0:2]
        msi = scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data']
        if len(msi.shape) == 2:
            self.MSI = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data'][:, :, None])\
                .to(torch.float32)
        else:
            self.MSI = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data']) \
                .to(torch.float32)

        self.bands_msi = self.MSI.shape[2]
        self.scale_spatial = self.height // self.height_hsi

        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]

    def __len__(self):
        return (self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1

    def __getitem__(self, item):
        if self.block_width == self.width_hsi:
            y_top = 0
        else:
            y_top = item // (self.width_hsi - self.block_width)

        item = item + y_top * self.block_width
        x_left = item - y_top*self.width_hsi
        y_bottom = y_top + self.block_height
        x_right = x_left + self.block_width
        x_left_msi = self.scale_spatial * x_left
        y_top_msi = self.scale_spatial * y_top
        y_bottom_msi = y_bottom * self.scale_spatial
        x_right_msi = x_right * self.scale_spatial

        data = {"gt": self.GT[y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :],
                "hsi": self.HSI[y_top:y_bottom, x_left:x_right, :],
                "msi": self.MSI[y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :]}

        return data


class DataSetBlockWV2(Dataset):
    def __init__(self,
                 dataset: Dict,
                 data_index: int,
                 blocksize: List
                 ):
        super(DataSetBlockWV2, self).__init__()
        self.root = dataset["root"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.lmsi_dir = dataset["lmsi_dir"]
        self.lmsi_step = dataset["lmsi_step"]
        self.data_index = data_index
        self.block_height, self.block_width = blocksize

        self.GT = torch.from_numpy(scio.loadmat(self.root + '/' + self.gt_dir + '/' + str(self.data_index))['data']) \
            .to(torch.float32)
        self.height, self.width, self.bands = self.GT.shape
        self.HSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.hsi_dir + '/' + str(self.data_index))['data']/255)\
            .to(torch.float32)
        self.height_hsi, self.width_hsi = self.HSI.shape[0:2]
        msi = scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data']/255
        if len(msi.shape) == 2:
            self.MSI = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data'][:, :, None]/255)\
                .to(torch.float32)
        else:
            self.MSI = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(self.data_index))['data']/255) \
                .to(torch.float32)

        self.bands_msi = self.MSI.shape[2]
        self.scale_spatial = self.height // self.height_hsi

        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]

    def __len__(self):
        return (self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1

    def __getitem__(self, item):
        if self.block_width == self.width_hsi:
            y_top = 0
        else:
            y_top = item // (self.width_hsi - self.block_width)

        item = item + y_top * self.block_width
        x_left = item - y_top*self.width_hsi
        y_bottom = y_top + self.block_height
        x_right = x_left + self.block_width
        x_left_msi = self.scale_spatial * x_left
        y_top_msi = self.scale_spatial * y_top
        y_bottom_msi = y_bottom * self.scale_spatial
        x_right_msi = x_right * self.scale_spatial

        data = {"gt": self.GT[y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :],
                "hsi": self.HSI[y_top:y_bottom, x_left:x_right, :],
                "msi": self.MSI[y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :], "steps": self.lmsi_step}

        return data


class DataSetImgPatch(Dataset):
    def __init__(self,
                 dataset: Dict,
                 data_index: List,
                 imgsize: List,
                 scale_factor: int,
                 blocksize: List
                 ):
        super(DataSetImgPatch, self).__init__()
        self.root = dataset["root"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.data_index = data_index
        self.block_height, self.block_width = blocksize
        self.height_hsi, self.width_hsi, self.bands, self.bands_msi = imgsize
        self.scale_factor = scale_factor
        self.height, self.width = self.height_hsi * self.scale_factor, self.width_hsi * self.scale_factor
        self.scale_spatial = self.height // self.height_hsi
        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]

        self.GT = torch.zeros([len(data_index), self.height, self.width, self.bands])
        self.HSI = torch.zeros([len(data_index), self.height_hsi, self.width_hsi, self.bands])
        self.MSI = torch.zeros([len(data_index), self.height, self.width, self.bands_msi])
        for i in range(len(data_index)):
            self.GT[i] = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.gt_dir + '/' + str(data_index[i]))['data']).to(torch.float32)
            self.HSI[i] = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.hsi_dir + '/' + str(data_index[i]))['data']).to(torch.float32)
            msi = scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data']
            if len(msi.shape) == 2:
                self.MSI[i] = torch.from_numpy(
                    scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data'][:, :, None])\
                    .to(torch.float32)
            else:
                self.MSI[i] = torch.from_numpy(
                    scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data']) \
                    .to(torch.float32)

    def __len__(self):
        return ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1) * len(self.data_index)

    def __getitem__(self, item):
        img_index = item // ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1)
        patch_item = item - img_index * \
                     ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1)

        if self.block_width == self.width_hsi:
            y_top = 0
        else:
            y_top = patch_item // (self.width_hsi - self.block_width)

        patch_item = patch_item + y_top * self.block_width
        x_left = patch_item - y_top * self.width_hsi
        y_bottom = y_top + self.block_height
        x_right = x_left + self.block_width
        x_left_msi = self.scale_spatial * x_left
        y_top_msi = self.scale_spatial * y_top
        y_bottom_msi = y_bottom * self.scale_spatial
        x_right_msi = x_right * self.scale_spatial

        data = {"gt": self.GT[img_index, y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :],
                "hsi": self.HSI[img_index, y_top:y_bottom, x_left:x_right, :],
                "msi": self.MSI[img_index, y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :]}

        return data


class DataSetImgPatchWV2(Dataset):
    def __init__(self,
                 dataset: Dict,
                 data_index: List,
                 imgsize: List,
                 scale_factor: int,
                 blocksize: List
                 ):
        super(DataSetImgPatchWV2, self).__init__()
        self.root = dataset["root"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.lmsi_dir = dataset["lmsi_dir"]
        self.lmsi_step = dataset["lmsi_step"]
        self.data_index = data_index
        self.block_height, self.block_width = blocksize
        self.height_hsi, self.width_hsi, self.bands, self.bands_msi = imgsize
        self.scale_factor = scale_factor
        self.height, self.width = self.height_hsi * self.scale_factor, self.width_hsi * self.scale_factor
        self.scale_spatial = self.height // self.height_hsi
        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]

        self.GT = torch.zeros([len(data_index), self.height, self.width, self.bands])
        self.HSI = torch.zeros([len(data_index), self.height_hsi, self.width_hsi, self.bands])
        self.MSI = torch.zeros([len(data_index), self.height, self.width, self.bands_msi])
        for i in range(len(data_index)):
            self.GT[i] = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.gt_dir + '/' + str(data_index[i]))['data']).to(torch.float32)
            self.HSI[i] = torch.from_numpy(
                scio.loadmat(self.root + '/' + self.hsi_dir + '/' + str(data_index[i]))['data']/255).to(torch.float32)
            msi = scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data']
            if len(msi.shape) == 2:
                self.MSI[i] = torch.from_numpy(
                    scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data'][:, :, None]/255)\
                    .to(torch.float32)
            else:
                self.MSI[i] = torch.from_numpy(
                    scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(data_index[i]))['data']/255) \
                    .to(torch.float32)

    def __len__(self):
        return ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1) * len(self.data_index)

    def __getitem__(self, item):
        img_index = item // ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1)
        patch_item = item - img_index * \
                     ((self.height_hsi - self.block_height) * (self.width_hsi - self.block_width) + 1)

        if self.block_width == self.width_hsi:
            y_top = 0
        else:
            y_top = patch_item // (self.width_hsi - self.block_width)

        patch_item = patch_item + y_top * self.block_width
        x_left = patch_item - y_top * self.width_hsi
        y_bottom = y_top + self.block_height
        x_right = x_left + self.block_width
        x_left_msi = self.scale_spatial * x_left
        y_top_msi = self.scale_spatial * y_top
        y_bottom_msi = y_bottom * self.scale_spatial
        x_right_msi = x_right * self.scale_spatial

        data = {"gt": self.GT[img_index, y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :],
                "hsi": self.HSI[img_index, y_top:y_bottom, x_left:x_right, :],
                "msi": self.MSI[img_index, y_top_msi:y_bottom_msi, x_left_msi:x_right_msi, :], "steps": self.lmsi_step}

        return data



class DataSetPatch(Dataset):
    def __init__(self,
                 dataset: Dict,
                 data_index: List,
                 imgsize: List,
                 scale_factor: int,
                 blocksize: List
                 ):
        super(DataSetPatch, self).__init__()
        self.root = dataset["root_patch"]
        self.hsi_dir = dataset["hsi_dir"]
        self.msi_dir = dataset["msi_dir"]
        self.gt_dir = dataset["gt_dir"]
        self.lmsi_dir = dataset["lmsi_dir"]
        self.lmsi_step = dataset["lmsi_step"]
        self.data_index = data_index
        self.block_height, self.block_width = blocksize
        self.height_hsi, self.width_hsi, self.bands, self.bands_msi = imgsize
        self.scale_factor = scale_factor
        self.height, self.width = self.height_hsi * self.scale_factor, self.width_hsi * self.scale_factor

        self.size = [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi]

    def __len__(self):
        return (self.height_hsi // self.block_height * self.width_hsi // self.block_width) * len(self.data_index)

    def __getitem__(self, item):

        self.GT = torch.from_numpy(scio.loadmat(self.root + '/' + self.gt_dir + '/' + str(item))['data']) \
            .to(torch.float32)
        self.HSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.hsi_dir + '/' + str(item))['data']) \
            .to(torch.float32)
        self.MSI = torch.from_numpy(scio.loadmat(self.root + '/' + self.msi_dir + '/' + str(item))['data']) \
            .to(torch.float32)

        data = {"gt": self.GT,
                "hsi": self.HSI,
                "msi": self.MSI}

        return data
