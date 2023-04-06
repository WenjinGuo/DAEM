import numpy as np
import os
import cv2
import scipy.io as scio
import torch.nn.functional as F
import torch


# cave path (download cave dataset from http://www.cs.columbia.edu/CAVE/databases/)
file_dir = './CAVE'
# save path
save_dir = '../CAVE'


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


def read_img_cave(file_dir, height, width, bands):
    for root, dirs, files in os.walk(file_dir):
        hr_HSI = np.zeros([height, width, bands], dtype=np.float32)
        i = 0
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                img_band = cv2.imread(file_dir + '/' + file, cv2.IMREAD_GRAYSCALE)
                hr_HSI[:, :, i] = img_band
                i = i + 1

    return norm(hr_HSI)


# makedir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(save_dir + '/' + 'gt'):
    os.mkdir(save_dir + '/' + 'gt')


i = 0
for root, dirs, files in os.walk(file_dir, topdown=False):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.path.exists(os.path.join(dir_path, dir)):
            print('processing %d-th image' % (i + 1))
            hr_HSI = read_img_cave(dir_path, 512, 512, 31)
            scio.savemat(save_dir + '/' + 'gt' + '/' + str(i) + '.mat', {'data': hr_HSI})
            i = i + 1
