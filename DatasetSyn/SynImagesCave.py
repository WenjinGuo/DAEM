import os

import numpy as np
import scipy.io as scio

from DegradationProcess import blur_downsample, spectral_downsample


data_root = "../CAVE/"
data_path = "../CAVE/gt/"
degradation_path = "../Degradation_params"

if not os.path.exists(degradation_path):
    os.mkdir(degradation_path)

for j in range(32):
    gt = scio.loadmat(data_path + str(j) + '.mat')['data']

    srf = scio.loadmat(degradation_path + '/srf/Nikon_srf.mat')['srf']
    msi = spectral_downsample(gt, srf)
    if not os.path.exists(data_root + 'msi'):
        os.mkdir(data_root + 'msi')
    scio.savemat(data_root + 'msi' + '/' + str(j) + '.mat', {'data': msi})

    for i in range(6):

        kernel = scio.loadmat(degradation_path + '/blur_kernel/' + str(i) + '.mat')['data']

        if not os.path.exists(data_root + 'hsi_' + str(i)):
            os.mkdir(data_root + 'hsi_' + str(i))
        hsi = blur_downsample(gt, kernel, scale_factor=32)
        scio.savemat(data_root + 'hsi_' + str(i) + '/' + str(j) + '.mat', {'data': hsi})

        print('generate {0} img in param {1}'.format(j, i))

