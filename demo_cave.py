import os
import commentjson
import logging
from typing import List
import numpy as np
import random
from tqdm import tqdm
import scipy.io as sio
import time

import torch
from torch import cuda
from torch.utils.data import DataLoader

from NetworkBaseModule.dataload import DataSetBlock, DataSetImgPatch
from model import Model
from Utils import utils_logger, utils_option, utils_record


def train(json_path: str = "./setting.json", train_index: List = [0, 1], test_index: List = [2, 3],
                 xlsx_path: str = "./index_record_train.xls"):
    with open(json_path) as file:
        opt = commentjson.load(file)

    logger_name = 'train'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log']['path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(utils_option.dict2str(opt))

    seed = opt['train']['manual_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    batch_size = opt["train"]["batchsize"]
    num_epoch = opt["train"]["num_epoch"]
    val_interval = opt["train"]["val_interval"]

    img_patchs = DataSetImgPatch(
            dataset=opt["dataset"],
            imgsize=[opt["dataset"]["height_hsi"], opt["dataset"]["width_hsi"], opt["dataset"]["bands"],
                     opt["dataset"]["bands_msi"]],
            scale_factor=opt["dataset"]["scale_factor"],
            blocksize=opt["train"]["blocksize"],
            data_index=train_index
        )

    img_batchs = DataLoader(img_patchs, batch_size=batch_size, shuffle=opt["train"]["shuffle"], num_workers=1)

    start_time = time.time()

    net = Model(opt)
    net.init()

    ite = img_patchs.height_hsi*img_patchs.width_hsi // (img_patchs.block_height * img_patchs.block_width)

    epoch_bar = tqdm(range(num_epoch))
    index_name = ['psnr_xr', 'sam_xr', 'psnr_yr', 'sam_yr',
                  'psnr_zr', 'sam_zr', 'ssim_zr', 'rmse_zr', 'ergas_zr', 'uqi_zr', 'psnr_xdsr', 'sam_xdsr',
                  'loss_all', 'loss_0', 'loss_1', 'loss_2', 'loss_3']
    index_better_condition = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    index_value = np.zeros([len(test_index), num_epoch//val_interval, len(index_name)])
    record_index = 0
    for epoch in epoch_bar:
        if epoch % val_interval == 0:
            # validate with training set (compared with validation on testing set
            # for judging over-fitting or under-fitting)
            val_psnr = 0
            val_sam = 0
            val_ergas = 0
            val_loss = [0., 0., 0., 0., 0., 0.]
            for i, img_index in enumerate(train_index):
                img_patchs_test = DataSetBlock(
                    dataset=opt["dataset"],
                    data_index=img_index
                )

                img_batchs_test = DataLoader(img_patchs_test, batch_size=1, shuffle=False)

                for batch in img_batchs_test:
                    net.feed_data(batch)
                    net.get_size(img_patchs_test.size)
                    net.test(is_validation=True)

                val_psnr = val_psnr + net.log_dict["psnr_zr_val"]
                val_sam = val_sam + net.log_dict["sam_zr_val"]
                val_ergas = val_ergas + net.log_dict["ergas_zr_val"]
                val_loss = val_loss + [net.log_dict['G_loss_val'], net.log_dict['hsi_likelihood_val'],
                                       net.log_dict['msi_likelihood_val'],
                                       net.log_dict['regularization_degradation_val']]

            net.log_dict['psnr_val_mean'] = val_psnr / len(train_index)
            net.log_dict['sam_val_mean'] = val_sam / len(train_index)
            net.log_dict['ergas_val_mean'] = val_ergas / len(train_index)
            net.log_dict['lossall_val_mean'] = val_loss[0] / len(train_index)
            net.log_dict['loss0_val_mean'] = val_loss[1] / len(train_index)
            net.log_dict['loss1_val_mean'] = val_loss[2] / len(train_index)
            net.log_dict['loss2_val_mean'] = val_loss[3] / len(train_index)

            # validate with test set
            test_psnr = 0
            test_sam = 0
            test_ergas = 0
            test_loss = [0., 0., 0., 0., 0., 0.]
            for i, img_index in enumerate(test_index):
                img_patchs_test = DataSetBlock(
                    dataset=opt["dataset"],
                    data_index=img_index
                )

                img_batchs_test = DataLoader(img_patchs_test, batch_size=1, shuffle=False)

                for batch in img_batchs_test:
                    net.feed_data(batch)
                    net.get_size(img_patchs_test.size)
                    net.test(is_validation=False)

                psnr_xr = net.log_dict['psnr_xr_test']
                sam_xr = net.log_dict['sam_xr_test']
                psnr_zr = net.log_dict['psnr_zr_test']
                sam_zr = net.log_dict['sam_zr_test']
                ssim_zr = net.log_dict['ssim_zr_test']
                rmse_zr = net.log_dict['rmse_zr_test']
                ergas_zr = net.log_dict['ergas_zr_test']
                uqi_zr = net.log_dict['uqi_zr_test']
                psnr_yr = net.log_dict['psnr_yr_test']
                sam_yr = net.log_dict['sam_yr_test']
                psnr_xdsr = net.log_dict['psnr_xdsr_test']
                sam_xdsr = net.log_dict['sam_xdsr_test']

                loss_all = net.log_dict['G_loss_test']
                loss_0, loss_1, loss_2 = \
                    net.log_dict['hsi_likelihood_test'], net.log_dict['msi_likelihood_test'], \
                    net.log_dict['regularization_degradation_test']

                loss_3 = net.log_dict['regularization_phi_test']

                index_value[i, record_index, :] = [psnr_xr, sam_xr, psnr_yr, sam_yr,
                                                   psnr_zr, sam_zr, ssim_zr, rmse_zr, ergas_zr, uqi_zr,
                                                   psnr_xdsr, sam_xdsr,
                                                   loss_all, loss_0, loss_1, loss_2, loss_3]

                if epoch % net.checkpoint_interval == 0 and epoch != 0:
                    net.log_train(batch_index, epoch, logger)

                save_dir = "./results/"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                sio.savemat(save_dir + str(img_index) + '.mat', {'data': net.result})

                test_psnr = test_psnr + net.log_dict["psnr_zr_test"]
                test_sam = test_sam + net.log_dict["sam_zr_test"]
                test_ergas = test_ergas + net.log_dict["ergas_zr_test"]
                test_loss = test_loss + [net.log_dict['G_loss_test'], net.log_dict['hsi_likelihood_test'],
                                         net.log_dict['msi_likelihood_test'],
                                         net.log_dict['regularization_degradation_test']]

            sio.savemat(save_dir + 'psf_train' + str(record_index) + '.mat', {'data': net.PSF})

            record_index = record_index + 1

        # training process
        for batch_index, batch in enumerate(img_batchs):
            net.feed_data(batch)
            net.get_size(img_patchs.size)

            net.train()

            net.update_learning_rate(epoch)

        net.save(logger, 0)
        if (epoch + 1) % net.pth_save_interval == 0 and epoch != 0:
            net.save(logger, epoch)
        if epoch % net.checkpoint_interval == 0 and epoch != 0:
            net.log_train(batch_index, epoch, logger)

    end_time = time.time()

    print('start: {}; end: {}'.format(start_time, end_time))

    for i in range(len(test_index)):
        utils_record.xlsx_record(index_name, index_value[i, :, :], index_better_condition, test_index[i], xlsx_path)


def test(json_path: str = "./setting.json", test_index: List = [2, 3],
                 xlsx_path: str = "./index_record_test.xls"):
    with open(json_path) as file:
        opt = commentjson.load(file)

    logger_name = 'test'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log']['path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(utils_option.dict2str(opt))

    seed = opt['train']['manual_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    num_epoch = opt["test"]["num_epoch"]
    val_interval = opt["test"]["val_interval"]

    start_time = time.time()

    net = Model(opt)
    net.phase = "test"
    net.change_schedule()

    epoch_bar = tqdm(range(num_epoch))
    index_name = ['psnr_xr', 'sam_xr', 'psnr_yr', 'sam_yr',
                  'psnr_zr', 'sam_zr', 'ssim_zr', 'rmse_zr', 'ergas_zr', 'uqi_zr', 'psnr_xdsr', 'sam_xdsr',
                  'loss_all', 'loss_0', 'loss_1', 'loss_2', 'loss_3']
    index_better_condition = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    index_value = np.zeros([len(test_index), num_epoch, len(index_name)])

    # testing process
    for i, img_index in enumerate(test_index):
        img_patchs_test = DataSetBlock(
            dataset=opt["dataset"],
            data_index=img_index
        )
        img_batchs_test = DataLoader(img_patchs_test, batch_size=1, shuffle=False)
        record_index = 0

        net.init()

        for epoch in epoch_bar:
            if epoch % val_interval == 0:
                for batch in img_batchs_test:
                    net.feed_data(batch)
                    net.get_size(img_patchs_test.size)
                    net.test(is_validation=False)

                psnr_xr = net.log_dict['psnr_xr_test']
                sam_xr = net.log_dict['sam_xr_test']
                psnr_zr = net.log_dict['psnr_zr_test']
                sam_zr = net.log_dict['sam_zr_test']
                ssim_zr = net.log_dict['ssim_zr_test']
                rmse_zr = net.log_dict['rmse_zr_test']
                ergas_zr = net.log_dict['ergas_zr_test']
                uqi_zr = net.log_dict['uqi_zr_test']
                psnr_yr = net.log_dict['psnr_yr_test']
                sam_yr = net.log_dict['sam_yr_test']
                psnr_xdsr = net.log_dict['psnr_xdsr_test']
                sam_xdsr = net.log_dict['sam_xdsr_test']

                loss_all = net.log_dict['G_loss_test']
                loss_0, loss_1, loss_2 = \
                    net.log_dict['hsi_likelihood_test'], net.log_dict['msi_likelihood_test'], \
                    net.log_dict['regularization_degradation_test']
                loss_3 = net.log_dict['regularization_phi_test']

                index_value[i, record_index, :] = [psnr_xr, sam_xr, psnr_yr, sam_yr,
                                                   psnr_zr, sam_zr, ssim_zr, rmse_zr, ergas_zr, uqi_zr,
                                                   psnr_xdsr, sam_xdsr,
                                                   loss_all, loss_0, loss_1, loss_2, loss_3]

                net.update_learning_rate(epoch)

                if epoch % net.checkpoint_interval == 0 and epoch != 0:
                    net.log_train(record_index, epoch, logger)

                if epoch % net.checkpoint_interval == 0 and epoch != 0:
                    net.log_train(record_index, epoch, logger)

                save_dir = "./results/"
                sio.savemat(save_dir + str(img_index) + '_test.mat', {'data': net.result})

                record_index = record_index + 1
            else:
                for batch in img_batchs_test:
                    net.feed_data(batch)
                    net.get_size(img_patchs_test.size)
                    net.test_incremental()

    end_time = time.time()

    print('start: {}; end: {}'.format(start_time, end_time))

    for i in range(len(test_index)):
        utils_record.xlsx_record(index_name, index_value[i, :, :], index_better_condition, test_index[i], xlsx_path)


def main(json_path: str = "./setting.json"):
    train_index = [i for i in range(0, 16)]
    test_index = [i for i in range(16, 32)]

    train(json_path=json_path, train_index=train_index, test_index=test_index, xlsx_path="./index_record_train.xls")
    test(json_path=json_path, test_index=test_index, xlsx_path="./index_record_test.xls")


if __name__ == '__main__':
    main()
