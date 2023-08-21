from typing import Any, Dict
import os
from logging import Logger
import functools

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tensorboardX import SummaryWriter

from NetworkBaseModule.weightinit import init_weights
from Network import DMFNet
from loss import cal_loss_theta, cal_loss_phi
from Utils import utils_image


def net_design(opt: Dict[str, Any]):
    opt_model = opt["model"]
    Net = DMFNet(
        ConvLayerAbundanceParam=opt_model['ConvLayerAbundanceParam'],
        ConvLayerSpectralParamMean=opt_model['ConvLayerSpectralParamMean'],
        ConvLayerSpectralParamVar=opt_model['ConvLayerSpectralParamVar'],
        bands=opt['dataset']['bands'],
        bands_msi=opt['dataset']['bands_msi'],
        scale_factor=opt['dataset']['scale_factor'],
        num_endmember=opt['dataset']['num_endmember'],
        factors=opt_model['GaussianSRFParam']['factor'],
        device=opt['device']
    )

    if opt["phase"] == "train":
        init_weights(
            Net,
            init_type="kaiming_normal",
            init_bn_type="kaiming_normal",
            gain=0.2
        )

    return Net


class Model:
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.opt_train = self.opt['train']
        self.opt_loss = self.opt["loss"]
        self.opt_test = self.opt["test"]
        self.phase = self.opt["phase"]

        self.schedulers = []
        self.log_dict = {}
        self.metrics = {}

        self.device = opt["device"]
        self.pth_save_dir = self.opt_train["pth_save_dir"]
        if not os.path.exists(self.pth_save_dir):
            os.mkdir(self.pth_save_dir)
        self.pth_save_interval = self.opt_train["pth_save_interval"]
        self.checkpoint_interval = self.opt_train["checkpoint_interval"]
        self.srf = None

        self.result_save_dir = self.opt_test["result_save_dir"]
        self.ite_train = 0
        self.ite_test = 0

        self.net = net_design(opt).to(self.device)

        # tensorboard --logdir="runs/8nodes"
        # http://localhost:6006/#scalars
        self.writer = SummaryWriter('runs/8nodes')

    def init(self):
        self.load()
        self.net.train()

        self.def_loss()
        self.def_optimizer()
        self.def_scheduler()

    def load(self):
        if self.phase == 'train':
            if self.opt_train["load"] != 0:
                load_path = self.opt_train["pretrained_path"]
                if load_path != 'None':
                    print('Loading model for G [{:s}] ...'.format(load_path))
                    state_dict = torch.load(load_path)
                    self.net.load_state_dict(state_dict, strict=True)
        else:
            if self.opt_test["load"] != 0:
                load_path = self.opt_test["pretrained_path"]
                if load_path != 'None':
                    print('Loading model for G [{:s}] ...'.format(load_path))
                    state_dict = torch.load(load_path)
                    self.net.load_state_dict(state_dict, strict=True)

    def save(self, logger: Logger, num):
        logger.info('Saving the model')
        net = self.net
        self.save_net(net, 'net' + str(num))
        self.save_net(net, 'net')

    def save_net(self, network: nn.Module, network_name):
        filename = '{}.pth'.format(network_name)
        save_path = os.path.join(self.pth_save_dir, filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

    def def_optimizer(self):
        optimizer_params = []
        for _, v in self.net.named_parameters():
            optimizer_params.append(v)
        self.optimizer = Adam(optimizer_params,
                              lr=self.opt_train['G_optimizer_lr'],
                              weight_decay=0)

    def def_scheduler(self):
        self.schedulers.append(
            lr_scheduler.MultiStepLR(self.optimizer,
                                     self.opt_train['G_scheduler_milestones'],
                                     self.opt_train['G_scheduler_gamma']))

    def update_learning_rate(self, n: int):
        for scheduler in self.schedulers:
            scheduler.step(n)

    @property
    def learning_rate(self) -> float:
        return self.schedulers[0].get_last_lr()[0]

    def feed_data(self, data: Dict):
        self.GT = data['gt'].to(self.device)
        self.MSI = data['msi'].to(self.device)
        self.HSI = data['hsi'].to(self.device)
        if len(self.HSI.shape) == 3:
            self.HSI = self.HSI.unsqueeze(dim=0)
        if len(self.MSI.shape) == 3:
            self.MSI = self.MSI.unsqueeze(dim=0)
        if len(self.GT.shape) == 3:
            self.GT = self.GT.unsqueeze(dim=0)

    def get_size(self, size):
        [self.height, self.width, self.bands, self.height_hsi, self.width_hsi, self.bands_msi] = size

    def def_loss(self):
        self.lossfn_theta = functools.partial(
            cal_loss_theta,
            loss_type_train=self.opt_loss["type_train_theta"],
            loss_weights_train=self.opt_loss["weights_train_theta"],
            loss_type_test=self.opt_loss["type_test_theta"],
            loss_weights_test=self.opt_loss["weights_test_theta"],
            device=self.device
        )
        self.lossfn_phi = functools.partial(
            cal_loss_phi,
            loss_type_train=self.opt_loss["type_train_phi"],
            loss_weights_train=self.opt_loss["weights_train_phi"],
            loss_type_test=self.opt_loss["type_test_phi"],
            loss_weights_test=self.opt_loss["weights_test_phi"],
            device=self.device
        )

    def cal_net_loss_theta(self, X, Y, X_r, Y_r, X_ds_r, X_psf, Y_srf, phase):
        loss, loss_all = self.lossfn_theta(X, Y, X_r, Y_r, X_ds_r, X_psf, Y_srf, phase)

        return loss, loss_all

    def cal_net_loss_phi(self, X, X_ds_r, phase):
        loss, loss_all = self.lossfn_phi(X, X_ds_r, phase)

        return loss, loss_all

    def log_train(self, current_step: int, epoch: int, logger: Logger):
        message = ''
        self.log_dict['lr_train'] = self.learning_rate
        for k, v in self.log_dict.items(
        ):  # merge log information into message
            if self.phase in k:
                message += f', {k:s}: {v:.3e}'
        logger.info(message)

    def train(self):
        self.phase = "train"

        self.pth_save_dir = self.opt_train["pth_save_dir"]
        self.pth_save_interval = self.opt_train["pth_save_interval"]

        A_ds, A, Z_r, X_r, Y_r, X_ds_r, X_psf, Y_srf = self.net(self.HSI, self.MSI, stage="train")

        self.log_dict['lr_train'] = self.learning_rate

        self.optimizer.zero_grad()
        # optimize phi
        X_ds_r = self.net.forward_opt_phi(self.HSI, self.MSI)

        loss_phi, loss_all_phi = \
            self.cal_net_loss_phi(self.HSI, X_ds_r, self.phase)
        self.optimizer.zero_grad()
        for name, param in self.net.named_parameters():
            if ("SRF" in name) or ("PSF" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
        loss_back = loss_phi[0]
        loss_back.backward(retain_graph=True)

        # optimize theta
        loss_theta, loss_all_theta = \
            self.cal_net_loss_theta(self.HSI, self.MSI, X_r, Y_r, X_ds_r, X_psf, Y_srf, self.phase)

        self.log_dict['phase'] = 0.0
        self.log_dict['G_loss_theta_train'] = loss_all_theta.item()
        self.log_dict['hsi_likelihood_train'], self.log_dict['msi_likelihood_train'], \
        self.log_dict['regularization_degradation_train'] \
            = loss_theta[0], loss_theta[1], loss_theta[2]

        for name, param in self.net.named_parameters():
            if ("Encoder" in name) or ("Spectral" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
        loss_back0 = loss_theta[0] + loss_theta[1] + loss_theta[2]
        loss_back0.backward(retain_graph=True)

        self.log_dict['regularization_phi_train'] = loss_phi[0]

        self.optimizer.step()

        for name, param in self.net.named_parameters():
            param.requires_grad = True

        for k, v in self.log_dict.items():
            if 'train' in k:
                self.writer.add_scalar(k, v, global_step=self.ite_test)

        self.ite_train = self.ite_train + 1

        # del A, Z_r, X_r, Y_r, X_ds_r, Endmember_mean, Endmember_var
        # torch.cuda.empty_cache()

    def change_schedule(self):
        self.pth_save_dir = self.opt_test["pth_save_dir"]
        self.pth_save_interval = self.opt_test["pth_save_interval"]

    def test_incremental(self):
        self.phase = "test"

        self.net.eval()

        A_ds, A, Z_r, X_r, Y_r, X_ds_r, X_psf, Y_srf = self.net(self.HSI, self.MSI, self.A_ds, "fine-tune")

        self.log_dict['lr_train'] = self.learning_rate

        self.optimizer.zero_grad()

        loss_phi, loss_all_phi = \
            self.cal_net_loss_phi(self.HSI, X_ds_r, self.phase)
        self.optimizer.zero_grad()
        for name, param in self.net.named_parameters():
            if "Spectral" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss_back = loss_phi[0]
        loss_back.backward(retain_graph=True)

        self.log_dict['regularization_phi_test'] = loss_phi[0]

        self.optimizer.step()

        for name, param in self.net.named_parameters():
            param.requires_grad = True

    def test(self, is_validation=False):
        self.phase = "test"

        self.net.eval()

        with torch.no_grad():
            self.A_ds, A, Z_r, X_r, Y_r, X_ds_r, X_psf, Y_srf = self.net(self.HSI, self.MSI, stage="test")
            self.num_endmember = A.shape[1]
            loss_theta, loss_all_theta = \
                self.cal_net_loss_theta(self.HSI, self.MSI, X_r, Y_r, X_ds_r, X_psf, Y_srf, self.phase)
            loss_phi, loss_all_phi = \
                self.cal_net_loss_phi(self.HSI, X_ds_r, self.phase)

        X_r = X_r.squeeze(dim=0).permute(1, 2, 0)
        X = self.HSI.squeeze(dim=0)
        Z_r = Z_r.squeeze(dim=0).permute(1, 2, 0)
        Z = self.GT.squeeze(dim=0)
        Y = self.MSI.squeeze(dim=0)
        Y_r = Y_r.squeeze(dim=0).permute(1, 2, 0)
        X_ds_r = X_ds_r.squeeze(dim=0).permute(1, 2, 0)

        X = X.detach().to(device='cpu').numpy()
        # X = utils_image.normlization(X) * 255
        X = X * 255
        X_r = X_r.detach().to(device='cpu').numpy()
        # X_r = utils_image.normlization(X_r) * 255
        X_r = X_r * 255
        Z = Z.detach().to(device='cpu').numpy()
        # Z = utils_image.normlization(Z) * 255
        Z = Z * 255
        Z_r = Z_r.detach().to(device='cpu').numpy()
        # Z_r = utils_image.normlization(Z_r) * 255
        Z_r = Z_r * 255
        Y = Y.detach().to(device='cpu').numpy()
        # Y = utils_image.normlization(Y) * 255
        Y = Y * 255
        Y_r = Y_r.reshape(self.height, self.width, self.bands_msi).detach().to(device='cpu').numpy()
        # Y_rz = utils_image.normlization(Y_rz) * 255
        Y_r = Y_r * 255
        X_ds_r = X_ds_r.detach().to(device='cpu').numpy()
        X_ds_r = X_ds_r * 255

        rmse_zr = utils_image.rmse(Z, Z_r)
        psnr_zr, sam_zr, ergas_zr, ssim_zr, uqi_zr = utils_image.MetricsCal(Z, Z_r, scale=32)

        psnr_yr = utils_image.psnr(Y, Y_r)
        sam_yr = utils_image.sam(Y, Y_r)

        psnr_xr = utils_image.psnr(X, X_r)
        sam_xr = utils_image.sam(X, X_r)

        psnr_xdsr = utils_image.psnr(X, X_ds_r)
        sam_xdsr = utils_image.sam(X, X_ds_r)

        self.result = Z_r
        self.PSF = self.net.PSFEst.KernelAdaption.squeeze(dim=0).squeeze(dim=0).to(device='cpu').detach().numpy()
        self.SRF = self.net.SRFEst.SRF.to(device='cpu').detach().numpy()

        if is_validation:
            self.log_dict['G_loss_val'] = loss_all_theta.item()
            self.log_dict['hsi_likelihood_val'], self.log_dict['msi_likelihood_val'], \
            self.log_dict['regularization_degradation_val'] \
                = loss_theta[0], loss_theta[1], loss_theta[2]
            self.log_dict['psnr_xr_val'] = psnr_xr
            self.log_dict['sam_xr_val'] = sam_xr
            self.log_dict['psnr_yr_val'] = psnr_yr
            self.log_dict['sam_yr_val'] = sam_yr
            self.log_dict['psnr_zr_val'] = psnr_zr
            self.log_dict['sam_zr_val'] = sam_zr
            self.log_dict['ssim_zr_val'] = ssim_zr
            self.log_dict['rmse_zr_val'] = rmse_zr
            self.log_dict['ergas_zr_val'] = ergas_zr
            self.log_dict['uqi_zr_val'] = uqi_zr
            self.log_dict['psnr_xdsr_val'] = psnr_xdsr
            self.log_dict['sam_xdsr_val'] = sam_xdsr
        else:
            self.log_dict['G_loss_test'] = loss_all_theta.item()
            self.log_dict['hsi_likelihood_test'], self.log_dict['msi_likelihood_test'], \
            self.log_dict['regularization_degradation_test'] \
                = loss_theta[0], loss_theta[1], loss_theta[2]
            self.log_dict['regularization_phi_test'] = loss_phi[0]
            self.log_dict['psnr_xr_test'] = psnr_xr
            self.log_dict['sam_xr_test'] = sam_xr
            self.log_dict['psnr_yr_test'] = psnr_yr
            self.log_dict['sam_yr_test'] = sam_yr
            self.log_dict['psnr_zr_test'] = psnr_zr
            self.log_dict['sam_zr_test'] = sam_zr
            self.log_dict['ssim_zr_test'] = ssim_zr
            self.log_dict['rmse_zr_test'] = rmse_zr
            self.log_dict['ergas_zr_test'] = ergas_zr
            self.log_dict['uqi_zr_test'] = uqi_zr
            self.log_dict['psnr_xdsr_test'] = psnr_xdsr
            self.log_dict['sam_xdsr_test'] = sam_xdsr

        for k, v in self.log_dict.items():
            if 'test' or 'val' in k:
                self.writer.add_scalar(k, v, global_step=self.ite_test)

        self.ite_test = self.ite_test + 1

        self.net.train()
