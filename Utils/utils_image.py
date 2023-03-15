import numpy as np
import math


def psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0):
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w, bands = img1.shape
    # img1 = img1[..., border:h - border, border:w - border]
    # img2 = img2[..., border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    psnr_bands = np.zeros([bands])
    for i in range(bands):
        psnr_bands[i] = 20 * math.log10(np.max(img1[:, :, i]) / np.sqrt(np.mean((img1[:, :, i] - img2[:, :, i])**2)))
        # psnr_bands[i] = 20 * math.log10(255. / np.sqrt(np.mean((img1[:, :, i] - img2[:, :, i]) ** 2)))
    return np.mean(psnr_bands)


def sam(img_true: np.ndarray, img_pred: np.ndarray):
    assert img_true.shape == img_pred.shape
    if len(img_true.shape) == 3:
        img_true = np.expand_dims(img_true, 0)
        img_pred = np.expand_dims(img_pred, 0)

    batchsize, h, w, bands = img_true.shape
    img_true = img_true.reshape(batchsize * h * w, bands)
    img_pred = img_pred.reshape(batchsize * h * w, bands)

    img_pred[np.where((np.linalg.norm(img_pred, 2, 1)) == 0), ] += 0.0001
    img_true[np.where((np.linalg.norm(img_true, 2, 1)) == 0), ] += 0.0001

    eps = 1e-6
    sam = (img_true * img_pred).sum(axis=1) / ((np.linalg.norm(img_true, 2, 1) + eps) *
                                               (np.linalg.norm(img_pred, 2, 1) + eps))

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()

    return mSAM


def rmse(img_true: np.ndarray, img_pred: np.ndarray):
    assert img_true.shape == img_pred.shape

    h, w, bands = img_true.shape

    temp = np.linalg.norm(img_pred - img_true, ord=None, axis=None)
    rmse_ = temp / np.sqrt((h*w*bands))

    return rmse_


def ssim(img_true: np.ndarray, img_pred: np.ndarray):
    L = 255
    assert img_true.shape == img_pred.shape

    h, w, bands = img_true.shape

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2

    ssim_ = .0

    for i in range(bands):
        img_true_i = img_true[:, :, i]
        img_pred_i = img_pred[:, :, i]
        # ux
        ux = img_true_i.mean()
        # uy
        uy = img_pred_i.mean()
        # ux^2
        ux_sq = ux ** 2
        # uy^2
        uy_sq = uy ** 2
        # ux*uy
        uxuy = ux * uy
        # ox、oy方差计算
        ox_sq = img_true_i.var()
        oy_sq = img_pred_i.var()
        ox = np.sqrt(ox_sq)
        oy = np.sqrt(oy_sq)
        oxoy = ox * oy
        oxy = np.mean((img_true - ux) * (img_pred - uy))
        # 公式一计算
        L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
        C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
        S = (oxy + C3) / (oxoy + C3)
        ssim = L * C * S
        # 验证结果输出
        # print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
        ssim_ = ssim_ + ssim

    return ssim_/bands


def ergas(img_true: np.ndarray, img_pred: np.ndarray):
    assert img_true.shape == img_pred.shape

    h, w, bands = img_true.shape
    err = img_true - img_pred
    ergas_ = 0.
    for i in range(bands):
        ergas_ = ergas_ + np.mean(err[:, :, i] ** 2) / np.mean(img_true[:, :, i])**2

    ergas_ = (100/32) * np.sqrt(1/bands * ergas_)

    return ergas_


def normlization(img: np.ndarray):
    max_ = img.max()
    min_ = img.min()
    img = (img - min_) / (max_ - min_)

    return img


def compute_ergas(img1, img2, scale):
    d = img1 - img2
    ergasroot = 0
    for i in range(d.shape[2]):
        ergasroot = ergasroot + np.mean(d[:, :, i] ** 2) / np.mean(img1[:, :, i]) ** 2

    ergas = 100 / scale * np.sqrt(ergasroot / d.shape[2])
    return ergas


def compute_psnr(img1, img2):
    assert img1.ndim == 3 and img2.ndim == 3

    img_w, img_h, img_c = img1.shape
    ref = img1.reshape(-1, img_c)
    tar = img2.reshape(-1, img_c)
    msr = np.mean((ref - tar) ** 2, 0)
    max1 = np.max(ref, 0)

    psnrall = 10 * np.log10(max1 ** 2 / msr)
    out_mean = np.mean(psnrall)
    return out_mean, max1


def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape

    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)

    x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

    eps = 1e-6
    sam = (x_true * x_pred).sum(axis=1) / ((np.linalg.norm(x_true, 2, 1) + eps)
                                           * (np.linalg.norm(x_pred, 2, 1) + eps))

    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()
    var_sam = np.var(sam)
    return mSAM, var_sam


def MetricsCal(GT, P, scale):  # c,w,h

    m1, GTmax = compute_psnr(GT, P)  # bandwise mean psnr

    m2, _ = compute_sam(GT, P)  # sam

    m3 = compute_ergas(GT, P, scale)

    from skimage.metrics import structural_similarity as ssim
    ssims = []
    for i in range(GT.shape[2]):
        ssimi = ssim(GT[:, :, i], P[:, :, i], data_range=P[:, :, i].max() - P[:, :, i].min())
        ssims.append(ssimi)
    m4 = np.mean(ssims)

    from sewar.full_ref import uqi
    m5 = uqi(GT, P)

    return np.float64(m1), np.float64(m2), m3, m4, m5