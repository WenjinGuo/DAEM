import numpy as np


def blur_downsample(img,
                    blur_kernel,
                    scale_factor):
    """
    Simulate the degradation to lr-hsi
    """
    height, width, bands = img.shape[0], img.shape[1], img.shape[2]
    kernel_size = blur_kernel.shape[0]
    if kernel_size != blur_kernel.shape[1]:
        raise Exception('Height and width of blur kernel should be equal')

    # Padding
    img_aligned = np.zeros((height + kernel_size - scale_factor, width + kernel_size - scale_factor, bands))
    img_aligned[(kernel_size - scale_factor) // 2:height + (kernel_size - scale_factor) // 2,
    (kernel_size - scale_factor) // 2:width + (kernel_size - scale_factor) // 2, :] = img

    # Only calculate the needed pixels
    img_result = np.zeros((height // scale_factor, width // scale_factor, bands))
    for i in range(height // scale_factor):
        for j in range(width // scale_factor):
            A = np.multiply(img_aligned[i * scale_factor:i * scale_factor + kernel_size,
                            j * scale_factor:j * scale_factor + kernel_size, :],
                            blur_kernel[:, :, None])
            A = np.sum(A, axis=0)
            A = np.sum(A, axis=0)
            img_result[i, j, :] = A

    return img_result


def spectral_downsample(img,
                        srf):
    height, width, bands = img.shape[0], img.shape[1], img.shape[2]
    bands_msi = srf.shape[1]
    img = img.reshape(height * width, bands)
    img_result = np.matmul(img, srf).reshape(height, width, bands_msi)

    return img_result
