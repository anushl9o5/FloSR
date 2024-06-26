from imageio import imread
import numpy as np
import os
import cv2

def ar0820_linearize(data):
    if data.max() > 2**12:
        data = data >> 4
    _knee_points_in = [0, 0x200, 0x400, 0x800, 0x820, 0x860, 0x8E0, 0x9E0, 0xBE0, 0xC20,
                               0xCA0, 0xDA0, 0xFA0]
    _knee_points_out = [0, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17,
                                2**18, 2**19, 2**20]
    decompanded = np.zeros_like(data)
    for i in range(len(_knee_points_in) - 1):
        mask = (_knee_points_in[i] <= data) & (data < _knee_points_in[i + 1])
        slope = (_knee_points_out[i + 1] - _knee_points_out[i]) / (_knee_points_in[i + 1] - _knee_points_in[i])
        decompanded = (1 - mask) * decompanded + mask * (_knee_points_out[i] + (data - _knee_points_in[i]) * slope)

    return decompanded


def ar0820_linearize_loader(path, whitelevel=(1 << 20) - 1, wb=np.array([1.0, 1.0, 1.0])):
    """ Linearize companded images from AR0820 """
    if whitelevel is None:
        whitelevel = (1 << 22) - 1
    if wb is None:
        wb = np.array([1.0, 1.0, 1.0])
    if os.path.splitext(path)[1] == '.raw':
        img = np.fromfile(path, np.uint16)
        size = img.shape[0]
        if size % 3840 == 0:
            img = img.reshape(-1, 3840)
        elif size % 3848 == 0:
            img = img.reshape(-1, 3848)
        else:
            assert False, f'Unknown shape {size}'
    else:
        img = imread(path)
    #if img.max() > (1 << 12 - 1):
    #    img >>= 4
    dH2 = (img.shape[0] - 2160) // 2
    dH2 = dH2 + (dH2 & 1)
    img = img[dH2:-dH2]

    img = ar0820_linearize(img)
    wb = np.array(wb) / wb[1]
    """
    AR0820 is a YRCyY (ordering similar ot GRBG)
    Y R
    Cy Y
    The indices are
    00,01
    10,11
    In the following, 0::2, 1::2 means 01 (X::Y is start from X and stride 2)
    and 1::2, 0::2 is 10
    """
    img[0::2, 1::2] = img[0::2, 1::2] / wb[0]
    img[1::2, ::2] = img[1::2, 0::2] / wb[2]
    return np.clip(img.astype(np.float32) / whitelevel, 0, 1)


def norm(im, d):
    return ((im-im.min()) / (im.max()-im.min()) * (2**d-1)).astype(f'uint{d}')

def contrast_stretch(img: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    return img

def load_ar0820_image(filename):
    img = ar0820_linearize_loader(filename)
    img = norm(img, 16)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    img = np.log(img.astype(np.float32)+1)
    img = norm(img,8)
    img = contrast_stretch(img)
    return img

if __name__ == "__main__":
    img = load_ar0820_image('/nas/EOS/dataset/torc/wk3/fixed_structure/CAPT-273/dusk_CAPT-273_20211203164924/ar0820/image0/frame_image0_00026454.raw')
    cv2.imwrite('test.png', img)
