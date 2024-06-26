import numpy as np
import os
import cv2

def fsc231_linearize(im):
    knee_points_in = [0, 512, 1024, 1328, 1632, 1936, 2240, 2544, 2848, 3152, 3456, 3760, 4095]
    knee_points_out = [0, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15, 2 ** 16, 2 ** 17, 2 ** 18,
                       2 ** 19, 2 ** 20]
    decompanded = np.zeros_like(im)
    for idx in range(len(knee_points_in) - 1):
        mask = (knee_points_in[idx] <= im) & (im < knee_points_in[idx + 1])
        slope = (knee_points_out[idx + 1] - knee_points_out[idx]) / (knee_points_in[idx + 1] - knee_points_in[idx])
        decompanded = (1 - mask) * decompanded + mask * (knee_points_out[idx] + (im - knee_points_in[idx]) * slope)

    return decompanded


def fsc231_linearize_loader(path, whitelevel=(1 << 20) - 1, wb=np.array([1.0, 1.0, 1.0])):
    """ Linearize companded images from FSC231 """
    if whitelevel is None:
        whitelevel = (1 << 20) - 1
    if wb is None:
        wb = np.array([1.0, 1.0, 1.0])
    if os.path.splitext(path)[1] == '.raw':
        img = np.fromfile(path, np.uint16).reshape(-1, 3848)
    else:
        img = cv2.imread(path)

    dH2 = (img.shape[0] - 2160) // 2
    dH2 = dH2 + (dH2 & 1)
    img = img[dH2:-dH2]

    img = fsc231_linearize(img)
    wb = np.array(wb) / wb[1]
    """
    FSC231 is GRBG
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

def load_conti_image(filename):
    img = fsc231_linearize_loader(filename)
    img = norm(img, 16)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    img = np.log(img.astype(np.float32)+1)
    img = norm(img,8)
    img = contrast_stretch(img)
    return img
