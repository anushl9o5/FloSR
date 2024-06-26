import cv2
import numpy as np

def read_raw(f):
    try: 
        return np.fromfile(f, 'uint16').reshape(-1,1928)
    except:
        return None

def decompand_12bit_legacy_ar0231(im):
    filter_2047 = im <= 2047
    filter_3040 = im <= 3040
    im = (filter_2047) * im + ((~filter_2047) & (filter_3040)) * ((im - 2048) * 64 + 2048) + \
        (~filter_3040) * ((im - 3040) * 1024 + 65536)
    # im = (im <= 2047) * im + ((im >= 2048) & (im <= 3040)) * ((im - 2048) * 64. + 2048) + \
        # (im >= 3040) * ((im - 3040) * 1024 + 65536)
    return im

def linearize(buffer: np.ndarray): 
    """ Decompand image from sensor ar0231 
    Args: 
        buffer: 14 bits compended buffer image 
    Return: 
        20 bits linear image""" 
    raw_64 = buffer.astype(np.int64) 
    raw_20bit_linear_tmp = np.where(buffer > 2047, (raw_64 - 2048) * 64 + 2048, raw_64) 
    raw_20bit_linear_tmp = np.where(buffer > 3040, (raw_64 - 3040) * 1024 + 65536, raw_20bit_linear_tmp)
    raw_20bit_linear_tmp.clip(0, 2 ** 20 - 1, out=raw_20bit_linear_tmp) 
    return raw_20bit_linear_tmp.astype(np.float) 

def white_balance(img: np.ndarray, wb): 
    img_out = np.float64(img) 
    ch_arr = [(0, 0), (0, 1), (1, 0), (1, 1)] 
    rch = (0,1)#ch_arr[self.BAYER.upper().find('R')] 
    bch = (1,0)#ch_arr[self.BAYER.upper().find('B')] 
    img_out[rch[0]::2, rch[1]::2] = img_out[rch[0]::2, rch[1]::2] * wb[0]
    img_out[bch[0]::2, bch[1]::2] = img_out[bch[0]::2, bch[1]::2] * wb[1]
    return img_out 

def norm(im, d, whitelevel, blacklevel): 
    # return ((im-im.min()) / (im.max()-im.min()) * (2**d-1)).astype(f'uint{d}')
    return ((im-im.min()) / (im.max()-im.min()) * (2**d-1)).astype(f'uint{d}')

def norm8(im, whitelevel=131072, blacklevel=0): 
    # return ((im-im.min()) / (im.max()-im.min()) * (255)).astype(f'uint8')
    return (im / im.max() * (255)).astype(f'uint8')
    # return ((im-blacklevel) / (whitelevel-blacklevel) * (255)).astype(f'uint8')

def norm16(im, whitelevel=131072, blacklevel=0): 
    # return ((im-im.min()) / (im.max()-im.min()) * (65535)).astype(f'uint16')
    return (im / im.max() * (65535)).astype(f'uint16')
    # return ((im-blacklevel) / (whitelevel-blacklevel) * (65535)).astype(f'uint16')


def contrast_stretch(img: np.ndarray): 
    # TODO add docstring 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0]) 
    img[:, :, 1] = clahe.apply(img[:, :, 1]) 
    img[:, :, 2] = clahe.apply(img[:, :, 2]) 
    return img 

def convert(filename):
    im = read_raw(filename)[4:-2]
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_GB2BGR)
    
    whitelevel = 2600
    blacklevel = 0
    im = ((im-blacklevel) / (whitelevel-blacklevel) * (255)).astype(f'uint8')
    
    return im[...,::-1] 

def convert_raw2rgb_fast(raw_im):
    im = raw_im
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_GB2BGR)
    whitelevel = 2600
    blacklevel = 0
    im = ((im-blacklevel) // int((whitelevel-blacklevel)/ (255))).astype(f'uint8')

    return im[...,::-1]

def decompand_12bit_legacy_ar0231(im):
    filter_2047 = im <= 2047
    filter_3040 = im <= 3040
    im = (filter_2047) * im + ((np.logical_not(filter_2047)) & (filter_3040)) * ((im - 2048) * 64 + 2048) + \
        (np.logical_not(filter_3040)) * ((im - 3040) * 1024 + 65536)
    return im

def norm(im, d): 
    return ((im-im.min()) / (im.max()-im.min()) * (2**d-1)).astype(f'uint{d}')

def contrast_stretch(img: np.ndarray): 
    # TODO add docstring 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0]) 
    img[:, :, 1] = clahe.apply(img[:, :, 1]) 
    img[:, :, 2] = clahe.apply(img[:, :, 2]) 
    return img 

def convert_raw2rgb(raw_im):
#     im = read_raw(filename)[4:-2]
    im = raw_im[4:-2]
    # im = decompand_12bit_legacy_ar0231(im)
    im = linearize(im)
    im = norm(im, 16)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_GB2BGR)
    im = np.log(im.astype(np.float32)+1)
    im = norm(im, 8)
    im = contrast_stretch(im) 
    # return im[...,::-1]
    return im
