from __future__ import absolute_import, division, print_function

import os
import random
from turtle import left
from matplotlib import use
import numpy as np
import csv
import json
import copy
import pandas as pd
from PIL import Image  # using pillow-simd for increased speed
import glob
import cv2
from scipy.spatial.transform import Rotation
from utility.data_utils import calibrate_stereo_pair

import torch
import torch.utils.data as data
from torchvision import transforms

def rgb_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

def crop_img(img:np.ndarray, crop_height:int, crop_width:int)->np.ndarray:
    """create central crop for the image

    Args:
        img (np.ndarray): input image
        crop_height (np.ndarray): _description_
        crop_width (np.ndarray): _description_

    Returns:
        np.ndarray: central cropped image
    """
    height,width = img.shape[:2]
    top = (height - crop_height)//2
    left = (width - crop_width)//2
    bottom = (height + crop_height)//2
    right = (width + crop_width)//2

    return img[top:bottom,left:right,...]

def resize_img(img:np.ndarray, crop_height:int, crop_width:int)->np.ndarray:
    """create resized image of crop dimension

    Args:
        img (np.ndarray): input image
        crop_height (np.ndarray): _description_
        crop_width (np.ndarray): _description_

    Returns:
        np.ndarray: resized image
    """
    return cv2.resize(img, (crop_width, crop_height), interpolation=cv2.INTER_AREA)

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, scale = 4, verbose=False, invert=False):
 
    kernel_x = np.linspace(-(size[0] // 2), size[0] // 2, size[0])
    for i in range(size[0]):
        kernel_x[i] = dnorm(kernel_x[i], 0, size[0]/scale)

    kernel_y = np.linspace(-(size[1] // 2), size[1] // 2, size[1])
    for i in range(size[1]):
        kernel_y[i] = dnorm(kernel_y[i], 0, size[1]/scale)

    kernel_2D = np.outer(kernel_x.T, kernel_y.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()

    if invert:
        kernel_2D = 1-kernel_2D
 
    if verbose:
        cv2.imshow('mask', kernel_2D)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return kernel_2D

def transpose_dict(pandas_dict):
    converted_dict = {}

    for item in pandas_dict:
        key = item['sensor_name']
        converted_dict[key] = item
    
    return converted_dict

class Argo2Dataset(data.Dataset):

    def __init__(self, pose_gt_dir,
                       frames_dir,
                       datafile_path,
                       height,
                       width,
                       is_train,
                       pose_repr,
                       is_crop,
                       resize_orig_img,
                       perturbation_level) -> None:
        super(Argo2Dataset, self).__init__()
        self.datafile_path = datafile_path
        self.pose_repr = pose_repr

        self.height = height
        self.width = width
        
        self.is_train = is_train

        self.rgb_loader = rgb_loader

        self.is_crop = is_crop

        self.resize_orig_img = resize_orig_img

        self.perturbation_level = perturbation_level
        #print('Reading groundtruth pose information from json files...')
        #self.load_pose_json_data()

        self.data = []
        print('Reading training files information...')
        with open(self.datafile_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.data.append(row)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.to_tensor = transforms.ToTensor()

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if 'target' in k or 'reference' in k:
                inputs[k+'_aug'] = color_aug(inputs[k])

        for k in list(inputs):
            if 'target' in k or 'reference' in k:
                inputs[k] = self.to_tensor(inputs[k])

    def __getitem__(self, index):
        
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        # do_flip = self.is_train and random.random() > 0.5
        left_frame_path, right_frame_path, extrinsic_calib, intrinsic_calib, capture = self.data[index]
        
        # read calibration data
        inputs['dataset'] = capture

        self.parse_calib(extrinsic_calib, intrinsic_calib)
        
        R, t = np.array(self.pose_gt['rmat']).reshape(3,3), np.array(self.pose_gt['tvec']).reshape(3,)

        if self.pose_repr in ['use_perturb', 'use_perturb_6D', 'use_perturb_quat']:
            R_euler = Rotation.from_matrix(R).as_euler('xyz', degrees = True)
        
            if random.random() < 0.5:
                R_perturb_euler = R_euler + np.random.normal(0, self.perturbation_level, R_euler.shape)
            else:
                R_perturb_euler = R_euler
        
            R_perturb = Rotation.from_euler('xyz', R_perturb_euler, degrees = True).as_matrix()

        elif self.pose_repr in ['use_yaw_perturb']:
            R_perturb_euler = Rotation.from_matrix(R).as_euler('xyz', degrees = True)
            R_pitch = R_perturb_euler[0]
        
            if random.random() < 0.9:
                R_perturb_pitch = R_pitch + np.random.normal(0, self.perturbation_level, 1)
            else:
                R_perturb_pitch = R_pitch
        
            R_perturb_euler[0] = R_perturb_pitch
            R_perturb = Rotation.from_euler('xyz', R_perturb_euler, degrees = True).as_matrix()
        
        elif self.pose_repr == 'use_identity':
            R_perturb = np.eye(3)

        '''
        R_euler = Rotation.from_matrix(R).as_euler('xyz', degrees = True)
        R_perturb_euler = R_euler + np.random.normal(0, self.perturbation_level, R_euler.shape)

        R = Rotation.from_euler('xyz', R_perturb_euler, degrees = True).as_matrix()
        '''

        K_left, dist_left = np.array(self.pose_gt["K_02"]).reshape(3,3), np.array(self.pose_gt["D_02"]).reshape(-1)
        K_right, dist_right = np.array(self.pose_gt["K_03"]).reshape(3,3), np.array(self.pose_gt["D_03"]).reshape(-1)
        
        dist_left_full = np.zeros((14,))
        dist_right_full = np.zeros((14,))
        # Images are already undistorted
        #dist_left_full[:len(dist_left)] = dist_left
        #dist_right_full[:len(dist_right)] = dist_right

        inputs['dist_coeff1'] = dist_left_full
        inputs['dist_coeff2'] = dist_right_full

        left_frame_path = os.path.join(left_frame_path)
        right_frame_path = os.path.join(right_frame_path)

        img_id = os.path.basename(left_frame_path).split('.')[0]

        inputs['img_id'] = f'{img_id}_{capture}'

        left_img = self.rgb_loader(left_frame_path)
        right_img = self.rgb_loader(right_frame_path)

        self.full_height, self.full_width = left_img.shape[0], left_img.shape[1]

        '''
        left_img, right_img, _ = calibrate_stereo_pair(left_img, right_img, 
                                                       K_left.astype(np.float64), K_right.astype(np.float64), 
                                                       dist_left_full.astype(np.float64), dist_right_full.astype(np.float64), 
                                                       R.astype(np.float64), t.astype(np.float64), 
                                                       self.full_height, self.full_width)
        '''
        '''
        left_img, right_img, _ = calibrate_stereo_pair(left_img, right_img, 
                                                       K_left.astype(np.float64), K_right.astype(np.float64), 
                                                       dist_left_full.astype(np.float64), dist_right_full.astype(np.float64), 
                                                       (R.T).astype(np.float64), ((-R.T)@t).astype(np.float64), 
                                                       self.full_height, self.full_width)
        '''

        if self.resize_orig_img:
            resize_width, resize_height = 1024, 768
            
            left_img = resize_img(left_img, resize_height, resize_width)
            right_img = resize_img(right_img, resize_height, resize_width)

            width_scale, height_scale = resize_width/self.full_width, resize_height/self.full_height
            scale_mat = np.array([[width_scale, 0, width_scale], [0, height_scale, height_scale], [0, 0, 1]])

            K_left = scale_mat*K_left
            K_right = scale_mat*K_right

            self.full_height, self.full_width = left_img.shape[0], left_img.shape[1]

        inputs['intrns_left'] = K_left
        inputs['intrns_right'] = K_right

        inputs['full_height'] = self.full_height
        inputs['full_width'] = self.full_width

        self.radial_mask = (3*gaussian_kernel((self.full_height, self.full_width), scale=4, invert=True)) + 1

        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K_left, dist_left_full, K_right, dist_right_full,
                                                          (self.full_width, self.full_height), R, t)
            
        xl1,yl1,xl2,yl2 = roi_left
        xr1,yr1,xr2,yr2 = roi_right        
        
        mask_left = np.zeros((self.full_height, self.full_width), dtype=bool)
        mask_right = np.zeros((self.full_height, self.full_width), dtype=bool)

        mask_left[yl1:yl2,xl1:xl2] = True
        mask_right[yr1:yr2,xr1:xr2] = True

        inputs['mask_left'] = mask_left
        inputs['mask_right'] = mask_right

        # undistort left and right image
        left_undist_img = cv2.undistort(left_img, K_left, dist_left_full)
        right_undist_img = cv2.undistort(right_img, K_right, dist_right_full)

        # Passing full images for rectification display in tensorboard
        inputs['full_left'] = self.to_tensor(left_undist_img.copy())
        inputs['full_right'] = self.to_tensor(right_undist_img.copy()) 

        if self.pose_repr in ['use_perturb', 'use_perturb_6D', 'use_perturb_quat', 'use_yaw_perturb', 'use_identity']:
            left_undist_img, right_undist_img, _ = calibrate_stereo_pair(left_undist_img, right_undist_img, K_left.astype(np.float64), K_right.astype(np.float64),\
                                                        dist_left, dist_right, R_perturb.astype(np.float64), t.astype(np.float64), self.full_height, self.full_width)

        if self.is_crop:
            left = crop_img(left_undist_img, self.height, self.width)
            right = crop_img(right_undist_img, self.height, self.width)
        else:
            left = resize_img(left_undist_img, self.height, self.width)
            right = resize_img(right_undist_img, self.height, self.width)

        inputs['target'] = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        inputs['reference'] = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x : x

        self.preprocess(inputs, color_aug)

        if self.pose_repr == 'use_3D':
            inputs['rotation'] = Rotation.from_matrix(R).as_rotvec()
        elif self.pose_repr == 'use_quat':
            inputs['rotation'] = Rotation.from_matrix(R).as_quat()
        elif self.pose_repr == 'use_6D':
            inputs['rotation'] = R[:2,:].reshape(-1) # taking first 2 rows of rotation matrix
        elif self.pose_repr == 'use_9D':
            inputs['rotation'] = R
        elif self.pose_repr == 'use_perturb':
            inputs['rotation'] = R
            inputs['rotation_perturb'] = R_perturb
        elif self.pose_repr == 'use_perturb_6D':
            inputs['rotation'] = R[:2,:].reshape(-1)
            inputs['rotation_perturb'] = R_perturb[:2, :].reshape(-1)
        elif self.pose_repr == 'use_perturb_quat':
            inputs['rotation'] = Rotation.from_matrix(R_perturb).as_quat()
        elif self.pose_repr == 'use_identity':
            inputs['rotation'] = R
            inputs['rotation_perturb'] = R_perturb
        elif self.pose_repr == 'use_yaw_perturb':
            inputs['rotation'] = R
            inputs['rotation_perturb'] = R_perturb
        elif self.pose_repr == 'use_yaw':
            rotation_euler = Rotation.from_matrix(R).as_euler('xyz', degrees = False)
            rotation_euler[0] = 0.0
            R_init = Rotation.from_euler('xyz', rotation_euler, degrees=False).as_matrix()
            inputs['rotation_perturb'] = R_init
            inputs['rotation'] = R
        elif self.pose_repr == 'use_rectify':
            inputs["R1"] = R1
            inputs["R2"] = R2
            inputs["P1"] = P1
            inputs["P2"] = P2
            inputs["Q"] = Q
            inputs['rotation'] = R[:2,:].reshape(-1) # taking first 2 rows of rotation matrix

        inputs['rotation'] = inputs['rotation'].astype(np.float32)
        inputs['axisangles'] = Rotation.from_matrix(R).as_rotvec(degrees=True).astype(np.float32)
        inputs['translation'] = t.astype(np.float32)

        inputs['radial_mask_full'] = self.to_tensor(self.radial_mask)

        '''
        # If you encounter
        # RuntimeError: Trying to resize storage that is not resizable
        # Uncomment loop and check for msmatches in shapes between items in 'inputs'
        
        for key, val in inputs.items():
            try:
                print(key, val.shape)
            except:
                print(key, val)

        print("---------------------")
        '''
        
        return inputs
    
    def __len__(self):
        return len(self.data)

    def parse_calib(self, extrinsic_calib, intrinsic_calib):
        self.pose_gt = {}

        extrinsics = transpose_dict(pd.read_feather(extrinsic_calib).to_dict('records'))
        intrinsics = transpose_dict(pd.read_feather(intrinsic_calib).to_dict('records'))

        self.pose_gt["K_02"], self.pose_gt["D_02"] = self.get_intrinsics(intrinsics['stereo_front_left'])
        self.pose_gt["K_03"], self.pose_gt["D_03"] = self.get_intrinsics(intrinsics['stereo_front_right'])

        self.pose_gt["R_03"], self.pose_gt["T_03"] = self.get_extrinsics(extrinsics['stereo_front_left'])
        self.pose_gt["R_02"], self.pose_gt["T_02"] = self.get_extrinsics(extrinsics['stereo_front_right'])


        self.get_relative_pose()
    
    def get_extrinsics(self, data_dict):
        rot_quat = np.array([data_dict['qx'], data_dict['qy'], data_dict['qz'], data_dict['qw']])
        trans_vec = np.array([data_dict['tx_m'], data_dict['ty_m'], data_dict['tz_m']])

        return Rotation.from_quat(rot_quat).as_matrix(), trans_vec

    def get_intrinsics(self, data_dict):
        I = np.eye(3)
        I[0, 0] = data_dict['fx_px']
        I[1, 1] = data_dict['fy_px']
        I[0, -1] = data_dict['cx_px']
        I[1, -1] = data_dict['cy_px']

        dist_coeffs = np.array([data_dict['k1'], data_dict['k2'], data_dict['k3']])

        return I, dist_coeffs

    def get_relative_pose(self):
        R_02 = self.pose_gt["R_02"].reshape(3, 3)
        t_02 = self.pose_gt["T_02"].reshape(3, 1)
        R_03 = self.pose_gt["R_03"].reshape(3, 3)
        t_03 = self.pose_gt["T_03"].reshape(3, 1)

        T_03 = np.eye(4)
        T_03[:3,:3] = R_03
        T_03[:3, 3:] = t_03

        T_02 = np.eye(4)
        T_02[:3, :3] = R_02
        T_02[:3, 3:] = t_02

        relative_pose = np.linalg.inv(T_02)@T_03

        self.pose_gt["rmat"], self.pose_gt["tvec"] = relative_pose[:3, :3], relative_pose[:3, 3:]

        