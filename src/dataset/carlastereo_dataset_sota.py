from __future__ import absolute_import, division, print_function

import os
import random
from turtle import left
from matplotlib import use
import numpy as np
import csv
import json
import copy
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

class CarlaStereoPoseDatasetSOTA(data.Dataset):

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
        super(CarlaStereoPoseDatasetSOTA, self).__init__()
        self.pose_gt_dir = pose_gt_dir
        self.frame_dir = frames_dir
        self.datafile_path = datafile_path
        self.pose_repr = pose_repr

        self.height = height
        self.width = width

        self.is_crop = is_crop

        self.resize_orig_img = resize_orig_img
        
        self.perturbation_level = perturbation_level

        with open(os.path.join(frames_dir,"config_args.json"),'r') as fh:
            config_data = json.load(fh)

        self.fov = config_data["fov"][0]
        self.full_height, self.full_width = config_data["dim"][0], config_data["dim"][1]

        self.is_train = is_train

        self.rgb_loader = rgb_loader

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
        
        target_cam_num, ref_cam_num, is_left_pertb, is_right_pertb, left_frame_path, right_frame_path = self.data[index]

        assert os.path.basename(left_frame_path) == os.path.basename(right_frame_path), "Left and Right image names are not same"

        is_left_pertb, is_right_pertb = is_left_pertb == "True", is_right_pertb == "True"
        img_id = os.path.basename(left_frame_path).split('.')[0]

        inputs['img_id'] = f'{img_id}_{target_cam_num}_{ref_cam_num}'

        sota = {'dnet': "/nas/EOS/users/anush/DirectionNet/checkpoints/pred_poses", #"/nas/EOS/users/anush/DirectionNet/final_model/pred_poses", 
                'vit': "/nas/EOS/users/anush/sota/3dv/output/carla/pred_poses",
                'rpnet': "/nas/EOS/users/anush/sota/RPNet/rpnet/carla_train_rpnetplus/pred_rotation",
                'fepe': "/nas/EOS/users/anush/sota/DeepFEPE/logs/carla_eval/pred_poses"}

        for sota_name, sota_folder in sota.items():
            sota_pose = np.load(os.path.join(sota_folder, inputs['img_id']+'.npy'))
            inputs['R_{}'.format(sota_name)] = Rotation.from_matrix(sota_pose[:3, :3]).as_matrix()
            inputs['axisangles_{}'.format(sota_name)] = Rotation.from_matrix(sota_pose[:3, :3]).as_rotvec(degrees=True).astype(np.float32)

        # Calculate rectification parameters
        pose_json_fpath = os.path.join(self.pose_gt_dir,f'{img_id}.json')
        with open(pose_json_fpath,'r') as fh:
            calib_data = json.load(fh)

        if is_left_pertb:
            T_carla_cam0 = np.array(calib_data[f'rgb_p_cam{target_cam_num}_matrix'])
        else:
            T_carla_cam0 = np.array(calib_data[f'rgb_cam{target_cam_num}_matrix'])

        if is_right_pertb:
            T_carla_cam1 = np.array(calib_data[f'rgb_p_cam{ref_cam_num}_matrix'])
        else:
            T_carla_cam1 = np.array(calib_data[f'rgb_cam{ref_cam_num}_matrix'])

        P = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])

        T_opencv_cam0 = P @ T_carla_cam0 @ np.linalg.inv(P)
        T_opencv_cam1 = P @ T_carla_cam1 @ np.linalg.inv(P)

        T_opencv_cam0_inv = np.linalg.inv(T_opencv_cam0)
        T_opencv_cam1_inv = np.linalg.inv(T_opencv_cam1)

        R_opencv_cam0, t_opencv_cam0 = T_opencv_cam0_inv[0:3,0:3], T_opencv_cam0_inv[0:3,3]
        R_opencv_cam1, t_opencv_cam1 = T_opencv_cam1_inv[0:3,0:3], T_opencv_cam1_inv[0:3,3]

        R = (R_opencv_cam1 @ R_opencv_cam0.T).reshape(3,3)
        t =  (t_opencv_cam1 - R @ t_opencv_cam0).reshape(3,)

        if self.pose_repr in ['use_perturb', 'use_perturb_6D', 'use_perturb_quat']:
            R_euler = Rotation.from_matrix(R).as_euler('xyz', degrees = True)
            if random.random() < 0.9:
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

        distCoeff1 = np.zeros((14,))
        distCoeff2 = np.zeros((14,))

        inputs['dist_coeff1'] = distCoeff1
        inputs['dist_coeff2'] = distCoeff2

        left_img = self.rgb_loader(os.path.join(self.frame_dir, left_frame_path))
        right_img = self.rgb_loader(os.path.join(self.frame_dir, right_frame_path))

        self.full_height, self.full_width = left_img.shape[0], left_img.shape[1]

        focal_length = self.full_width / (2.0 * np.tan(self.fov * np.pi / 360.0))

        self.K = np.array([[focal_length, 0, self.full_width/2],
                          [0, focal_length, self.full_height/2],
                          [0, 0, 1]])

        if self.resize_orig_img:
            resize_width, resize_height = 1024, 512
            
            left_img = resize_img(left_img, resize_height, resize_width)
            right_img = resize_img(right_img, resize_height, resize_width)

            width_scale, height_scale = resize_width/self.full_width, resize_height/self.full_height
            scale_mat = np.array([[width_scale, 0, width_scale], [0, height_scale, height_scale], [0, 0, 1]])

            self.K = scale_mat*self.K
            self.full_height, self.full_width = left_img.shape[0], left_img.shape[1]
        

        inputs["intrns_left"], inputs["intrns_right"] = self.K, self.K

        inputs['full_height'] = self.full_height
        inputs['full_width'] = self.full_width

        self.radial_mask = (3*gaussian_kernel((self.full_height, self.full_width), scale=4, invert=True)) + 1
        
        _,_,_,_,_,roi_left, roi_right = cv2.stereoRectify(self.K,distCoeff1,self.K,distCoeff2,
                                                          (self.full_width,self.full_height),R,t)
            
        xl1,yl1,xl2,yl2 = roi_left
        xr1,yr1,xr2,yr2 = roi_right        
        
        
        mask_left = np.zeros((self.full_height, self.full_width),dtype=bool)
        mask_right = np.zeros((self.full_height,self.full_width),dtype=bool)

        mask_left[yl1:yl2,xl1:xl2] = True
        mask_right[yr1:yr2,xr1:xr2] = True

        inputs['mask_left'] = mask_left
        inputs['mask_right'] = mask_right
        
        # Passing full images for rectification display in tensorboard
        inputs['full_left'] = self.to_tensor(left_img.copy())
        inputs['full_right'] = self.to_tensor(right_img.copy())

        if self.pose_repr in ['use_perturb', 'use_perturb_6D', 'use_perturb_quat', 'use_yaw_perturb', 'use_identity']:
            left_img, right_img = calibrate_stereo_pair(left_img, right_img, self.K.astype(np.float64), self.K.astype(np.float64),\
                                                      np.zeros((8,)), np.zeros((8,)), R_perturb.astype(np.float64), t.astype(np.float64), self.full_height, self.full_width)

        # inputs['full_left_rectf'] = self.to_tensor(left_img_rect)
        # inputs['full_right_rectf'] = self.to_tensor(right_img_rect)

        # take center crop
        if self.is_crop:
            left = crop_img(left_img, self.height, self.width)
            right = crop_img(right_img, self.height, self.width)
        else:
            left = resize_img(left_img, self.height, self.width)
            right = resize_img(right_img, self.height, self.width)

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
