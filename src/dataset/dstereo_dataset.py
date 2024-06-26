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
from utility.data_utils import calibrate_stereo_pair, undistortrectify_cv2, unrectify_np

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

def get_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t

    return T

def invert_transformation(T):
    T_inv = np.eye(4)
    R, t = T[:3, :3], T[:3, 3:]

    R_inv = R.T
    t_inv = -(R_inv)@t

    T_inv[:3, :3], T_inv[:3, 3:] = R_inv, t_inv

    return T_inv

class DStereoDataset(data.Dataset):

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
        super(DStereoDataset, self).__init__()
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
        left_frame_path, right_frame_path, calib, capture = self.data[index]
        
        # read calibration data
        inputs['dataset'] = capture

        self.parse_calib(calib)
        
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

        R_noise = Rotation.from_euler('xyz', R_perturb_euler, degrees = True).as_matrix()
        '''

        K_left, dist_left = np.array(self.pose_gt["K_101"]).reshape(3,3), np.array(self.pose_gt["D_101"]).reshape(-1)
        K_right, dist_right = np.array(self.pose_gt["K_103"]).reshape(3,3), np.array(self.pose_gt["D_103"]).reshape(-1)
        
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

        if self.resize_orig_img:
            resize_width, resize_height = 832, 384
            
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
        inputs['og_left'] = self.to_tensor(left_undist_img.copy())
        inputs['og_right'] = self.to_tensor(right_undist_img.copy()) 

        R1_w = Rotation.from_matrix(np.eye(3)).as_euler('xyz', degrees = True) + np.random.normal(0, 0.5, (3,))
        R2_w = Rotation.from_matrix(np.eye(3)).as_euler('xyz', degrees = True) + np.random.normal(0, 0.5, (3,))
        T1_w = get_transformation_matrix(Rotation.from_euler('xyz', R1_w, degrees = True).as_matrix(), np.zeros((3, 1)))
        T2_w = get_transformation_matrix(Rotation.from_euler('xyz', R2_w, degrees = True).as_matrix(), t[:, None])

        #T21 = np.linalg.inv(T1_w)@T2_w
        #R, t = T21[:3, :3], T21[:3, 3:].reshape(3,)
      
        left_undist_img, right_undist_img = unrectify_np(left_undist_img, right_undist_img, 
                                           K_left, K_right,
                                           np.eye(4), invert_transformation(self.pose_gt["T_rect"]),
                                           np.matmul(K_left, T1_w[:3, :]), np.matmul(K_right, T2_w[:3, :]),
                                           self.full_width, self.full_height)
        
        
        R = self.opencv_pose_estimation(left_undist_img, right_undist_img, K_left, K_right)

        # Passing full images for rectification display in tensorboard
        inputs['full_left'] = self.to_tensor(left_undist_img.copy())
        inputs['full_right'] = self.to_tensor(right_undist_img.copy()) 

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
            inputs['rotation'] = R[:2,:].reshape(-1) #taking first 2 rows of rotation matrix
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

    def parse_calib(self, calib_path):
        self.pose_gt = {}
        calib = {}

        with open(calib_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(':')
                    calib[key.strip()] = [float(val) for val in value.strip().split(" ")]

        self.get_extrinsics(calib)

        self.get_intrinsics(calib)

        self.get_relative_pose()
    
    def get_extrinsics(self, data_dict):
        self.pose_gt["R_101"] = np.array(data_dict["R_101"]).reshape(3, 3)
        self.pose_gt["R_rect_101"] = np.array(data_dict["R_rect_101"]).reshape(3, 3)
        self.pose_gt["t_101"] = np.array(data_dict["T_101"]).reshape(3, 1)
        self.pose_gt["T_101"] = get_transformation_matrix(self.pose_gt["R_101"], self.pose_gt["t_101"])
        self.pose_gt["T_rect_101"] = get_transformation_matrix(self.pose_gt["R_rect_101"], self.pose_gt["t_101"])

        self.pose_gt["R_103"] = np.array(data_dict["R_103"]).reshape(3, 3)
        self.pose_gt["R_rect_103"] = np.array(data_dict["R_rect_103"]).reshape(3, 3)
        self.pose_gt["t_103"] = np.array(data_dict["T_103"]).reshape(3, 1)
        self.pose_gt["T_103"] = get_transformation_matrix(self.pose_gt["R_103"], self.pose_gt["t_103"])
        self.pose_gt["T_rect_103"] = get_transformation_matrix(self.pose_gt["R_rect_103"], self.pose_gt["t_103"])

    def get_intrinsics(self, data_dict):
        self.pose_gt["K_101"] = np.array(data_dict['K_101']).reshape(3, 3)
        self.pose_gt["K_103"] = np.array(data_dict['K_103']).reshape(3, 3)

        self.pose_gt["D_101"] = np.array(data_dict['D_101'])
        self.pose_gt["D_103"] = np.array(data_dict['D_103'])

        self.pose_gt["P_101"] = np.matmul(self.pose_gt["K_101"], self.pose_gt["T_101"][:3, :])
        self.pose_gt["P_103"] = np.matmul(self.pose_gt["K_103"], self.pose_gt["T_103"][:3, :])

        self.pose_gt["P_rect_101"] = np.array(data_dict["P_rect_101"]).reshape(3, 4)
        self.pose_gt["P_rect_103"] = np.array(data_dict["P_rect_103"]).reshape(3, 4)

        self.pose_gt["K_rect_101"] = self.pose_gt["P_rect_101"][:3, :3]
        self.pose_gt["K_rect_103"] = self.pose_gt["P_rect_103"][:3, :3]


    def get_relative_pose(self):
        R_02 = self.pose_gt["R_103"].reshape(3, 3)
        t_02 = self.pose_gt["t_103"].reshape(3, 1)
        R_03 = self.pose_gt["R_101"].reshape(3, 3)
        t_03 = self.pose_gt["t_101"].reshape(3, 1)

        T_03 = np.eye(4)
        T_03[:3,:3] = R_03
        T_03[:3, 3:] = t_03

        R_02_inv = R_02.T
        t_02_inv = (-R_02_inv)@t_02

        T_02_inv = np.eye(4)
        T_02_inv[:3, :3] = R_02_inv
        T_02_inv[:3, 3:] = t_02_inv

        relative_pose = T_02_inv@T_03

        self.pose_gt["T_rect"] = relative_pose

        self.pose_gt["rmat"], self.pose_gt["tvec"] = relative_pose[:3, :3], relative_pose[:3, 3:]
    
    def opencv_pose_estimation(self, img1, img2, K1, K2):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        
        pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
        pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                       method=cv2.RANSAC, prob=0.999, threshold=3e-3)
        try:
            points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm[mask==1], pts_r_norm[mask==1])
        except:
            print(pts_l_norm.shape, pts_r_norm.shape)
            R_est = np.eye(3)
            t_est = np.zeros((3))

        #print("T_opencv", get_transformation_matrix(R_est, t_est))
        #print("T_calib", self.pose_gt["T_rect"])

        return R_est.reshape(3, 3)




    