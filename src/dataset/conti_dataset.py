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
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from utility.data_utils import calibrate_stereo_pair


import torch
import torch.utils.data as data
from torchvision import transforms


class ContiDataset(data.Dataset):

    def __init__(self, data_dir, resize=[960, 544]):
        self.img0_list = sorted(glob.glob(os.path.join(data_dir, "image0/*.jpg")))
        self.img1_list = sorted(glob.glob(os.path.join(data_dir, "image1/*.jpg")))

        self.K0 = np.array([[7.20382032e+03, 0, 1.83550125e+03],
                          [0, 7.20987350e+03, 1.06925881e+03],
                          [0, 0, 1.]])

        self.K1 = np.array([[7136.3619475835485, 0, 1816.0883556332105],
                          [0, 7134.168540246251, 837.1057492474125],
                          [0, 0, 1.]])

        self.dist0 = np.zeros((14,))
        self.dist0[0], self.dist0[1] = -0.20436933,  0.35243867

        self.dist1 = np.zeros((14,))
        self.dist1[0], self.dist1[1] = -0.20793085661457059, 0.610783608258124

        self.resize = resize

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img0_list)

    def __getitem__(self, idx):
        inputs = {}
        img0_path = self.img0_list[idx]
        img1_path = self.img1_list[idx]

        frame_id = os.path.basename(img0_path).split('_')[0]+'_'+os.path.basename(img0_path).split('_')[1]
        img0 = cv2.imread(img0_path, -1)
        img1 = cv2.imread(img1_path, -1)

        img0 = cv2.undistort(img0, self.K0, self.dist0)
        img1 = cv2.undistort(img1, self.K1, self.dist1)

        inputs['left_full'] = self.to_tensor(img0.copy())
        inputs['right_full'] = self.to_tensor(img1.copy())

        full_height, full_width, _ = img0.shape

        if len(self.resize) != 0:
            img0_pert = cv2.resize(img0.copy(), (self.resize[0], self.resize[1]))
            img1_pert = cv2.resize(img1.copy(), (self.resize[0], self.resize[1]))
            img0 = cv2.resize(img0, (962, 540))
            img1 = cv2.resize(img1, (962, 540))
            img0_sift_kps, img1_sift_kps, sift_mconf = find_match(img0, img1)
            scaled_height, scaled_width, _ = img0.shape

        inputs['scale'] = full_width / scaled_width
        inputs['left_gray'] = self.to_tensor(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY))
        inputs['right_gray'] = self.to_tensor(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        inputs['left_img'] = self.to_tensor(img0_pert)
        inputs['right_img'] = self.to_tensor(img1_pert)
        inputs['intrns_left'] = self.to_tensor(self.K0)
        inputs['intrns_right'] = self.to_tensor(self.K1)
        inputs['dist_coeff1'] = self.dist0
        inputs['dist_coeff2'] = self.dist1
        inputs['frame_id'] = frame_id
        inputs['sift_keypoints0'] = img0_sift_kps
        inputs['sift_keypoints1'] = img1_sift_kps
        inputs['translation'] = np.array([[1, 0, 0]]).T
        inputs['sift_mconf'] = sift_mconf

        return inputs

def find_match(img1, img2):
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    kp1 = np.asarray([np.asarray((kpt.pt[0], kpt.pt[1])) for kpt in kp1])
    kp2 = np.asarray([np.asarray((kpt.pt[0], kpt.pt[1])) for kpt in kp2])

    kp1_model = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(desc1)
    dist, indices = kp1_model.kneighbors(desc2)
    ratio_test_1 = (dist[:, 0]/dist[:, 1]) < 0.75

    index_1 = indices[ratio_test_1, 0]

    pts1 = kp1[index_1].squeeze()
    pts2 = kp2[np.argwhere(ratio_test_1)].squeeze()

    sift_mconf = dist[ratio_test_1, 0]/dist[ratio_test_1, 1]

    return pts1, pts2, sift_mconf
