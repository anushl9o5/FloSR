from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import pdb
from .unet_feat_extractor import unet
from matplotlib import pyplot as plt
from utility.data_utils import calibrate_stereo_pair, calibrate_stereo_pair_torch, gram_schmidt, svd_orthogonalize

class HSMNet(nn.Module):
    def __init__(self, maxdisp=192, clean=-1, level=1, n_out=[9, 3], use_tanh=False, use_linear4=False, dataset="algolux", transformer=False):
        super(HSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.clean = clean
        self.feature_extraction = unet()
        self.level = level
        self.use_tanh = use_tanh
        self.use_linear4 = use_linear4
        self.n_out = n_out

        self.dataset_outs = {"torc": [23040, 5760, 720, 360], 
                             "kitti": [23040, 5760, 720, 504], 
                             "dstereo": [23040, 5760, 720, 234], 
                             "carla": [23040, 5760, 720, 360],
                             "carla2": [23040, 5760, 720, 360], 
                             "argo2": [23040, 5760, 720, 576],
                             "flyingthings": [23040, 5760, 720, 336],
                             "sintel": [23040, 5760, 720, 384],
                             "hd1k": [23040, 5760, 720, 480],
                             }

        if not transformer:
            self.linear3 = nn.Sequential(
                nn.Linear(in_features=self.dataset_outs[dataset][0], out_features=n_out[0]), 
                nn.Tanh(),
                )

            self.linear4 = nn.Sequential(
                nn.Linear(in_features=self.dataset_outs[dataset][1], out_features=n_out[0]), 
                nn.Tanh(),
                )

            self.linear5 = nn.Sequential(
                nn.Linear(in_features=self.dataset_outs[dataset][2], out_features=n_out[0]), 
                nn.Tanh(),
                )

            self.linear6 = nn.Sequential(
                nn.Linear(in_features=self.dataset_outs[dataset][3], out_features=n_out[0]), 
                nn.Tanh(),
                )

            '''
            self.conv3d_3 = nn.Conv3d(16, 1, 3, padding='same')
            self.conv3d_3_vert = nn.Conv3d(16, 1, 3, padding='same')
            self.conv3d_4 = nn.Conv3d(16, 1, 3, padding='same')
            self.conv3d_5 = nn.Conv3d(16, 1, 3, padding='same')
            self.conv3d_6 = nn.Conv3d(32, 1, (2, 3, 3), padding='same')
            '''

            # block 4
            self.decoder6 = decoderBlock(6,32,32,up=False, pool=True)
            if self.level > 2:
                self.decoder5 = decoderBlock(6,16,16,up=False, pool=True)
            else:
                self.decoder5 = decoderBlock(6,16,16,up=False, pool=True)
                if self.level > 1:
                    self.decoder4 = decoderBlock(6,16,16, up=False, pool=True)
                else:
                    self.decoder4 = decoderBlock(6,16,16, up=False, pool=True)
                    self.decoder3 = decoderBlock(5,16,16, stride=(2,1,1),up=False, nstride=1)

            self.decoders = [self.decoder6, self.decoder5, self.decoder4, self.decoder3]
            #self.conv3ds = [self.conv3d_6, self.conv3d_5, self.conv3d_4, self.conv3d_3]
            self.linears = [self.linear6, self.linear5, self.linear4, self.linear3]

    def feature_vol(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        '''
        diff feature volume
        '''

        width = refimg_fea.shape[-1]
        cost = Variable(torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp, refimg_fea.size()[2], refimg_fea.size()[3]).fill_(0.))

        for i in range(min(maxdisp, width)):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]

            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:] = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i] = torch.abs(featb-feata)

        cost = cost.contiguous()

        return cost

    def get_rotation(self, feat, linear, nsample):
        feat_flat = feat.view(nsample, -1)
        R = linear(feat_flat)
        return R

    def rectify_feats(self, feat_l, feat_r, rot, translation, intrinsics_left, intrinsics_right, dist1, dist2):
        b, c, h, w = feat_l.shape
        #rotation = gram_schmidt(rot[:, :3], rot[:, 3:])
        rotation = svd_orthogonalize(rot.reshape(-1, 3, 3))
        feat_l_rectf, feat_r_rectf, _ = \
            calibrate_stereo_pair_torch(
                feat_l, feat_r, intrinsics_left.double(), intrinsics_right.double(), dist1.double(), dist2.double(), rotation, translation.double(),
                h, w)

        return feat_l_rectf, feat_r_rectf

    def forward(self, left, right, translation, intrinsics_left, intrinsics_right, dist1, dist2, full_height, full_width):
        nsample = left.shape[0]
        R = torch.eye(3).unsqueeze(0).to(left.device).double()
        R = R.repeat(nsample, 1, 1).view(nsample, -1)
        R = R[:, :self.n_out[0]]
        conv4, conv3, conv2, conv1 = self.feature_extraction(torch.cat([left, right],0))
        conv40,conv30,conv20,conv10 = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        conv41,conv31,conv21,conv11 = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]

        conv_l = [conv40,conv30,conv20,conv10]
        conv_r = [conv41,conv31,conv21,conv11]
        levels = [64, 32, 16, 8]
        pools = [False, True, True, True]
        pred_rotations = []
        vert_vols = None

        for i in range(len(conv_l[0:1])):
            _, _, h, w = conv_l[i].shape
            width_scale, height_scale = w/full_width, h/full_height
            scale_mat = torch.tensor([[[width_scale, 0, width_scale], [0, height_scale, height_scale], [0, 0, 1]]])
            scale_mat = scale_mat.to(intrinsics_left.device)
            
            conv_l_rectf, conv_r_rectf = self.rectify_feats(conv_l[i], conv_r[i], R, translation, intrinsics_left*scale_mat, intrinsics_right*scale_mat, dist1, dist2)
            feat_vol = self.feature_vol(conv_l_rectf, conv_r_rectf, self.maxdisp//levels[i]) #self.get_cost_vol(conv_l_rectf, conv_r_rectf, hor_conv3d=self.conv3ds[i], pool=False, level=levels[i])
            _, rectf_vol = self.decoders[i](feat_vol)

            #disp_lr = self.wta_disparity(rectf_vol)
            #conv_l_warp = self.interpolate_left_to_right(conv_l_rectf, conv_r_rectf, disp_lr)
            #vert_vols = self.feature_vol(conv_l_rectf.permute(0, 1, 3, 2), conv_r_rectf.permute(0, 1, 3, 2), self.maxdisp//self.maxdisp)

            R_diff = self.get_rotation(F.softmax(rectf_vol, 1), self.linears[i], nsample)
            
            R = R + R_diff #torch.bmm(svd_orthogonalize(R_diff.reshape(-1, 3, 3)).permute(0, 2, 1).double(), svd_orthogonalize(R.reshape(-1, 3, 3))).reshape(-1, 9)
            
            pred_rotations.append(R)
        
        final = R

        return final, {"conv10": conv40, "conv11": conv41}, pred_rotations
    

    def wta_disparity(self, cost_volume):
        """
        Compute the winner-takes-all (WTA) disparity map from the cost volume.
        
        Args:
            cost_volume (torch.Tensor): cost volume tensor with shape (B, max_disp, H, W).
            
        Returns:
            torch.Tensor: disparity map tensor with shape (B, 1, H, W).
        """
        if len(cost_volume.shape) != 4:
            cost_volume = cost_volume.unsqueeze(0)
        
        disparity_map = torch.argmin(cost_volume, dim=1, keepdim=True)
        
        return disparity_map.float()

    def interpolate_left_to_right(self, left_image, right_image, disparity):
        batch_size, channels, height, width = left_image.shape
        device = left_image.device

        # Create grid for right image coordinate space
        _, _, disp_height, disp_width = disparity.shape
        y, x = torch.meshgrid(torch.arange(disp_height), torch.arange(disp_width))
        x = x.to(device) / (0.5*disp_width)
        y = y.to(device) / (0.5*disp_height)
        y = y - 1

        x_offset = disparity.clone() / (0.5*disp_width)  # horizontal disparity
        x_new = x.repeat(batch_size, 1, 1, 1).float() + x_offset
        x_new = x_new - 1

        # Interpolate left image to right image coordinate space
        x_new = torch.clamp(x_new, -1, 1)
        y_new = y.repeat(batch_size, 1, 1, 1)
        grid = torch.cat([x_new, y_new], dim=1).permute(0, 2, 3, 1)

        interpolated = torch.nn.functional.grid_sample(left_image, grid, mode='bilinear', padding_mode='border')
        
        # Mask out invalid regions
        
        mask = (x_new > 1) | (x_new < -1)
        mask = mask.repeat(1, channels, 1, 1)
        interpolated[mask] = 0
        

        return interpolated