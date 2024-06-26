from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2

import numpy as np
from typing import Tuple
import torch
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix, convert_points_to_homogeneous
from pytorch3d.transforms import matrix_to_euler_angles,axis_angle_to_matrix
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
from typing import Optional
import matplotlib.cm as cm
from kornia.geometry.calibration.distort import distort_points, tilt_projection
from kornia.geometry import find_fundamental, essential_from_fundamental, ransac
from kornia.geometry.epipolar import motion_from_essential_choose_solution
from sklearn.neighbors import NearestNeighbors
import os
import random
import json
from torch import nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import argparse

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def degrees_to_radians(degree):
  """Convert degrees to radians."""
  return math.pi * degree / 180.0


def radians_to_degrees(radians):
  """Convert radians to degrees."""
  return 180.0 * radians / math.pi

# Taken from
# https://github.com/google-research/google-research/blob/956727811ae2c72e9aebea254b2c8c79b0ad6251/direction_net/util.py#L186
def gram_schmidt(x,y):
  """
  Convert 6D representation to SO(3) using a partial Gram-Cchmidt process.
    Args:
        x: [BATCH, 1, 3] 1x3 matrices.
        y: [BATCH, 1, 3] 1x3 matrices. 
    Returns:
        [BATCH, 3, 3] SO(3) rotation matrices.
  """
  x_norm = torch.nn.functional.normalize(x, p = 2, dim = -1)
  z  = torch.cross(x_norm, y) 
  z_norm = torch.nn.functional.normalize(z, p = 2, dim = -1)
  y = torch.cross(z_norm, x_norm)
  r  = torch.stack([x_norm, y, z_norm], 1)
  return r


# Taken from
# https://github.com/arthurchen0518/DirectionNet/blob/2ed8478ff0a3d929542d0a5288da18a18907242a/util.py
def svd_orthogonalize(m):
  """Convert 9D representation to SO(3) using SVD orthogonalization.
  Args:
    m: [BATCH, 3, 3] 3x3 matrices.
  Returns:
    [BATCH, 3, 3] SO(3) rotation matrices.
  """
  m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), 1, 2)
  u, _, v = torch.svd(m_transpose)
  det = torch.linalg.det(torch.matmul(v, torch.transpose(u, 1, 2)))
  # Check orientation reflection.
  r = torch.matmul(
      torch.concat([v[:, :, :-1], v[:, :, -1:] * torch.reshape(det, [-1, 1, 1])], 2),
      torch.transpose(u, 1, 2))

  return r

def calibrate_stereo_pair(img_left, img_right, K_left, K_right, dist_left, dist_right, R, t, height, width):
    flags=cv2.CALIB_ZERO_DISPARITY
    
    R1,R2,P1,P2,Q,roi_left, roi_right = cv2.stereoRectify(K_left,dist_left,K_right,dist_right,
                                                          (width,height),R,t,flags,alpha=-1)
    maplx,maply = cv2.initUndistortRectifyMap(K_left,dist_left,R1,P1,(width,height),cv2.CV_32FC1)
    maprx,mapry = cv2.initUndistortRectifyMap(K_right,dist_right,R2,P2,(width,height),cv2.CV_32FC1)

    img_left_rect = cv2.remap(img_left, maplx, maply, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, maprx, mapry, cv2.INTER_LINEAR)

    return img_left_rect, img_right_rect, {"R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q}

def undistortrectify_cv2(img_left, img_right, K_left, K_right, dist_left, dist_right, R1, R2, P1, P2, height, width):
    maplx,maply = cv2.initUndistortRectifyMap(K_left,
                                              dist_left,
                                              R1,
                                              newCameraMatrix=K_left,
                                              size=(width,height),
                                              m1type=cv2.CV_32FC1)
    
    maprx,mapry = cv2.initUndistortRectifyMap(K_right,
                                              dist_right,
                                              R2,
                                              newCameraMatrix=K_right,
                                              size=(width,height),
                                              m1type=cv2.CV_32FC1)

    img_left_rect = cv2.remap(img_left, maplx, maply, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, maprx, mapry, cv2.INTER_LINEAR)

    return img_left_rect, img_right_rect

def draw_epipolar_lines(image1,image2, num_lines = 100,line_color=(0,255,0), line_thickness=2, text = None):
    assert image1.shape == image2.shape, "two image dimensions are of different dimensions"
    h,w,_ = image1.shape
    comb = np.concatenate((image1,image2),axis=1) # concatenating along width dimesnion
    delta = h//num_lines
    left_indxs = np.arange(0, h, delta)
    for indx in left_indxs:
        cv2.line(comb, (0,indx),(2*w,indx),line_color,line_thickness)
    
    if text is not None:
        cv2.putText(comb,text,(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),3)
    return comb


def get_rectangles_like_cv2(cameraMatrix: torch.Tensor, distCoeff: torch.Tensor, new_cameraMatrix: torch.Tensor, R: Optional[torch.Tensor], dims, batch):

    N = 9
    height, width = dims[0], dims[1]

    xx = torch.linspace(0, width, N)
    yy = torch.linspace(0, height, N)
    
    x, y = torch.meshgrid(xx, yy, indexing='ij')
    x, y = torch.atleast_2d(x), torch.atleast_2d(y)

    pts = torch.zeros(batch, y.shape[0], x.shape[0], 2, device=cameraMatrix.device).double()
    pts[:, ...] = torch.dstack((y, x))
    pts = torch.reshape(pts, (batch, N*N, 2))

    pts_undistort = undistort_points_like_cv2(pts, cameraMatrix, distCoeff, new_cameraMatrix)
    
    inner = torch.zeros(batch, 4)
    outer = torch.zeros(batch, 4)

    for b in range(batch):
        iX0 = torch.Tensor([-float('inf')]).to(cameraMatrix.device)
        iX1 = torch.Tensor([float('inf')]).to(cameraMatrix.device)
        iY0 = torch.Tensor([-float('inf')]).to(cameraMatrix.device)
        iY1 = torch.Tensor([float('inf')]).to(cameraMatrix.device)
        oX0 = torch.Tensor([float('inf')]).to(cameraMatrix.device)
        oX1 = torch.Tensor([-float('inf')]).to(cameraMatrix.device)
        oY0 = torch.Tensor([float('inf')]).to(cameraMatrix.device)
        oY1 = torch.Tensor([-float('inf')]).to(cameraMatrix.device)
        k = 0
        
        for j in range(N):
            for i in range(N):
                p = pts_undistort[b, k]

                oX0 = torch.minimum(oX0, p[1])
                oX1 = torch.maximum(oX1, p[1])
                oY0 = torch.minimum(oY0, p[0])
                oY1 = torch.maximum(oY1, p[0])

                if i == 0:
                    iX0 = torch.maximum(iX0, p[1])
                if i == N-1:
                    iX1 = torch.minimum(iX1, p[1])
                if j == 0:
                    iY0 = torch.maximum(iY0, p[0])
                if j == N-1:
                    iY1 = torch.minimum(iY1, p[0])

                k+=1

        inner[b, ...] = torch.Tensor([iX0, iY0, iX1-iX0, iY1-iY0])
        outer[b, ...] = torch.Tensor([oX0, oY0, oX1-oX0, oY1-oY0])

    return inner, outer

def undistort_points_like_cv2(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor] = None, num_iters: int = 5) -> torch.Tensor:
    r"""Compensate for lens distortion a set of 2D image points.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.
        num_iters: Number of undistortion iterations. Default: 5.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.
    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f'points shape is invalid. Got {points.shape}.')

    if K.shape[-2:] != (3, 3):
        raise ValueError(f'K matrix shape is invalid. Got {K.shape}.')

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f'new_K matrix shape is invalid. Got {new_K.shape}.')

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y: torch.Tensor = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        x, y = transform_points(inv_tilt, torch.stack([x, y], dim=-1)).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3
        )
        deltaX = (
            2 * dist[..., 2:3] * x * y
            + dist[..., 3:4] * (r2 + 2 * x * x)
            + dist[..., 8:9] * r2
            + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
            dist[..., 2:3] * (r2 + 2 * y * y)
            + 2 * dist[..., 3:4] * x * y
            + dist[..., 10:11] * r2
            + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

        # print(x.shape)
        # print(y.shape)

    '''
      This is done to match OpenCV implementation of undistortPoints
    '''

    # Convert points from normalized camera coordinates to pixel coordinates
    # new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    # new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    # new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    # new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)
    # x = new_fx * x + new_cx
    # y = new_fy * y + new_cy

    return torch.stack([x, y], -1)

def stereorectify_torch(cameraMatrix1:torch.Tensor, distCoeff1: torch.Tensor, cameraMatrix2:torch.Tensor,
                        distCoeff2: torch.Tensor,img_size:Tuple[np.uint16,np.uint16],
                        R:torch.Tensor, T:torch.Tensor, new_image_size: Optional[Tuple[np.uint16,np.uint16]] = (0,0)):

  b, _, _ = cameraMatrix1.shape

  uu = torch.zeros(b,3,1, device=cameraMatrix1.device)

  nx, ny = img_size

  if len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3: # R is in Rotation Matrix format
    om = matrix_to_euler_angles(R, convention='XYZ')                       # get vector rotation format
  else:
    om = R

  om *= -0.5 # get average rotation

  r_r = angle_axis_to_rotation_matrix(om) # rotate cameras to same orientation by averaging
  t = r_r @ T.clone()

#   idx = torch.where(torch.abs(t[:,0,0]) > torch.abs(t[:,1,0]),0,1).type(torch.long)
  idx = torch.zeros(b,device=cameraMatrix1.device,dtype=torch.long)

  c = torch.gather(t.clone().squeeze(2), dim=1, index = idx.reshape(-1,1))
  nt = torch.linalg.norm(t,ord=2,dim=1)
  uu = torch.scatter(uu.squeeze(2),dim=1, index = idx.reshape(-1,1), src=torch.where(c > 0, 1.0, -1.0)).unsqueeze(2).double()

  assert torch.all(nt>0), 'L2 norm should be positive'

  # Calculate global Z rotation
  ww_ = torch.cross(t.squeeze(2),uu.squeeze(2)).unsqueeze(2)
  nw = torch.linalg.norm(ww_,ord=2,dim=1)
  pos_nw_indx = torch.where(nw > 0.0)[0]
  
  ww = ww_.clone() # cloning to avoid in-place operation

  ww[pos_nw_indx]  *= (torch.acos(torch.abs(c[pos_nw_indx])/nt[pos_nw_indx])/nw[pos_nw_indx]).unsqueeze(1)
  wR = axis_angle_to_matrix(ww.squeeze(2))

  # apply to both views 
  R1 = wR @ r_r.transpose(1,2)
  R2 = wR @ r_r
  t = R2 @ T

  # calculate projection/camera matrices
  # these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
  new_image_size = new_image_size if new_image_size[0] * new_image_size[1] else (nx,ny)
  ratio_x = (new_image_size[0] / nx) * 0.5
  ratio_y = (new_image_size[1] / ny) * 0.5
  ratio = ratio_y * torch.ones_like(idx)
  ratio[torch.where(idx==1)[0]] = ratio_x

  fc_new = (torch.gather(torch.diagonal(cameraMatrix1,dim1=-2,dim2=-1), 1, torch.bitwise_xor(idx,torch.ones_like(idx)).view(-1,1)).squeeze(1) +
          torch.gather(torch.diagonal(cameraMatrix2,dim1=-2,dim2=-1), 1, torch.bitwise_xor(idx,torch.ones_like(idx)).view(-1,1)).squeeze(1)) * ratio

  cc_new_x = torch.zeros(b,2,device=cameraMatrix1.device)
  cc_new_y = torch.zeros(b,2,device=cameraMatrix1.device)

  for k in range(2): # k = 0 is left camera and k = 1 is left camera
    A = cameraMatrix1 if k == 0 else cameraMatrix2
    Dk = distCoeff1 if k == 0 else distCoeff2 

    pts = torch.Tensor([[0,      0    ],
                        [nx - 1, 0    ],
                        [0,      ny -1],
                        [nx - 1, ny -1]]).repeat(b,1,1).to(cameraMatrix1.device)

    pts = undistort_points_like_cv2(pts,A,Dk)
    pts_3 = convert_points_to_homogeneous(pts)
    
    # Change camera matrix to have cc=[0,0] and fc = fc_new
    A_tmp = torch.eye(3,device=cameraMatrix1.device).double().repeat(b,1,1)
    A_tmp[:,0,0] = fc_new
    A_tmp[:,1,1] = fc_new

    # project points to image with A-tmp as intrinsics
    R_tmp = R1 if k == 0 else R2
    pts_tmp = A_tmp @ R_tmp @ pts_3.transpose(1,2)
    pts = pts_tmp/pts_tmp[:,2,:].reshape(-1,1,4)
    avg = torch.mean(pts.clone(),dim=2)[:,:2]

    cc_new_x[:,k] = (nx-1)/2.0 - avg[:,0]
    cc_new_y[:,k] = (ny-1)/2.0 - avg[:,1]

  # For simplicity, set the principal points for both cameras to be the average
  # of the two principal points (either one of or both x- and y- coordinates)
  # This is CALIB_ZERO_DISPARITY flag for opencv
  cc_mean_x = torch.mean(cc_new_x,dim=1)
  cc_new_x[:,0] = cc_mean_x
  cc_new_x[:,1] = cc_mean_x

  cc_mean_y = torch.mean(cc_new_y,dim=1)
  cc_new_y[:,0] = cc_mean_y
  cc_new_y[:,1] = cc_mean_y

  # constructing projection matrix
  P1 = torch.zeros(b,3,4,device=cameraMatrix1.device).double()
  P2 = torch.zeros(b,3,4,device=cameraMatrix1.device).double()

  P1[:,0,0] = fc_new
  P1[:,1,1] = fc_new
  P1[:,0,2] = cc_new_x[:,0]
  P1[:,1,2] = cc_new_y[:,0]
  P1[:,2,2] = 1.0

  P2[:,0,0] = fc_new
  P2[:,1,1] = fc_new
  P2[:,0,2] = cc_new_x[:,1]
  P2[:,1,2] = cc_new_y[:,1]
  bf = torch.gather(t.squeeze(2), dim=1, index = idx.reshape(-1,1)).squeeze(1) * fc_new

  #inner1, outer1 = get_rectangles_like_cv2(cameraMatrix1, distCoeff1, P1[:, :, :3], None, img_size, b)
  #inner2, outer2 = get_rectangles_like_cv2(cameraMatrix2, distCoeff2, P2[:, :, :3], None, img_size, b)
  #print(inner1, outer1, inner2, outer2)

  # setting P2[idx][3] = bf 
  idx_zero = torch.where(idx==0)[0]
  idx_one = torch.where(idx==1)[0]
  P2[idx_zero,0,3] = bf[idx_zero]
  P2[idx_one,1,3] = bf[idx_one]

  # Constructing Q matrix
  matQ = torch.eye(4).repeat(b,1,1).double()
  matQ[:,0,3] = -cc_new_x[:,0]
  matQ[:,1,3] = -cc_new_y[:,0]
  matQ[:,2,3] = fc_new
  matQ[:,3,2] = -1.0/torch.gather(t.squeeze(2), dim=1, index = idx.reshape(-1,1)).squeeze(1)

  return R1, R2, P1, P2, matQ

def undistortrectify(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, R: torch.Tensor, new_K: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
    
    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f'K matrix shape is invalid. Got {K.shape}.')
    
    if R.shape[-2:] != (3, 3):
        print(R)
        raise ValueError(f'R matrix shape is invalid. Got {R.shape}.')

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f'new_K matrix shape is invalid. Got {new_K.shape}.')

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f'Invalid number of distortion coefficients. Got {dist.shape[-1]}')

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])


    if not image.is_floating_point():
        raise ValueError(f'Invalid input image data type. Input should be float. Got {image.dtype}.')

    if image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]:
        # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
        # allowed to avoid a breaking change.
        if not all((image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())):
            raise ValueError(
                f'Input shape is invalid. Input batch dimensions should match. '
                f'Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}.'
            )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(rows, cols, False, image.device, image.dtype)
    points = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Convert 2D points from pixels to normalized camera coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (points[..., 0] - new_cx) / new_fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y: torch.Tensor = (points[..., 1] - new_cy) / new_fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Applying R^-1 * [x,y,1]^T to get [X,Y,W]. Then x = X/W, y = Y/W
    R_inv = torch.inverse(R)
    p = torch.stack([x,y,torch.ones_like(x)],dim=1).type_as(R)
    XYW = R_inv @ p
    xy = XYW/XYW[:,2,:].reshape(-1,1,XYW.shape[-1])
    x: torch.Tensor = xy[:,0,:].float() 
    y: torch.Tensor = xy[:,1,:].float()

    # Distort points
    r2 = x * x + y * y

    rad_poly = (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3) / (
        1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3
    )
    xd = (
        x * rad_poly
        + 2 * dist[..., 2:3] * x * y
        + dist[..., 3:4] * (r2 + 2 * x * x)
        + dist[..., 8:9] * r2
        + dist[..., 9:10] * r2 * r2
    )
    yd = (
        y * rad_poly
        + dist[..., 2:3] * (r2 + 2 * y * y)
        + 2 * dist[..., 3:4] * x * y
        + dist[..., 10:11] * r2
        + dist[..., 11:12] * r2 * r2
    )

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        tilt = tilt_projection(dist[..., 12], dist[..., 13])

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        points_untilt = torch.stack([xd, yd, torch.ones_like(xd)], -1) @ tilt.transpose(-2, -1)
        xd = points_untilt[..., 0] / points_untilt[..., 2]
        yd = points_untilt[..., 1] / points_untilt[..., 2]

    # Convert points from normalized camera coordinates to pixel coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    x = fx * xd + cx
    y = fy * yd + cy

    ptsd = torch.stack([x, y], -1)

    mapx: torch.Tensor = ptsd[..., 0].reshape(B, rows, cols)  # B x rows x cols, float
    mapy: torch.Tensor = ptsd[..., 1].reshape(B, rows, cols)  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols), mapx.type_as(image), mapy.type_as(image), align_corners=True,)

    return out.view_as(image) 

def calibrate_stereo_pair_torch(img_left, img_right, K_left, K_right, dist_left, dist_right, R, t, height, width):
    
    R1,R2,P1,P2,Q = stereorectify_torch(K_left,dist_left,K_right,dist_right,(width,height),R,t)
    img_left_rect =  undistortrectify(img_left,K_left,dist_left,R1,P1[...,:3])
    img_right_rect = undistortrectify(img_right,K_right,dist_right,R2,P2[...,:3])

    return img_left_rect, img_right_rect, {"R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q}


#def get_disparity_torch(img_left, img_right):

def estimate_pose_kornia(kpts0, kpts1, K0, K1, mconf,
                         thresh=1., scale=None, filter_height=-1, 
                         filter_conf=-1, top_8=False):

    if scale != None:
        kpts0 = kpts0*scale
        kpts1 = kpts1*scale

    if len(kpts0) < 5:
        return None

    if filter_height != -1:
        filtered_kps = kpts0[:, 1] > filter_height
        kpts0 = kpts0[filtered_kps]
        kpts1 = kpts1[filtered_kps]
        mconf = mconf[filtered_kps]

    if top_8:
        top_eight_kps = torch.argsort(mconf, dim=0, descending=True)[:8]
        kpts0 = kpts0[top_eight_kps]
        kpts1 = kpts1[top_eight_kps]
        mconf = mconf[top_eight_kps]

    if filter_conf != -1:
        filtered_kps = mconf > filter_conf

        if len(filtered_kps) >= 8:
            kpts0 = kpts0[filtered_kps]
            kpts1 = kpts1[filtered_kps]
            mconf = mconf[filtered_kps]

    kpts0 = kpts0.unsqueeze(0)
    kpts1 = kpts1.unsqueeze(0)
    mconf = mconf.unsqueeze(0)

    F = find_fundamental(kpts0, kpts1, mconf)
    E = essential_from_fundamental(F, K0.float(), K1.float())

    assert E is not None

    R, t, kpts3D = motion_from_essential_choose_solution(E, K0.float(), K1.float(), kpts0, kpts1)

    return R, t, kpts3D

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, margin=10, small_text=[]):

    H0, W0, C = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    for x, y in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    for x, y in kpts1:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                    lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                    lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Small text.
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)

    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    return out

def estimate_pose_ransac(kpts0, kpts1, K0, K1, mconf,
                         thresh=1., scale=None, filter_height=-1, 
                         filter_conf=-1, top_8=False):

    if scale != None:
        kpts0 = kpts0*scale
        kpts1 = kpts1*scale

    if len(kpts0) < 5:
        return None

    if filter_height != -1:
        filtered_kps = kpts0[:, 1] > filter_height
        kpts0 = kpts0[filtered_kps]
        kpts1 = kpts1[filtered_kps]
        mconf = mconf[filtered_kps]

    if top_8:
        top_eight_kps = torch.argsort(mconf, dim=0, descending=True)[:8]
        kpts0 = kpts0[top_eight_kps]
        kpts1 = kpts1[top_eight_kps]
        mconf = mconf[top_eight_kps]

    if filter_conf != -1:
        filtered_kps = mconf > filter_conf

        if len(filtered_kps) >= 8:
            kpts0 = kpts0[filtered_kps]
            kpts1 = kpts1[filtered_kps]
            mconf = mconf[filtered_kps]

    kpts0 = kpts0.unsqueeze(0)
    kpts1 = kpts1.unsqueeze(0)
    mconf = mconf.unsqueeze(0)
    rnsc = ransac.RANSAC(model_type='fundamental', inl_th=0.5)
    #F = find_fundamental(kpts0, kpts1, mconf)
    F, inliers = rnsc(kpts0[0, ...], kpts1[0, ...], mconf[0, ...])
    F = F.unsqueeze(0)
    E = essential_from_fundamental(F, K0.float(), K1.float())

    assert E is not None

    R, t, kpts3D = motion_from_essential_choose_solution(E, K0.float(), K1.float(), kpts0, kpts1)

    return R, t, kpts3D

def unrectify_np(left_img, right_img, K1, K2, T1_inv, T2_inv, P1, P2, width, height):
    # Define output image size
    output_size = (width, height)  # Specify the desired size of the output images

    # Create output images
    output1 = np.zeros(left_img.shape, dtype=np.uint8)
    output2 = np.zeros(right_img.shape, dtype=np.uint8)

    # Generate pixel grid
    x_grid, y_grid = np.meshgrid(np.arange(output_size[0]), np.arange(output_size[1]))

    # Flatten pixel grid
    pixels = np.column_stack((x_grid.flatten(), y_grid.flatten(), np.ones(x_grid.size)))

    # Back-project pixel coordinates to camera_coordinate system
    x1_cam_grid = np.matmul(np.linalg.inv(K1), pixels.T)
    x2_cam_grid = np.matmul(np.linalg.inv(K2), pixels.T)
    x1_cam_grid = np.concatenate([x1_cam_grid, np.ones((1, x1_cam_grid.shape[1]))], axis=0)
    x2_cam_grid = np.concatenate([x2_cam_grid, np.ones((1, x2_cam_grid.shape[1]))], axis=0)

    X1_world_grid = np.matmul(T1_inv, x1_cam_grid)
    X2_world_grid = np.matmul(T2_inv, x2_cam_grid)

    X1 = np.matmul(P1, X1_world_grid).T
    X2 = np.matmul(P2, X2_world_grid).T

    # Perform image interpolation
    x1 = (X1[:, 0] / X1[:, 2]).reshape(output_size)
    y1 = (X1[:, 1] / X1[:, 2]).reshape(output_size)

    x2 = (X2[:, 0] / X2[:, 2]).reshape(output_size)
    y2 = (X2[:, 1] / X2[:, 2]).reshape(output_size)

    # Interpolate pixel values
    pixel_values1 = cv2.remap(left_img, x1.astype(np.float32), y1.astype(np.float32), cv2.INTER_LINEAR)
    pixel_values2 = cv2.remap(right_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)

    # Reshape interpolated pixel values to output image shape
    output1 = pixel_values1.reshape(left_img.shape)
    output2 = pixel_values2.reshape(right_img.shape)

    return output1, output2

def find_sift_match(torch_img1, torch_img2):
    device = torch_img1.device

    img1, img2 = torch_img1.detach().cpu().numpy(), torch_img2.detach().cpu().numpy()
    img1, img2 = img1[0].transpose(1, 2, 0), img2[0].transpose(1, 2, 0)
    img1, img2 = (img1*255).astype('uint8'), (img2*255).astype('uint8')

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

    return {'mkeypoints0': torch.from_numpy(pts1).to(device).float(), 'mkeypoints1': torch.from_numpy(pts2).to(device).float(), 'match_confidence': torch.from_numpy(sift_mconf).to(device).float()}

def find_sift_match2(torch_img1, torch_img2):
    device = torch_img1.device

    img1, img2 = torch_img1.detach().cpu().numpy(), torch_img2.detach().cpu().numpy()
    img1, img2 = img1[0].transpose(1, 2, 0), img2[0].transpose(1, 2, 0)
    img1, img2 = (img1*255).astype('uint8'), (img2*255).astype('uint8')

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)

    pts1 = []
    pts2 = []
    mconf = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            mconf.append(n.distance/m.distance)

    return {'mkeypoints0': torch.from_numpy(np.array(pts1)).to(device).float(), 
            'mkeypoints1': torch.from_numpy(np.array(pts2)).to(device).float(), 
            'match_confidence': torch.from_numpy(np.array(mconf)).to(device).float()}

def magsac_pose_estimation(kpts1, kpts2, mconf, intrinsics_left, intrinsics_right, hw, filter_conf=-1):

    device = kpts1.device

    kpts1, kpts2, mconf = kpts1.squeeze(), kpts2.squeeze(), mconf.squeeze()
    
    #import pdb; pdb.set_trace();

    if filter_conf != -1:
        filtered_kps = mconf > filter_conf

        if len(filtered_kps) >= 8:
            kpts1 = kpts1[filtered_kps]
            kpts2 = kpts2[filtered_kps]
            mconf = mconf[filtered_kps]

    pts1 = kpts1.detach().cpu().numpy().squeeze()
    pts2 = kpts2.detach().cpu().numpy().squeeze()
    K1 = intrinsics_left.detach().cpu().numpy().squeeze()
    K2 = intrinsics_right.detach().cpu().numpy().squeeze()

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    '''
    pts_l_norm = np.expand_dims(pts1, axis=1) #cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
    pts_l_norm[:, :, 1] /= hw[0]
    pts_l_norm[:, :, 0] /= hw[1]
    pts_r_norm = np.expand_dims(pts2, axis=1) #cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)
    pts_r_norm[:, :, 1] /= hw[0]
    pts_r_norm[:, :, 0] /= hw[1]
    '''

    #E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, cv2.USAC_MAGSAC, 0.25, 0.999)
    F, inliers = cv2.findFundamentalMat(pts_l_norm, pts_r_norm, cv2.USAC_MAGSAC, 0.25, 0.9, 100000)
    E = F #K2.T @ F @ K1

    try:
        points, R_est, _, mask_pose = cv2.recoverPose(E, pts_l_norm[inliers==1], pts_r_norm[inliers==1])
    except:
        print(pts_l_norm.shape, pts_r_norm.shape)
        R_est = np.eye(3)

    return torch.from_numpy(R_est.reshape(1, 3, 3)).to(device).float()


def opencv_pose_estimation(torch_img1, torch_img2, torch_K1, torch_K2):
    
    device = torch_img1.device
    img1, img2 = torch_img1.detach().cpu().numpy(), torch_img2.detach().cpu().numpy()
    img1, img2 = img1[0].transpose(1, 2, 0), img2[0].transpose(1, 2, 0)
    img1, img2 = (img1*255).astype('uint8'), (img2*255).astype('uint8')
    K1, K2 = torch_K1[0].detach().cpu().numpy(), torch_K2[0].detach().cpu().numpy()

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
    mconf = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            mconf.append(n.distance/m.distance)
    
    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

    #E, inliers = cv2.findEssentialMat(pts_l_norm, pts_r_norm, camera)
    
    F, inliers = cv2.findFundamentalMat(pts_l_norm, pts_r_norm, cv2.USAC_DEFAULT)
    E = F #K2.T @ F @ K1
    
    try:
        points, R_est, _, mask_pose = cv2.recoverPose(E, pts_l_norm[inliers==1], pts_r_norm[inliers==1])
    except:
        print(pts_l_norm.shape, pts_r_norm.shape)
        R_est = np.eye(3)

    #print("T_opencv", get_transformation_matrix(R_est, t_est))

    return {'mkeypoints0': torch.from_numpy(np.asarray(pts1)).to(device).float(), 
            'mkeypoints1': torch.from_numpy(np.asarray(pts2)).to(device).float(), 
            'match_confidence': torch.from_numpy(np.asarray(mconf)).to(device).float()}, torch.from_numpy(R_est.reshape(1, 3, 3)).to(device).float()