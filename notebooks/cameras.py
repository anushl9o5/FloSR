import numpy as np
import os
import cv2


class Camera:
    def __init__(self, cam_mat, dist, name=None):
        self.name = name
        self.cam_mat = cam_mat
        self.dist = dist
    
    def __repr__(self):
        return f'<Camera name = "{self.name}", cam_mat = {self.cam_mat}, cam_dist = {self.dist} >'

class StereoCamera:
    def __init__(self, cam_ref:Camera, cam_follow:Camera, rvec, tvec, imsize = (1928, 1208)):
        self.cam_ref = cam_ref
        self.cam_follow = cam_follow
        self.rvec = rvec
        # keep the units in meters (to fix the units inconsistency of extrinsic parameters in different captures)
        if np.linalg.norm(tvec) > 10:
            print("fix the units inconsistency")
            tvec /=1000
        self.tvec = tvec
        self.imsize= imsize

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(cam_ref.cam_mat, cam_ref.dist, cam_follow.cam_mat, cam_follow.dist,
                                                                                                 self.imsize, self.rvec, self.tvec)

        self.mapx1, self.mapy1 = cv2.initUndistortRectifyMap(self.cam_ref.cam_mat, self.cam_ref.dist, self.R1, self.P1, self.imsize, 5)
        self.mapx2, self.mapy2 = cv2.initUndistortRectifyMap(self.cam_follow.cam_mat, self.cam_follow.dist, self.R2, self.P2, self.imsize, 5)

    @property
    def focal_length(self):
        '''
            return the focal length of rectified camera.
        '''

        return self.Q[2,3]

    @property
    def baseline(self):
        '''
            returns the baselin in meters.
        '''
        # return np.linalg.norm(self.tvec)
        return 1/self.Q[3,2]
    
    def get_rectf_rot_matrix(self):
        return self.R1, self.R2
    
    def get_rectf_proj_matrix(self):
        return self.P1,self.P2 
    
    def intrs_post_rectf(self):
        '''
            return intrisnic matrix for rectified cameras for the stereo cameras
        '''
        cameraMatrix1, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P1)
        cameraMatrix2, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(self.P2)
        
        return cameraMatrix1,cameraMatrix2
    
    def dispToPointcloud(self, disp, is_ref=True):
        m = 1 if is_ref else -1
        pc = cv2.reprojectImageTo3D(m*disp, self.Q)
        return pc

    def project_to_2d(self, p3d, is_ref=True):
        ''' project a pointcloud to image (only the visible points).
        '''
        assert len(p3d.shape) == 2, 'p3d must be 2D array (Nx3) or (Nx4).'
        assert p3d.shape[1] in [3,4], 'each point in p3d must has 3 or 4 dimension (Nx3) or (Nx4).'
        if p3d.shape[1]==3:
            p3d = np.concatenate((p3d, np.ones((p3d.shape[0],1))), axis=1)
        P = self.P1 if is_ref else self.P2
        R = self.R1 #if is_ref else self.R2
        p3d[:,:3] = (R.dot(p3d[:,:3].T)).T
        visible_inds = p3d[:,2]>0
        p2d = P.dot(p3d[visible_inds].T)
        p2d = p2d[:2]/p2d[2]
        return p2d.T, visible_inds

    def rectifyImage(self, im, is_ref = True):
        assert im.shape[:2] == self.imsize[::-1], f'{im.shape[:2]} vs {self.imsize[::-1]}'
        if is_ref:
            im_rect = cv2.remap(im, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        else:
            im_rect = cv2.remap(im, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return im_rect