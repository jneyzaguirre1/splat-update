import cv2
import numpy as np
import math
import torch
import pybullet as p



class Camera:

    def __init__(self, config):
        """
        Camera object that poses the parameters of each camera.
        
        Parameters
        config  : dictionary with the following camera parameters:
            - cameraEyePosition : list, [x, y, z]
            - cameraTargetPosition : list, [x, y, z]
            - cameraUpVector : list, [x, y, z]
            - fov : int
            - aspect : float
            - near : float
            - far : float
            - image_width : int
            - image_height : int
        """
        # parameters for pybullet
        for key, value in config.items():
            setattr(self, key, value)
        self.intrinsics = self.compute_intrinsics()
        self.view_matrix = self.compute_view_matrix()
        self.proj_matrix = self.compute_proj_matrix()   # pybullet
        self.pose, self.extrinsics = self.compute_pose()

        # parameters for 3DGS:
        # TODO: finish setting all the necessary variables
        self.FoVx = self.FoVy = self.fov
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda() # 3DGS
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def compute_intrinsics(self):
        """
        https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        """
        f = self.image_height / (2.0 * np.tan(self.fov * np.pi / 360.0))
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        intrinsics = np.array([[f, 0, cx],
                               [0, f, cy],
                               [0, 0, 1]])
        return intrinsics
    
    def compute_view_matrix(self):
        """
        https://gamedev.stackexchange.com/questions/178643/the-view-matrix-finally-explained
        """
        view_matrix = p.computeViewMatrix(self.cameraEyePosition, 
                                          self.cameraTargetPosition, 
                                          self.cameraUpVector)
        view_matrix_ = np.array(view_matrix).reshape(4, 4).T
        return view_matrix
    
    def compute_proj_matrix(self):
        """
        """
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, 
                                                   self.aspect, 
                                                   self.near, 
                                                   self.far)
        return proj_matrix
    
    def compute_pose(self):
        """
        """
        extrinsics = np.array(self.view_matrix).reshape(4, 4).T
        return self.compute_X_inverse(extrinsics), extrinsics

    def compute_X_inverse(self, X):
        """
        """
        R_inv = X[:3, :3].T
        t_inv = -R_inv @ X[:3, 3]
        inv_matrix = np.eye(4)
        inv_matrix[:3, :3] = R_inv
        inv_matrix[:3, 3] = t_inv

        return inv_matrix
    
    def getProjectionMatrix(znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    

