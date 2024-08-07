#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import open3d as o3d
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.camera import Camera
from scene.sim_env import Simulator
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}


        # added simulation scene info reader 
        if args.sim:
            self.sim_env = self.create_sim_env(args)
            scene_info = sceneLoadTypeCallbacks["Sim"](self.sim_env, args)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            print(args.source_path)
            assert False, "Could not recognize scene type!"

        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #         dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if args.sim:
            self.train_cameras[1.0] = self.sim_env.cameras
        else:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def create_sim_env(self, args):
        cameras = self.create_cameras(args.width, args.height, args.cam_num)
        environment = Simulator(args.urdf_path, cameras, mode='direct')
        environment.add_blocks()
        return environment

    def create_cameras(self, width, height, cam_num):
        cameras = []
        camera_configs = {
            'cameraTargetPosition': [0, 0, 0.2],
            'cameraUpVector': [0, 0, 1],
            'fov': 60,
            'aspect': 1.33,
            'near': 0.1,
            'far': 3.1,
            'width': width,
            'height': height
        }
        ang_space = np.linspace(0, 2*np.pi, num=cam_num)
        for angle in ang_space:
            cameraEyePosition = self.spherical_to_cartesian(1., 0., angle)
            cameraEyePosition[3] = 0.5
            camera_configs['cameraEyePosition'] = cameraEyePosition
            cam = Camera(camera_configs)
            cameras.append(cam)
        random.shuffle(cameras)
        return cameras

    def spherical_to_cartesian(self, r, theta, phi):
        """
        Convert spherical coordinates to Cartesian coordinates.

        Parameters:
        r     : radius (the distance from the origin to the point)
        theta : inclination/polar angle (angle from the positive z-axis)
        phi   : azimuthal angle (angle from the positive x-axis in the xy-plane)

        Angles should be given in radians.

        Returns:
        A tuple (x, y, z) representing the Cartesian coordinates of the point.
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return [x, y, z]



