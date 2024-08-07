import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import torch


class Simulator:

    def __init__(self, urdf_file_path, cameras, mode='gui'):
        # Connect to the PyBullet physics server
        if mode == 'gui':
            self.physics_client = p.connect(p.GUI)
        elif mode == 'direct':
            self.physics_client = p.connect(p.DIRECT)
        else:
            raise NotImplementedError("This mode doesn't exists or hasn't been implemented yet")
        
        # Load the PyBullet data package, which contains various URDF files and assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the plane and set its friction characteristics
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

        # Create gravity in the simulation
        p.setGravity(0, 0, -9.81)

        # create ids of objects in the scene
        self.robot_id = p.loadURDF(urdf_file_path, useFixedBase=True)
        self.cameras = cameras
        self.block_id_list = []

    def add_blocks(self):
        block_urdf_path = pybullet_data.getDataPath() + "/cube_small.urdf"
        
        # Define some block positions
        block_positions = [
            (0.5, 0, 0.025),
            (0.6, 0, 0.025),
            (0.5, 0.1, 0.025),
            (0.6, 0.1, 0.025)
        ]
        
        # Define colors for each block (R, G, B, Alpha)
        block_colors = [
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1],  # Blue
            [1, 1, 0, 1]   # Yellow
        ]

        for pos, color in zip(block_positions, block_colors):
            block_id = p.loadURDF(block_urdf_path, pos, (0, 0, 0, 1))
            p.changeVisualShape(block_id, -1, rgbaColor=color)  # Change the color of the block
            self.block_id_list.append(block_id)

    def capture_images(self, cam_id=0):
        """
        Renders RGB, depth, and object segmentation from the scene for an specific camera.

        Parameters:
        cam_id  : int, id of the camera from where to render.

        Returns:
        A tuple containing the RGB, depth, and segmentation arrays 
        """
        camera = self.cameras[cam_id]

        img_arr = p.getCameraImage(width=camera.image_width, 
                                   height=camera.image_height, 
                                   viewMatrix=camera.view_matrix, 
                                   projectionMatrix=camera.proj_matrix, 
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # get the rendered images into usable format
        rgb_image = np.reshape(np.array(img_arr[2]), (camera.image_height, camera.image_width, 4))[:, :, :3]
        depth_image = np.reshape(np.array(img_arr[3]), (camera.image_height, camera.image_width))
        seg_image = np.reshape(np.array(img_arr[4]), (camera.image_height, camera.image_width))

        return (rgb_image, depth_image, seg_image)
    
    def capture_image_torch(self, cam_id=0):
        """
        """
        rgb_image, depth_image, seg_image = self.capture_image(cam_id)
        rgb_image = (torch.from_numpy(rgb_image) / 255.).permute(2, 0, 1)   # C, H, W
        depth_image = (torch.from_numpy(depth_image)).unsqueeze(dim=0)      # C, H, W
        seg_image = (torch.from_numpy(seg_image)).unsqueeze(dim=0)          # C, H, W
        return rgb_image, depth_image, seg_image

    def display_camera_images(self, cam_id=0):
        rgb, depth, seg = self.capture_images(cam_id)
        print(rgb.shape, depth.shape, seg.shape)
        seg_ = np.where(seg > 0 , seg * 25 , 0).astype(np.uint8)
        seg_ = cv2.applyColorMap(seg_, cv2.COLORMAP_JET)
        cv2.imshow(f'Camera View {cam_id} - RGB', rgb)
        cv2.imshow(f'Camera View {cam_id} - Depth', depth)
        cv2.imshow(f'Camera View {cam_id} - Segmentation', seg_)

if __name__ == '__main__':
    from camera import Camera

    camera_configs = {
        'cameraEyePosition': [1, 0, 1],
        'cameraTargetPosition': [0, 0, 0.2],
        'cameraUpVector': [0, 0, 1],
        'fov': 60,
        'aspect': 1.33,
        'near': 0.1,
        'far': 3.1,
        'image_width': 640,
        'image_height': 480
    }

    cam = Camera(camera_configs)
    sim_env = Simulator("kinova.urdf", [cam], mode='direct')
    sim_env.add_blocks()
    
    # Run the simulation and display the images in real time
    while True:
        # Step the simulation
        p.stepSimulation()
        
        # Display the captured images
        sim_env.display_camera_images(cam_id=0)
        
        # Wait for a short duration to simulate real-time
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    # Disconnect from the physics server
    p.disconnect()