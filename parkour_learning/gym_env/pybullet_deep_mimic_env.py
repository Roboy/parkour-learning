import numpy as np
import math
import gym
import os
from pybullet_utils import bullet_client
import time
import pybullet_data
import pybullet
import random
import gym, gym.spaces, gym.utils
from parkour_learning.gym_env.motion_capture_data import MotionCaptureData
from parkour_learning.gym_env.pd_control.humanoid_stable_pd import HumanoidStablePD


class PyBulletDeepMimicEnv(gym.Env):

    def __init__(self, mocap_file_path='humanoid3d_run.txt', render=False):
        self.action_dim = 43
        self.obs_dim = 197
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/bullet_data')
        self._plane_id = self._pybullet_client.loadURDF("plane.urdf", [0, 0, 0],
                                                        self._pybullet_client.getQuaternionFromEuler(
                                                            [-math.pi * 0.5, 0, 0]),
                                                        useMaximalCoordinates=True)
        self._pybullet_client.setGravity(0, -9.8, 0)
        self._pybullet_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)
        self.mocap_data = MotionCaptureData(mocap_file_path)
        self.timestep_length = 1 / 500
        self._humanoid = HumanoidStablePD(self._pybullet_client, self.mocap_data, self.timestep_length, False)
        self._pybullet_client.setTimeStep(self.timestep_length)
        self.time_in_episode = 0
        self.action_repeat = 10

    def reset(self):
        # RSI : sample random start time
        self.time_in_episode = random.uniform(0, self.mocap_data.num_frames() * self.mocap_data.key_frame_duration())
        self._humanoid.resetPose(self.time_in_episode)
        return np.array(self._humanoid.getState())

    def step(self, action):
        desired_pose = np.array(self._humanoid.convertActionToPose(action))
        desired_pose[:7] = 0
        # we need the target root positon and orientation to be zero, to be compatible with deep mimic
        maxForces = [
            0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
            90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
            100, 100, 60
        ]
        for i in range(self.action_repeat):
            self.time_in_episode += self.timestep_length
            self._humanoid.step_kin_model(self.time_in_episode)
            self._humanoid.computeAndApplyPDForces(desired_pose, maxForces)
            self._pybullet_client.stepSimulation()

        observation = np.array(self._humanoid.getState())
        reward = self._humanoid.getReward()
        done = reward < 0.4
        return observation, reward, done, {}

    def render(self, mode='human'):
        current_camera_info = self._pybullet_client.getDebugVisualizerCamera()
        self._pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=current_camera_info[10],
            cameraYaw=current_camera_info[8],
            cameraPitch=current_camera_info[9],
            cameraTargetPosition=self._humanoid.get_position())
        if mode != "rgb_array":
            return np.array([])

        render_width = 320
        render_height = 240
        base_pos = [0, 0, 0]
        if (hasattr(self, 'robot')):
            if (hasattr(self.robot, 'body_xyz')):
                base_pos = self.robot.body_xyz
        target = [0, 0, 0]

        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(render_width) / render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=render_width, height=render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
