import numpy as np
import math
import gym
import os
from pybullet_utils import bullet_client
import time
import pybullet_data
import pybullet as p1
import random
import gym, gym.spaces, gym.utils
from parkour_learning.gym_env.motion_capture_data import MotionCaptureData
from parkour_learning.gym_env.pd_control.humanoid_stable_pd import HumanoidStablePD


class PyBulletDeepMimicEnv(gym.Env):

    def __init__(self, mocap_file_path='humanoid3d_jump.txt', render=False):
        self.render = render
        self.action_dim = 43
        self.obs_dim = 197
        high = np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([self.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        if self.render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        self._pybullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/bullet_data')
        self._plane_id = self._pybullet_client.loadURDF("plane.urdf", [0, 0, 0],
                                                        self._pybullet_client.getQuaternionFromEuler(
                                                            [-math.pi * 0.5, 0, 0]),
                                                        useMaximalCoordinates=True)
        self._pybullet_client.setGravity(0, -9.8, 0)
        self._pybullet_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)
        mocap_data = MotionCaptureData(mocap_file_path)
        self._humanoid = HumanoidStablePD(self._pybullet_client, mocap_data, 1 / 240, False)
        self.timestep_length = 1 / 240
        self.step_in_episode = 0
        self.action_repeat = 4

    def reset(self):
        self.step_in_episode = 0
        self._humanoid.resetPose()
        return np.array(self._humanoid.getState())

    def step(self, action):
        self._humanoid.computeAndApplyPDForces(action, [10] * self.action_dim)
        self._pybullet_client.setTimeStep(self.timestep_length)
        self._humanoid._time_step_length = self.timestep_length

        for i in range(1):
            self.step_in_episode += 1
            self._humanoid.step_kin_model(self.step_in_episode)
            self._pybullet_client.stepSimulation()

        observation = np.array(self._humanoid.getState())
        reward = self._humanoid.getReward()
        done = False
        return observation, reward, done, {}

    def render(self, mode='human'):
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
