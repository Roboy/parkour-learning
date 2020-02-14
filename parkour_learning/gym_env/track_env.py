import numpy as np
from random import sample, randint
import math
import os
import os.path as osp
from pybullet_utils import bullet_client
from parkour_learning.gym_env.pd_control.humanoid_pose_interpolator import HumanoidPoseInterpolator
import pybullet
import random
import gym, gym.spaces, gym.utils
from parkour_learning.gym_env.motion_capture_data import MotionCaptureData
from parkour_learning.gym_env.pd_control.humanoid_stable_pd import HumanoidStablePD
from parkour_learning.gym_env.humanoid_mimic import HumanoidMimic
from parkour_learning.gym_env.humanoid import Humanoid


class TrackEnv(gym.Env):

    def __init__(self, render=False):
        self.action_repeat = 10
        self.timestep_length = 1 / 500
        self.time_limit = 10
        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self._pybullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/bullet_data')
        self._plane_id = self._pybullet_client.loadURDF("plane.urdf", [0, 0, 0],
                                                        self._pybullet_client.getQuaternionFromEuler(
                                                            [-math.pi * 0.5, 0, 0]))
        self._pybullet_client.loadURDF('track.urdf', useFixedBase=1, basePosition=(7, 0, 0), baseOrientation=(0, 0, 0, -1))
        self.humanoid = Humanoid(self._pybullet_client, time_step_length=self.timestep_length)
        self._pybullet_client.setGravity(0, -9.8, 0)
        self._pybullet_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)
        # self._humanoid = HumanoidMimic(self._pybullet_client, self.mocap_objects[0], self.timestep_length, False)
        self._pybullet_client.setTimeStep(self.timestep_length)
        self.action_dim = 43
        self.obs_dim = 196
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(43,))
        observation_example = self.get_observation()
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=-1, high=1, shape=observation_example['state'].shape),
            'goal': gym.spaces.Box(low=-1, high=1, shape=observation_example['goal'].shape)
        })

    def reset(self):
        self.humanoid.reset()
        return self.get_observation()

    def step(self, action):
        desired_pose = np.array(self.humanoid.convertActionToPose(action))
        desired_pose[:7] = 0
        # we need the target root positon and orientation to be zero, to be compatible with deep mimic
        maxForces = [
            0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
            90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
            100, 100, 60
        ]
        for i in range(self.action_repeat):
            self.humanoid.computeAndApplyPDForces(desired_pose, maxForces)
            self._pybullet_client.stepSimulation()

        # reward = self._humanoid.getReward()
        done = False
        reward = 0
        observation = self.get_observation()
        return observation, reward, done, {}

    def get_observation(self):
        state_observation = np.array(self.humanoid.getState())
        goal_observation = np.array([])
        observation = dict(
            state=state_observation,
            goal=goal_observation
        )
        return observation

    def render(self, mode='human'):
        current_camera_info = self._pybullet_client.getDebugVisualizerCamera()
        self._pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=current_camera_info[10],
            cameraYaw=current_camera_info[8],
            cameraPitch=current_camera_info[9],
            cameraTargetPosition=self.humanoid.get_position())
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
