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
from collections import deque


class TrackEnv(gym.Env):
    camera_img_width = 20
    camera_img_heigth = 20

    def __init__(self, render=False):
        self.action_repeat = 10
        self.timestep_length = 1 / 500
        self.time_limit = 10
        self.target_pos = np.array([10, 0, 0])
        if render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self._pybullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/bullet_data')
        self._plane_id = self._pybullet_client.loadURDF("plane.urdf", [0, 0, 0],
                                                        self._pybullet_client.getQuaternionFromEuler(
                                                            [-math.pi * 0.5, 0, 0]))
        # self._pybullet_client.loadURDF('track.urdf', useFixedBase=1, basePosition=(7, 0, 0), baseOrientation=(0, 0, 0, -1))
        track_id = self._pybullet_client.loadSDF('RoboyParkourTrack/model.sdf')[0]
        self._pybullet_client.resetBasePositionAndOrientation(track_id, [5, 0, -2], [0, 0, 0, 1])
        self.humanoid = Humanoid(self._pybullet_client, time_step_length=self.timestep_length)
        self._pybullet_client.setGravity(0, -9.8, 0)
        self._pybullet_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.9)
        # self._humanoid = HumanoidMimic(self._pybullet_client, self.mocap_objects[0], self.timestep_length, False)
        self._pybullet_client.setTimeStep(self.timestep_length)
        self.action_dim = 43
        self.obs_dim = 196
        self.max_num_steps=2000
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(43,))
        observation_example = self.get_observation()
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=-1, high=1, shape=observation_example['state'].shape),
            'goal':  gym.spaces.Dict({
                'camera': gym.spaces.Box(low=-1, high=1, shape=observation_example['goal']['camera'].shape),
                'relative_target': gym.spaces.Box(low=-100, high=100, shape=(2,))
            })
        })
        self.last_100_goal_distances = None

    def reset(self):
        self.humanoid.reset()
        self.last_100_goal_distances = deque(maxlen=100)
        self.step_in_episode = 0
        return self.get_observation()

    def step(self, action):
        self.step_in_episode += 1
        desired_pose = np.array(self.humanoid.convertActionToPose(action))
        desired_pose[:7] = 0
        # we need the target root positon and orientation to be zero, to be compatible with deep mimic
        for i in range(self.action_repeat):
            self.humanoid.computeAndApplyPDForces(desired_pose)
            self._pybullet_client.stepSimulation()

        # reward = self._humanoid.getReward()
        goal_distance = np.linalg.norm(np.array(self.humanoid.get_position()) - self.target_pos)
        self.last_100_goal_distances.append(goal_distance)
        done = self.compute_done()
        reward = 10 - goal_distance
        observation = self.get_observation()
        return observation, reward, done, {}

    def get_observation(self):
        state_observation = np.array(self.humanoid.getState())
        head_pos = self.humanoid.get_head_pos()
        goal_direction = (self.target_pos - head_pos) / np.linalg.norm(self.target_pos - head_pos)
        camera_pos = np.array(head_pos) + goal_direction * 0.1
        view_matrix = self._pybullet_client.computeViewMatrix(
            cameraEyePosition=camera_pos,  # self.robot.body_xyz + [0, 0, 1],
            cameraTargetPosition=self.target_pos,
            cameraUpVector=(0, 1, 0)
        )
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=1.0,  # float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, depth_map, _) = self._pybullet_client.getCameraImage(
            width=self.camera_img_width, height=self.camera_img_heigth, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER
        )
        # rgb_array = np.array(px)
        # rgb_array = rgb_array[:, :, :3]
        # gray_img = np.mean(rgb_array, axis=2)
        observation = dict(
            state=state_observation,
            goal=dict(
                camera=depth_map,
                relative_target=goal_direction[:2]
            )
        )
        return observation

    def compute_done(self) -> bool:
        done = self.last_100_goal_distances[-1] < 1
        if len(self.last_100_goal_distances) == self.last_100_goal_distances.maxlen:
            done = (self.last_100_goal_distances[0] - self.last_100_goal_distances[-1]) < 1
        if self.step_in_episode > self.max_num_steps:
            done = True
        humanoid_collisions = self._pybullet_client.getContactPoints(bodyA=self.humanoid.humanoid_uid)
        for collision in humanoid_collisions:
            humanoid_link = collision[3]
            # don't know why the plane hase multiple links. But only if the plane link is 0, the result seems correct
            plane_link = collision[2]
            left_ankle_id = self.humanoid.joint_indeces['leftAnkle']
            right_ankle_id = self.humanoid.joint_indeces['rightAnkle']
            if plane_link == 0 and humanoid_link not in [left_ankle_id, right_ankle_id]:
                done = True
        return done

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
