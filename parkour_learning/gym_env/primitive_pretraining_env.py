import numpy as np
import math
import copy
from random import sample, randint
import math
import os
import os.path as osp
from pybullet_utils.bullet_client import BulletClient
from parkour_learning.gym_env.pd_control.humanoid_pose_interpolator import HumanoidPoseInterpolator
import pybullet
import random
import gym, gym.spaces, gym.utils
from parkour_learning.gym_env.motion_capture_data import MotionCaptureData
from parkour_learning.gym_env.pd_control.humanoid_stable_pd import HumanoidStablePD
from parkour_learning.gym_env.humanoid_mimic import HumanoidMimic
from parkour_learning.gym_env.humanoid import Humanoid


class PrimitivePretrainingEnv(gym.Env):
    mocap_files = ['run.txt'] #, 'walk.txt' , 'jump_and_roll.txt', 'vaulting.txt', 'run.txt']
    mocap_folder = osp.join(osp.dirname(__file__), '../motions/')

    def __init__(self, render=False):
        self.action_repeat = 8
        self.timestep_length = 1 / 240
        self.time_limit = 3
        self.min_time_per_mocap = 1
        self.time_in_episode = self.time_of_mocap = self.time_since_mocap_change = self.completed_mocap_cycles = None
        self.bullet_client = self._bullet_connect(render)
        self.bullet_client.setAdditionalSearchPath(os.path.dirname(__file__) + '/bullet_data')
        self._plane_id = self.bullet_client.loadURDF("plane.urdf", [0, 0, 0],
                                                     self.bullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0]),
                                                     useMaximalCoordinates=True)
        self.bullet_client.setGravity(0, -9.8, 0)
        self.bullet_client.changeDynamics(self._plane_id, linkIndex=-1, lateralFriction=0.95)
        self.bullet_client.setTimeStep(self.timestep_length)
        self.humanoid = Humanoid(self.bullet_client, self.timestep_length)
        self.mocap_humanoid = Humanoid(self.bullet_client, self.timestep_length)
        self.disable_all_collisions(self.mocap_humanoid)
        self.mocap_humanoid.set_alpha(0.7)
        self.pose_interpolator = HumanoidPoseInterpolator()
        self.current_mocap = None
        self.mocap_obects = []
        self.action_dim = 43
        self.obs_dim = 196
        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(43,))
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=-3, high=3, shape=(196,)),
            'goal': gym.spaces.Box(low=-3, high=3, shape=(80,)),
        })

    def _bullet_connect(self, render: bool) -> BulletClient:
        if render:
            bullet_client = BulletClient(connection_mode=pybullet.GUI)
            bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        else:
            bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
        return bullet_client

    def reset(self):
        self.set_random_mocap_file()
        self.time_in_episode = 0
        return self.get_observation()

    def step(self, action):
        action = np.clip(action, a_min=-3, a_max=3)
        desired_pose = np.array(self.humanoid.convertActionToPose(action))
        # we need the target root positon and orientation to be zero, to be compatible with deep mimic
        desired_pose[:7] = 0
        for i in range(self.action_repeat):
            if self.time_of_mocap/self.current_mocap.cycle_time and not self.current_mocap.is_cyclic_motion:
                self.set_random_mocap_file()
            self.set_mocap_pose(self.mocap_humanoid)
            self.humanoid.computeAndApplyPDForces(desired_pose)
            self.time_in_episode += self.timestep_length
            self.time_of_mocap += self.timestep_length
            self.time_since_mocap_change += self.timestep_length
            self.bullet_client.stepSimulation()

        reward = self.get_reward()
        done = self.compute_done(reward)
        observation = self.get_observation()
        return observation, reward, done, {}

    def compute_done(self, reward):
        done = False
        collisions = self.bullet_client.getContactPoints(bodyA=self.humanoid.humanoid_uid, bodyB=self._plane_id)
        for collision in collisions:
            collided_link = collision[3]
            if collided_link not in [self.humanoid.joint_indeces['leftAnkle'],
                                     self.humanoid.joint_indeces['rightAnkle'],
                                     self.humanoid.joint_indeces['leftElbow'],
                                     self.humanoid.joint_indeces['rightElbow']]:
                done = True

        if self.time_in_episode > self.time_limit:
            done = True

        return done

    def get_observation(self):
        state_observation = np.array(self.humanoid.getState())
        goal_observation = self.get_mocap_observation()
        assert not np.isnan(goal_observation).any(), 'goal observation is nan: ' + str(goal_observation)
        assert not np.isnan(state_observation).any(), 'state observation is nan: ' + str(state_observation)
        observation = dict(
            state=state_observation,
            goal=goal_observation
        )
        return observation

    def render(self, mode='human'):
        current_camera_info = self.bullet_client.getDebugVisualizerCamera()
        self.bullet_client.resetDebugVisualizerCamera(
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

        view_matrix = self.bullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(render_width) / render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=render_width, height=render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def set_mocap_pose(self, humanoid: Humanoid):
        mocap_cycle_fraction = self.time_of_mocap / self.current_mocap.cycle_time
        if mocap_cycle_fraction >= 1:
            self.completed_mocap_cycles += 1
            self.time_of_mocap = 0
            mocap_cycle_fraction -= 1
            self.set_mocap_objects()

        exact_frame = mocap_cycle_fraction * self.current_mocap.num_frames
        frame_fraction = exact_frame - int(exact_frame)

        frame_index_before = int(exact_frame)
        frame_index_after = (frame_index_before + 1) % self.current_mocap.num_frames
        frame_data_before = copy.copy(self.current_mocap.get_frame_data(frame_index_before))
        frame_data_after = copy.copy(self.current_mocap.get_frame_data(frame_index_after))

        base_pos_offset = self.completed_mocap_cycles * self.current_mocap.cycle_offset
        if frame_index_after == 0:
            frame_data_after[1:4] += self.current_mocap.cycle_offset
        self.pose_interpolator.Slerp(frame_fraction, frame_data_before, frame_data_after, self.bullet_client,
                                     base_pos_offset)
        humanoid.set_with_pose_interpolator(self.pose_interpolator)

    def set_mocap_objects(self):
        self.delete_mocap_objects()

        for data_file, position in self.current_mocap.get_objects_dict().items():
            position = np.array(position) + self.completed_mocap_cycles * self.current_mocap.cycle_offset
            self.mocap_obects.append(self.bullet_client.loadURDF(data_file, position, useFixedBase=True))

    def delete_mocap_objects(self):
        for object_uid in self.mocap_obects:
            self.bullet_client.removeBody(object_uid)
        self.mocap_obects = []

    def set_random_mocap_file(self):
        new_mocap_index = randint(0, len(self.mocap_files) - 1)
        self.current_mocap = MotionCaptureData(self.mocap_folder + self.mocap_files[new_mocap_index])
        self.time_of_mocap = randint(0, int(1000 * self.current_mocap.cycle_time)) / 1000
        self.time_since_mocap_change = 0
        self.completed_mocap_cycles = 0
        self.set_mocap_pose(self.humanoid)
        self.set_mocap_pose(self.mocap_humanoid)
        self.set_mocap_objects()

    def get_mocap_observation(self):
        """
        return observation vector base on mocap data
        """
        mocap_cycle_fraction = self.time_of_mocap / self.current_mocap.cycle_time
        next_frame_index = math.ceil(mocap_cycle_fraction * self.current_mocap.num_frames) % self.current_mocap.num_frames
        next_frame = self.current_mocap.get_frame_data(next_frame_index)
        next_next_frame = self.current_mocap.get_frame_data((next_frame_index + 1) % self.current_mocap.num_frames)
        next_joint_positions = next_frame[4:]
        next_next_joint_positions = next_next_frame[4:]
        return np.array(next_joint_positions + next_next_joint_positions)

    def get_reward(self):
        # from DeepMimic double cSceneImitate::CalcRewardImitate
        # todo: compensate for ground height in some parts, once we move to non-flat terrain
        pose_w = 0.65 # 0.5
        vel_w = 0.1 # 0.05
        end_eff_w = 0.15
        root_w = 0.1 # 0.2
        com_w = 0  # 0.1

        total_w = pose_w + vel_w + end_eff_w + root_w + com_w
        pose_w /= total_w
        vel_w /= total_w
        end_eff_w /= total_w
        root_w /= total_w
        com_w /= total_w

        pose_scale = 2
        vel_scale = 0.1
        end_eff_scale = 40
        root_scale = 5
        com_scale = 10
        err_scale = 1

        reward = 0

        pose_err = 0
        vel_err = 0
        end_eff_err = 0
        root_err = 0
        com_err = 0
        heading_err = 0

        # create a mimic reward, comparing the dynamics humanoid with a kinematic one

        root_id = 0
        mJointWeights = [
            0.20833, 0.10416, 0.0625, 0.10416, 0.0625, 0.041666666666666671, 0.0625, 0.0416, 0.00,
            0.10416, 0.0625, 0.0416, 0.0625, 0.0416, 0.0000
        ]

        num_end_effs = 0
        num_joints = 15

        root_rot_w = mJointWeights[root_id]
        rootPosSim, rootOrnSim = self.bullet_client.getBasePositionAndOrientation(self.humanoid.humanoid_uid)
        rootPosKin, rootOrnKin = self.bullet_client.getBasePositionAndOrientation(self.mocap_humanoid.humanoid_uid)
        linVelSim, angVelSim = self.bullet_client.getBaseVelocity(self.humanoid.humanoid_uid)
        # don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
        # see also issue https://github.com/bulletphysics/bullet3/issues/2401
        linVelKin = self.pose_interpolator._baseLinVel
        angVelKin = self.pose_interpolator._baseAngVel

        root_rot_err = self.calcRootRotDiff(rootOrnSim, rootOrnKin)
        pose_err += root_rot_w * root_rot_err

        root_vel_diff = [
            linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
        ]
        root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
            1] + root_vel_diff[2] * root_vel_diff[2]

        root_ang_vel_err = self.calcRootAngVelErr(angVelSim, angVelKin)
        vel_err += root_rot_w * root_ang_vel_err

        useArray = True

        if useArray:
            jointIndices = range(num_joints)
            simJointStates = self.bullet_client.getJointStatesMultiDof(self.humanoid.humanoid_uid, jointIndices)
            kinJointStates = self.bullet_client.getJointStatesMultiDof(self.mocap_humanoid.humanoid_uid, jointIndices)
        if useArray:
            linkStatesSim = self.bullet_client.getLinkStates(self.humanoid.humanoid_uid, jointIndices)
            linkStatesKin = self.bullet_client.getLinkStates(self.mocap_humanoid.humanoid_uid, jointIndices)
        for j in range(num_joints):
            curr_pose_err = 0
            curr_vel_err = 0
            w = mJointWeights[j]
            if useArray:
                simJointInfo = simJointStates[j]
            else:
                simJointInfo = self.bullet_client.getJointStateMultiDof(self.humanoid.humanoid_uid, j)

            # print("simJointInfo.pos=",simJointInfo[0])
            # print("simJointInfo.vel=",simJointInfo[1])
            if useArray:
                kinJointInfo = kinJointStates[j]
            else:
                kinJointInfo = self.bullet_client.getJointStateMultiDof(self.mocap_humanoid.humanoid_uid, j)
            # print("kinJointInfo.pos=",kinJointInfo[0])
            # print("kinJointInfo.vel=",kinJointInfo[1])
            if len(simJointInfo[0]) == 1:
                angle = simJointInfo[0][0] - kinJointInfo[0][0]
                curr_pose_err = angle * angle
                velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
                curr_vel_err = velDiff * velDiff
            if len(simJointInfo[0]) == 4:
                # print("quaternion diff")
                diffQuat = self.bullet_client.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
                axis, angle = self.bullet_client.getAxisAngleFromQuaternion(diffQuat)
                curr_pose_err = angle * angle
                diffVel = [
                    simJointInfo[1][0] - kinJointInfo[1][0], simJointInfo[1][1] - kinJointInfo[1][1],
                    simJointInfo[1][2] - kinJointInfo[1][2]
                ]
                curr_vel_err = diffVel[0] * diffVel[0] + diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2]

            pose_err += w * curr_pose_err
            vel_err += w * curr_vel_err

            is_end_eff = j in self.mocap_humanoid.end_effectors

            if is_end_eff:

                if useArray:
                    linkStateSim = linkStatesSim[j]
                    linkStateKin = linkStatesKin[j]
                else:
                    linkStateSim = self.bullet_client.getLinkState(self.humanoid.humanoid_uid, j)
                    linkStateKin = self.bullet_client.getLinkState(self.mocap_humanoid.humanoid_uid, j)
                linkPosSim = linkStateSim[0]
                linkPosKin = linkStateKin[0]
                linkPosDiff = [
                    linkPosSim[0] - linkPosKin[0], linkPosSim[1] - linkPosKin[1],
                    linkPosSim[2] - linkPosKin[2]
                ]
                curr_end_err = linkPosDiff[0] * linkPosDiff[0] + linkPosDiff[1] * linkPosDiff[
                    1] + linkPosDiff[2] * linkPosDiff[2]
                end_eff_err += curr_end_err
                num_end_effs += 1

        if num_end_effs > 0:
            end_eff_err /= num_end_effs

        # double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos())
        # double root_ground_h1 = kin_char.GetOriginPos()[1]
        # root_pos0[1] -= root_ground_h0
        # root_pos1[1] -= root_ground_h1
        root_pos_diff = [
            rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
        ]
        root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
            1] + root_pos_diff[2] * root_pos_diff[2]
        #
        # root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1)
        # root_rot_err *= root_rot_err

        # root_vel_err = (root_vel1 - root_vel0).squaredNorm()
        # root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm()

        root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err

        # com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

        # print("pose_err=",pose_err)
        # print("vel_err=",vel_err)
        pose_reward = math.exp(-err_scale * pose_scale * pose_err)
        vel_reward = math.exp(-err_scale * vel_scale * vel_err)
        end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
        root_reward = math.exp(-err_scale * root_scale * root_err)
        com_reward = math.exp(-err_scale * com_scale * com_err)

        reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward

        # pose_reward,vel_reward,end_eff_reward, root_reward, com_reward);
        # print("reward=",reward)
        # print("pose_reward=",pose_reward)
        # print("vel_reward=",vel_reward)
        # print("end_eff_reward=",end_eff_reward)
        # print("root_reward=",root_reward)
        # print("com_reward=",com_reward)

        return reward

    def quatMul(self, q1, q2):
        return [
            q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
            q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
            q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
            q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        ]

    def calcRootAngVelErr(self, vel0, vel1):
        diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
        return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    def calcRootRotDiff(self, orn0, orn1):
        orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
        q_diff = self.quatMul(orn1, orn0Conj)
        axis, angle = self.bullet_client.getAxisAngleFromQuaternion(q_diff)
        return angle * angle

    def disable_all_collisions(self, humanoid: Humanoid):
        self.bullet_client.setCollisionFilterGroupMask(humanoid.humanoid_uid,
                                                          -1,
                                                          collisionFilterGroup=0,
                                                          collisionFilterMask=0)
        for j in range(self.bullet_client.getNumJoints(humanoid.humanoid_uid)):
            self.bullet_client.setCollisionFilterGroupMask(humanoid.humanoid_uid,
                                                              j,
                                                              collisionFilterGroup=0,
                                                              collisionFilterMask=0)
