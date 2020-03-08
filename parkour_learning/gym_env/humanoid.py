# from pybullet_utils import pd_controller_stable
from parkour_learning.gym_env.pd_control import pd_controller_stable
import copy
# from pybullet_envs.deep_mimic.env import humanoid_pose_interpolator
from parkour_learning.gym_env.pd_control.humanoid_pose_interpolator import HumanoidPoseInterpolator
from parkour_learning.gym_env.motion_capture_data import MotionCaptureData
import numpy as np
import math

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
jointFrictionForce = 0


class Humanoid:
    maxForces = [
        0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
        90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
        100, 100, 60
    ]
    end_effectors = [5, 8, 11, 14]  # ankle and wrist, both left and right

    joint_indeces = dict(
        chest=1,
        neck=2,
        rightHip=3,
        rightKnee=4,
        rightAnkle=5,
        rightShoulder=6,
        rightElbow=7,
        leftHip=9,
        leftKnee=10,
        leftAnkle=11,
        leftShoulder=12,
        leftElbow=13,
    )

    def __init__(self, pybullet_client, time_step_length):
        self._pybullet_client = pybullet_client
        flags = self._pybullet_client.URDF_MAINTAIN_LINK_ORDER + self._pybullet_client.URDF_USE_SELF_COLLISION + self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self.humanoid_uid = self._pybullet_client.loadURDF(
            "humanoid.urdf", [0, 0.889540259, 0],
            globalScaling=0.25,
            flags=flags)

        self._pybullet_client.changeDynamics(self.humanoid_uid, -1, lateralFriction=0.9)
        for j in range(self._pybullet_client.getNumJoints(self.humanoid_uid)):
            self._pybullet_client.changeDynamics(self.humanoid_uid, j, lateralFriction=0.9)

        self._pybullet_client.changeDynamics(self.humanoid_uid, -1, linearDamping=0, angularDamping=0)

        self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
        self._time_step_length = time_step_length
        self._kpOrg = [
            0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500,
            400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
            400, 400, 400, 400, 300
        ]
        self._kdOrg = [
            0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40,
            40, 40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
        ]

        self._jointIndicesAll = [
            chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, leftHip, leftKnee,
            leftAnkle, leftShoulder, leftElbow
        ]
        for j in self._jointIndicesAll:
            # self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
            self._pybullet_client.setJointMotorControl2(self.humanoid_uid,
                                                        j,
                                                        self._pybullet_client.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        positionGain=0,
                                                        targetVelocity=0,
                                                        force=jointFrictionForce)
            self._pybullet_client.setJointMotorControlMultiDof(
                self.humanoid_uid,
                j,
                self._pybullet_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[jointFrictionForce, jointFrictionForce, jointFrictionForce])

        self._jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
        self.num_joints = np.sum(np.array(self._jointDofCounts))

        # [x,y,z] base position and [x,y,z,w] base orientation!
        self.pose_interpolator = HumanoidPoseInterpolator()
        self.episode_start_state = self._pybullet_client.saveState()

    def set_alpha(self, alpha):
        """
        change visual shape of humanoid
        :alpha 1 is normal, 0 is invisible, 0.5 is half seethrough
        """
        for link in range(self.num_joints):
            self._pybullet_client.changeVisualShape(self.humanoid_uid, link, rgbaColor=[1, 1, 1, alpha])

    def reset(self):
        """
        mocap_time_fraction: float between 0 and 1. this will be mapped to time of mocap data
        """
        self._pybullet_client.restoreState(self.episode_start_state)

    def get_position(self):
        return self._pybullet_client.getBasePositionAndOrientation(self.humanoid_uid)[0]

    def initializePose(self, pose, phys_model, initBase, initializeVelocity=True):
        useArray = True
        if initializeVelocity:
            if initBase:
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                                      pose._baseOrn)
                # print('reset to position: ' + str(pose._basePos))
                self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)
            if useArray:
                indices = [chest, neck, rightHip, rightKnee,
                           rightAnkle, rightShoulder, rightElbow, leftHip,
                           leftKnee, leftAnkle, leftShoulder, leftElbow]
                jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                                  pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                                  pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]

                jointVelocities = [pose._chestVel, pose._neckVel, pose._rightHipVel, pose._rightKneeVel,
                                   pose._rightAnkleVel, pose._rightShoulderVel, pose._rightElbowVel, pose._leftHipVel,
                                   pose._leftKneeVel, pose._leftAnkleVel, pose._leftShoulderVel, pose._leftElbowVel]
                self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,
                                                               jointPositions, jointVelocities)
            else:
                self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot,
                                                              pose._chestVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, pose._neckVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                              pose._rightHipVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot,
                                                              pose._rightKneeVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                              pose._rightAnkleVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                              pose._rightShoulderRot, pose._rightShoulderVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                              pose._rightElbowVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                              pose._leftHipVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot,
                                                              pose._leftKneeVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                              pose._leftAnkleVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                              pose._leftShoulderRot, pose._leftShoulderVel)
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot,
                                                              pose._leftElbowVel)
        else:

            if initBase:
                self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                                      pose._baseOrn)
            if useArray:
                indices = [chest, neck, rightHip, rightKnee,
                           rightAnkle, rightShoulder, rightElbow, leftHip,
                           leftKnee, leftAnkle, leftShoulder, leftElbow]
                jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                                  pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                                  pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]
                self._pybullet_client.resetJointStatesMultiDof(phys_model, indices, jointPositions)

            else:
                self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                              [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot, [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                              [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                              pose._rightShoulderRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                              [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                              [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot, [0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                              [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                              pose._leftShoulderRot, [0, 0, 0])
                self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot, [0])

    def convertActionToPose(self, action):
        pose = self.pose_interpolator.ConvertFromAction(self._pybullet_client, action)
        return pose

    def computeAndApplyPDForces(self, desiredPositions):
        dofIndex = 7
        scaling = 1
        indices = []
        forces = []
        targetPositions = []
        targetVelocities = []
        kps = []
        kds = []

        for index in range(len(self._jointIndicesAll)):
            jointIndex = self._jointIndicesAll[index]
            indices.append(jointIndex)
            kps.append(self._kpOrg[dofIndex])
            kds.append(self._kdOrg[dofIndex])
            if self._jointDofCounts[index] == 4:
                force = [
                    scaling * self.maxForces[dofIndex + 0],
                    scaling * self.maxForces[dofIndex + 1],
                    scaling * self.maxForces[dofIndex + 2]
                ]
                targetVelocity = [0, 0, 0]
                targetPosition = [
                    desiredPositions[dofIndex + 0],
                    desiredPositions[dofIndex + 1],
                    desiredPositions[dofIndex + 2],
                    desiredPositions[dofIndex + 3]
                ]
            if self._jointDofCounts[index] == 1:
                force = [scaling * self.maxForces[dofIndex]]
                targetPosition = [desiredPositions[dofIndex + 0]]
                targetVelocity = [0]
            forces.append(force)
            targetPositions.append(targetPosition)
            targetVelocities.append(targetVelocity)
            dofIndex += self._jointDofCounts[index]

        # static char* kwlist[] = { "bodyUniqueId",
        # "jointIndices",
        # "controlMode", "targetPositions", "targetVelocities", "forces", "positionGains", "velocityGains", "maxVelocities", "physicsClientId", NULL };
        self._pybullet_client.setJointMotorControlMultiDofArray(self.humanoid_uid,
                                                                indices,
                                                                self._pybullet_client.STABLE_PD_CONTROL,
                                                                targetPositions=targetPositions,
                                                                targetVelocities=targetVelocities,
                                                                forces=forces,
                                                                positionGains=kps,
                                                                velocityGains=kds,
                                                                )

    def buildHeadingTrans(self, rootOrn):
        # align root transform 'forward' with world-space x axis
        eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
        refDir = [1, 0, 0]
        rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
        heading = math.atan2(-rotVec[2], rotVec[0])
        heading2 = eul[1]
        # print("heading=",heading)
        headingOrn = self._pybullet_client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
        return headingOrn

    def buildOriginTrans(self):
        rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(self.humanoid_uid)

        # print("rootPos=",rootPos, " rootOrn=",rootOrn)
        invRootPos = [-rootPos[0], 0, -rootPos[2]]
        # invOrigTransPos, invOrigTransOrn = self._pybullet_client.invertTransform(rootPos,rootOrn)
        headingOrn = self.buildHeadingTrans(rootOrn)
        # print("headingOrn=",headingOrn)
        headingMat = self._pybullet_client.getMatrixFromQuaternion(headingOrn)
        # print("headingMat=",headingMat)
        # dummy, rootOrnWithoutHeading = self._pybullet_client.multiplyTransforms([0,0,0],headingOrn, [0,0,0], rootOrn)
        # dummy, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0,0,0],rootOrnWithoutHeading, invOrigTransPos, invOrigTransOrn)

        invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                    headingOrn,
                                                                                    invRootPos,
                                                                                    [0, 0, 0, 1])
        # print("invOrigTransPos=",invOrigTransPos)
        # print("invOrigTransOrn=",invOrigTransOrn)
        invOrigTransMat = self._pybullet_client.getMatrixFromQuaternion(invOrigTransOrn)
        # print("invOrigTransMat =",invOrigTransMat )
        return invOrigTransPos, invOrigTransOrn

    def getState(self):
        stateVector = []

        rootTransPos, rootTransOrn = self.buildOriginTrans()
        basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self.humanoid_uid)

        rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                     basePos, [0, 0, 0, 1])
        localPos, localOrn = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                      basePos, baseOrn)

        stateVector.append(rootPosRel[1])

        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        linkIndicesSim = []
        for pbJoint in range(self._pybullet_client.getNumJoints(self.humanoid_uid)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])

        linkStatesSim = self._pybullet_client.getLinkStates(self.humanoid_uid, linkIndicesSim,
                                                            computeForwardKinematics=True, computeLinkVelocity=True)

        for pbJoint in range(self._pybullet_client.getNumJoints(self.humanoid_uid)):
            j = self.pb2dmJoints[pbJoint]
            # print("joint order:",j)
            # ls = self._pybullet_client.getLinkState(self._sim_model, j, computeForwardKinematics=True)
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]
            linkPosLocal, linkOrnLocal = self._pybullet_client.multiplyTransforms(
                rootTransPos, rootTransOrn, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
                linkPosLocal[2] - rootPosRel[2]
            ]
            for l in linkPosLocal:
                stateVector.append(l)
            # re-order the quaternion, DeepMimic uses w,x,y,z

            if (linkOrnLocal[3] < 0):
                linkOrnLocal[0] *= -1
                linkOrnLocal[1] *= -1
                linkOrnLocal[2] *= -1
                linkOrnLocal[3] *= -1

            stateVector.append(linkOrnLocal[3])
            stateVector.append(linkOrnLocal[0])
            stateVector.append(linkOrnLocal[1])
            stateVector.append(linkOrnLocal[2])

        for pbJoint in range(self._pybullet_client.getNumJoints(self.humanoid_uid)):
            j = self.pb2dmJoints[pbJoint]
            # ls = self._pybullet_client.getLinkState(self._sim_model, j, computeLinkVelocity=True)
            ls = linkStatesSim[pbJoint]

            linkLinVel = ls[6]
            linkAngVel = ls[7]
            linkLinVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkLinVel, [0, 0, 0, 1])
            # linkLinVelLocal=[linkLinVelLocal[0]-rootPosRel[0],linkLinVelLocal[1]-rootPosRel[1],linkLinVelLocal[2]-rootPosRel[2]]
            linkAngVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                               linkAngVel, [0, 0, 0, 1])

            for l in linkLinVelLocal:
                stateVector.append(l)
            for l in linkAngVelLocal:
                stateVector.append(l)
        return stateVector

    def set_with_pose_interpolator(self, pose: HumanoidPoseInterpolator):
        self._pybullet_client.resetBasePositionAndOrientation(self.humanoid_uid, pose._basePos,
                                                              pose._baseOrn)
        # print('reset to position: ' + str(pose._basePos))
        self._pybullet_client.resetBaseVelocity(self.humanoid_uid, pose._baseLinVel, pose._baseAngVel)
        # indices = [chest, neck, rightHip, rightKnee,
        #            rightAnkle, rightShoulder, rightElbow, leftHip,
        #            leftKnee, leftAnkle, leftShoulder, leftElbow]
        indices = self.joint_indeces.values()
        jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                          pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                          pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]

        jointVelocities = [pose._chestVel, pose._neckVel, pose._rightHipVel, pose._rightKneeVel,
                           pose._rightAnkleVel, pose._rightShoulderVel, pose._rightElbowVel, pose._leftHipVel,
                           pose._leftKneeVel, pose._leftAnkleVel, pose._leftShoulderVel, pose._leftElbowVel]
        self._pybullet_client.resetJointStatesMultiDof(self.humanoid_uid, indices,
                                                       jointPositions, jointVelocities)

    def get_head_pos(self):
        link_states = self._pybullet_client.getLinkStates(self.humanoid_uid, [2])[0]
        pos = np.array(link_states[4])
        return pos
