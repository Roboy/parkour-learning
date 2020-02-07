import json
import numpy as np
import math
import os.path as osp


class MotionCaptureData(object):

    def __init__(self, mocap_file_name):
        path = osp.join(osp.dirname(__file__), osp.join('../motions/', mocap_file_name))
        with open(path) as f:
            self._motion_data = json.load(f)
        self.cycle_offset = self.computeCycleOffset()

    def get_frame_data(self, frame_index):
        return self._motion_data['Frames'][frame_index]

    def num_frames(self):
        return len(self._motion_data['Frames'])

    def key_frame_duration(self):
        return self._motion_data['Frames'][0][0]

    def is_cyclic_motion(self):
        return self._motion_data['Loop'] == 'wrap'

    def getCycleTime(self):
        keyFrameDuration = self.key_frame_duration()
        cycleTime = keyFrameDuration * (self.num_frames() - 1)
        return cycleTime

    def calcCycleCount(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count = math.floor(phases)
        loop = True
        # count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
        return count

    def computeCycleOffset(self):
        first_frame_data = self._motion_data['Frames'][0]
        lastFrame = self.num_frames() - 1
        last_frame_data = self._motion_data['Frames'][lastFrame]

        # basePosStart = [first_frame_data[1], first_frame_data[2], first_frame_data[3]]
        # basePosEnd = [last_frame_data[1], last_frame_data[2], last_frame_data[3]]
        # self._cycleOffset = [
        #     basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
        #     basePosEnd[2] - basePosStart[2]
        # ]
        base_pos_start = first_frame_data[1:4]
        base_pos_end = last_frame_data[1:4]
        cycle_offset = np.array(base_pos_end) - np.array(base_pos_start)
        return cycle_offset
