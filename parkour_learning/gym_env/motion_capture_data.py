import json
import numpy as np
import math
import os.path as osp


class MotionCaptureData(object):

    def __init__(self, mocap_file_name):
        path = osp.join(osp.dirname(__file__), osp.join('../motions/', mocap_file_name))
        with open(path) as f:
            self._motion_data = json.load(f)
        self.cycle_offset = self.compute_cycle_offset()
        self.cycle_time = self.key_frame_duration() * (self.num_frames() - 1)
        self.num_frames = len(self._motion_data['Frames'])

    def get_frame_data(self, frame_index):
        return self._motion_data['Frames'][frame_index]

    def num_frames(self):
        return len(self._motion_data['Frames'])

    def key_frame_duration(self):
        return self._motion_data['Frames'][0][0]

    def is_cyclic_motion(self):
        """
        returns true if mocap can be looped.
        The position offset after one cycle can be computed with compute_cycle_offset()
        """
        return self._motion_data['Loop'] == 'wrap'

    def getCycleTime(self):
        keyFrameDuration = self.key_frame_duration()
        cycleTime = keyFrameDuration * (self.num_frames() - 1)
        return cycleTime

    def calc_cycle_count(self, simTime, cycleTime):
        phases = simTime / cycleTime
        count = math.floor(phases)
        return count

    def compute_cycle_offset(self):
        first_frame_data = self._motion_data['Frames'][0]
        lastFrame = self.num_frames() - 1
        last_frame_data = self._motion_data['Frames'][lastFrame]
        base_pos_start = first_frame_data[1:4]
        base_pos_end = last_frame_data[1:4]
        cycle_offset = np.array(base_pos_end) - np.array(base_pos_start)
        return cycle_offset

    def get_objects_dict(self) -> dict:
        if 'Objects' in self._motion_data.keys():
            return self._motion_data['Objects']
        else:
            return dict()

