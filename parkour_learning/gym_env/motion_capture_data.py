import json
import math
import os.path as osp


class MotionCaptureData(object):

  def __init__(self, mocap_file_name):
    path = osp.join(osp.dirname(__file__), osp.join('../motions/', mocap_file_name))
    with open(path) as f:
      self._motion_data = json.load(f)

  def num_frames(self):
    return len(self._motion_data['Frames'])

  def key_frame_duration(self):
    return self._motion_data['Frames'][0][0]

  def getCycleTime(self):
    keyFrameDuration = self.key_frame_duration()
    cycleTime = keyFrameDuration * (self.num_frames() - 1)
    return cycleTime

  def calcCycleCount(self, simTime, cycleTime):
    phases = simTime / cycleTime
    count = math.floor(phases)
    loop = True
    #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
    return count

  def computeCycleOffset(self):
    firstFrame = 0
    lastFrame = self.num_frames() - 1
    frameData = self._motion_data['Frames'][0]
    frameDataNext = self._motion_data['Frames'][lastFrame]

    basePosStart = [frameData[1], frameData[2], frameData[3]]
    basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._cycleOffset = [
        basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
        basePosEnd[2] - basePosStart[2]
    ]
    return self._cycleOffset
