from rlpyt.agents.pg.mujoco import MujocoFfAgent
import numpy as np
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from typing import Dict
"""
Subclass of default rlpyt PPO agent. This allows to load state_dicts that contain only part of the model parameters
"""

class McpPPOAgent(MujocoFfAgent):
    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs, initial_model_state_dict=None)
        self._model_state_dict = initial_model_state_dict

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.load_state_dict()

    def load_state_dict(self):
        if self._model_state_dict is not None:
            self.model.load_state_dict(self.get_updated_dict(self.model.state_dict(), self._model_state_dict))

    @staticmethod
    def get_updated_dict(original_dict: Dict, dict_update: Dict):
        original_dict.update(dict_update)
        return original_dict
