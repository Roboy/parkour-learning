from rlpyt.agents.qpg.sac_agent import SacAgent
from typing import Dict


class SacAgentSafeLoad(SacAgent):
    # Sac agent that loads state dict even when keys are missing; useful for training mcp model on new task
    def load_state_dict(self, state_dict):
        if 'q1_model' in state_dict.keys():
            self.q1_model.load_state_dict(self.get_updated_dict(self.q1_model.state_dict(), state_dict["q1_model"]))
        if 'q2_model' in state_dict.keys():
            self.q2_model.load_state_dict(self.get_updated_dict(self.q2_model.state_dict(), state_dict["q2_model"]))
        if 'target_q1_model' in state_dict.keys():
            self.target_q1_model.load_state_dict(
                self.get_updated_dict(self.target_q1_model.state_dict(), state_dict["target_q1_model"]))
        if 'target_q2_model' in state_dict.keys():
            self.target_q2_model.load_state_dict(
                self.get_updated_dict(self.target_q2_model.state_dict(), state_dict["target_q2_model"]))
        if 'model' in state_dict.keys():
            self.model.load_state_dict(self.get_updated_dict(self.model.state_dict(), state_dict["model"]))

    @staticmethod
    def get_updated_dict(original_dict: Dict, dict_update: Dict):
        original_dict.update(dict_update)
        return original_dict
