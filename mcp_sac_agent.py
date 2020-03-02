from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from typing import Dict
import torch
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])

class MCPSacAgent(SacAgent):
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

    def pi(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        mean, log_std, gating, primitive_means, primtive_stds = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        # Action stays on device for q models.
        return action, log_pi, dist_info, gating, primitive_means, primtive_stds

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        mean, log_std, gating, primitive_means, primitive_stds = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
