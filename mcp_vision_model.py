import torch
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from torch import sigmoid, relu, exp, softmax
from torch.nn import Linear, Conv2d
from torch.nn import ModuleList
from rlpyt.models.conv2d import Conv2dModel


# MCP models with convolutional layers for goal observation

class PiMcpVisionModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            num_primitives=4,
            freeze_primitives=False,
            hidden_sizes=None  # necessary for rlpyt compatibility
    ):
        super().__init__()
        assert hasattr(observation_shape, 'state'), "mcp model requires observation dict to contain state attribute"
        assert hasattr(observation_shape, 'goal'), "mcp model requires observation to contain goal attribute"
        self.num_primitives = num_primitives
        self.action_size = action_size
        self.primitives_l1 = Linear(observation_shape.state[0], 512)
        self.primitives_l2 = Linear(512, 256)
        self.primitives_l3s = ModuleList()
        self.primitives_l4s = ModuleList()
        for i in range(num_primitives):
            self.primitives_l3s.append(Linear(256, 256))
            # action size x2 because of mean standard deviation for each action
            self.primitives_l4s.append(Linear(256, action_size * 2))

        self.height, self.width = observation_shape.goal.camera
        # self.conv = Conv2dModel(
        #     in_channels=1,
        #     channels=[8, 20],
        #     kernel_sizes=[5, 4],
        #     strides=[3, 3],
        #     paddings=[1, 1],
        # )
        self.gating_state_l1 = Linear(observation_shape.state[0], 512)
        self.gating_state_l2 = Linear(512, 256)
        # conv_out_size = self.conv.conv_out_size(self.height, self.width)
        self.gating_l3 = Linear(observation_shape.goal.relative_target[0] + 256, 256)
        self.gating_l4 = Linear(256, num_primitives)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        # goal_input = observation.goal.view(T * B, -1)
        state_input = observation.state.view(T * B, -1)
        camera_obs_flat = observation.goal.camera.view(T * B, 1, self.height, self.width)
        relative_target_flat = observation.goal.relative_target.view(T * B, -1)
        assert not torch.isnan(camera_obs_flat).any(), "goal input is nan"
        assert not torch.isnan(state_input).any(), "state input is nan"
        # print('primitives: ' + str(next(self.primitives_l2.parameters())[0][0]))
        # print('gating: ' + str(next(self.gating_l4.parameters())[0][0]))
        # print('gating: ' + str(next(self.gating_l3.parameters())[0][0]))
        # inputs now with just one batch dimension at front
        # cnn_out = self.conv(camera_obs_flat).view(T * B, -1)  # apply conv and flatten afterwards
        gating_state = relu(self.gating_state_l1(state_input))
        gating_state = relu(self.gating_state_l2(gating_state))
        # gating_goal = relu(self.gating_goal_l1(goal_input))
        # gating_goal = relu(self.gating_goal_l2(gating_goal))
        gating = relu(self.gating_l3(torch.cat((gating_state, relative_target_flat), -1)))
        gating = sigmoid(self.gating_l4(gating))
        assert not torch.isnan(gating).any(), 'gating is nan'
        with torch.no_grad():
            primitives = relu(self.primitives_l1(state_input))
            primitives = relu(self.primitives_l2(primitives))
            primitives_means = []
            primitives_stds = []
            for i in range(self.num_primitives):
                x = relu(self.primitives_l3s[i](primitives))
                x = self.primitives_l4s[i](x)
                primitives_means.append(x[:, :self.action_size])
                # interpret last outputs as log stds
                primitives_stds.append(exp(x[:, self.action_size:].clamp(min=-20, max=20)))

        std = state_input.new_zeros((T * B, self.action_size,))
        mu = state_input.new_zeros((T * B, self.action_size))
        gating = gating.reshape((T * B, self.num_primitives, 1)).expand(-1, -1, self.action_size)
        for i in range(self.num_primitives):
            x = torch.div(gating[:, i].expand((T * B, self.action_size)), primitives_stds[i].clamp(min=1e-5))
            assert not torch.isnan(x).any(), 'x is nan'
            std = torch.add(std, x)
            mu = torch.add(mu, torch.mul(x, primitives_means[i]))
            assert not torch.isnan(mu).any(), 'mu is nan ' + str(mu) + str(x) + str(primitives_means)
        assert not torch.isnan(std).any(), 'std nan: '
        std = torch.div(1, std.clamp(min=1e-5))
        mu = torch.mul(mu, std)
        assert not torch.isnan(std).any(), 'std div nan: '
        log_std = torch.log(std.clamp(min=1e-5))
        assert not torch.isnan(mu).any(), "mu is nan " + str(mu) + str(std)
        assert not torch.isnan(log_std).any(), 'log std is nan'

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

    @staticmethod
    def remove_gating_from_snapshot(snapshot_dict):
        keys_to_remove = ['gating_state_l1.weight',
                          'gating_state_l1.bias',
                          'gating_state_l2.weight',
                          'gating_state_l2.bias',
                          'gating_goal_l1.weight',
                          'gating_goal_l1.bias',
                          'gating_goal_l2.weight',
                          'gating_goal_l2.bias',
                          'gating_l3.weight',
                          'gating_l3.bias',
                          'gating_l4.weight',
                          'gating_l4.bias',
                          ]
        [snapshot_dict.pop(key) for key in keys_to_remove]
        return snapshot_dict


class QofMcpVisionModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
    ):
        super().__init__()
        assert hasattr(observation_shape, 'state'), "mcp model requires observation dict to contain state attribute"
        assert hasattr(observation_shape, 'goal'), "mcp model requires observation to contain goal attribute"
        self.height, self.width = observation_shape.goal
        # self.conv = Conv2dModel(
        #     in_channels=1,
        #     channels=[8, 20],
        #     kernel_sizes=[5, 4],
        #     strides=[3, 3],
        #     paddings=[1, 1],
        # )
        # conv_out_size = self.conv.conv_out_size(self.height, self.width)
        self.robot_state_mlp = MlpModel(
            input_size=observation_shape.state[0],
            hidden_sizes=[512],
            output_size=256
        )
        self.mlp = MlpModel(
            input_size=256 + observation_shape.goal.relative_target[0] + action_size,
            hidden_sizes=[256],
            output_size=1
        )

    def forward(self, observation, prev_action, prev_reward, action):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        state_obs_flat = observation.state.view(T * B, -1)
        # camera_obs_flat = observation.goal.camera.view(T * B, 1, self.height, self.width)
        relative_target_flat = observation.goal.relative_target.view(T * B, -1)
        state = relu(self.robot_state_mlp(state_obs_flat))
        # cnn_out = self.conv(camera_obs_flat).view(T * B, -1)  # apply conv and flatten afterwards
        mlp_head_in = torch.cat((state, relative_target_flat, action.view(T * B, -1)), -1)
        q = self.mlp(mlp_head_in).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class PpoMcpVisionModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            num_primitives=4,
            freeze_primitives=False,
            hidden_sizes=None  # necessary for rlpyt compatibility
    ):
        self.normalize_observations = False
        super().__init__()
        assert hasattr(observation_shape, 'state'), "mcp model requires observation dict to contain state attribute"
        assert hasattr(observation_shape, 'goal'), "mcp model requires observation to contain goal attribute"
        self.v_model = MlpModel(
            input_size=observation_shape.state[0] + observation_shape.goal.relative_target[0],
            hidden_sizes=[1024, 512],
            output_size=1
        )
        self.num_primitives = num_primitives
        self.action_size = action_size
        self.primitives_l1 = Linear(observation_shape.state[0], 512)
        self.primitives_l2 = Linear(512, 256)
        self.primitives_l3s = ModuleList()
        self.primitives_l4s = ModuleList()
        for i in range(num_primitives):
            self.primitives_l3s.append(Linear(256, 256))
            # action size x2 because of mean standard deviation for each action
            self.primitives_l4s.append(Linear(256, action_size * 2))

        self.gating_state_l1 = Linear(observation_shape.state[0], 512)
        self.gating_state_l2 = Linear(512, 256)
        self.gating_goal_l1 = Linear(observation_shape.goal.relative_target[0], 512)
        self.gating_goal_l2 = Linear(512, 256)
        # self.gating_l3 = Linear(512, 256)
        # self.gating_l4 = Linear(256, num_primitives)
        self.gating_l3 = Linear(observation_shape.goal.relative_target[0] + 256, 256)
        self.gating_l4 = Linear(256, num_primitives)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        # goal_input = observation.goal.view(T * B, -1)
        state_input = observation.state.view(T * B, -1)
        relative_target_flat = observation.goal.relative_target.view(T * B, -1)
        # inputs now with just one batch dimension at front
        gating_state = relu(self.gating_state_l1(state_input))
        gating_state = relu(self.gating_state_l2(gating_state))
        # gating_goal = relu(self.gating_goal_l1(goal_input))
        # gating_goal = relu(self.gating_goal_l2(gating_goal))
        # gating = relu(self.gating_l3(torch.cat((gating_state, gating_goal), -1)))
        gating = relu(self.gating_l3(torch.cat((gating_state, relative_target_flat), -1)))
        gating = torch.softmax(self.gating_l4(gating), dim=-1)
        assert not torch.isnan(gating).any(), 'gating is nan'
        with torch.no_grad():
            primitives = relu(self.primitives_l1(state_input))
            primitives = relu(self.primitives_l2(primitives))
            primitives_means = []
            primitives_stds = []
            for i in range(self.num_primitives):
                x = relu(self.primitives_l3s[i](primitives))
                x = self.primitives_l4s[i](x)
                primitives_means.append(x[:, :self.action_size])
                # interpret last outputs as log stds
                primitives_stds.append(exp(x[:, self.action_size:].clamp(min=-20, max=20)))

        std = state_input.new_zeros((T * B, self.action_size,))
        mu = state_input.new_zeros((T * B, self.action_size))
        gating = gating.reshape((T * B, self.num_primitives, 1)).expand(-1, -1, self.action_size)
        for i in range(self.num_primitives):
            x = torch.div(gating[:, i].expand((T * B, self.action_size)), primitives_stds[i].clamp(min=1e-6))
            std = torch.add(std, x)
            mu = torch.add(mu, torch.mul(x, primitives_means[i]))

        std = torch.div(1, std.clamp(min=1e-5))
        mu = torch.mul(mu, std)
        log_std = torch.log(std)

        v = self.v_model(torch.cat((state_input, relative_target_flat), -1)).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
        return mu, log_std, v

    @staticmethod
    def remove_gating_from_snapshot(snapshot_dict):
        keys_to_remove = ['gating_state_l1.weight',
                          'gating_state_l1.bias',
                          'gating_state_l2.weight',
                          'gating_state_l2.bias',
                          'gating_goal_l1.weight',
                          'gating_goal_l1.bias',
                          'gating_goal_l2.weight',
                          'gating_goal_l2.bias',
                          'gating_l3.weight',
                          'gating_l3.bias',
                          'gating_l4.weight',
                          'gating_l4.bias',
                          'v_model.model.0.weight',
                          'v_model.model.0.bias',
                          'v_model.model.2.weight',
                          'v_model.model.2.bias',
                          'v_model.model.4.weight',
                          'v_model.model.4.bias',
                          ]
        [snapshot_dict.pop(key) for key in keys_to_remove]
        return snapshot_dict

    def update_obs_rms(self, observation):
        pass
