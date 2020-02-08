import numpy as np
import torch
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dModel


class PiMCPModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            num_primitives=4
    ):
        super().__init__()
        assert hasattr(observation_shape, 'state'), "mcp model requires observation dict to contain state attribute"
        assert hasattr(observation_shape, 'goal'), "mcp model requires observation to contain goal attribute"
        robot_state_out = 256
        self.primitives_state_mlp = MlpModel(
            input_size=observation_shape.state,
            hidden_sizes=[512, 256],
            output_size=robot_state_out
        )
        self.primitive_mlps = []
        for primitive in range(num_primitives):
            primitive_mlp = MlpModel(
                input_size=256,
                hidden_sizes=[256],
                output_size=2 * action_size  # 2x for mu and std
            )
            self.primitive_mlps.append(primitive_mlp)

        self.gating_state_mlp = MlpModel(
            input_size=observation_shape.state,
            hidden_sizes=[512],
            output_size=256,
        )
        self.gating_goal_mlp = MlpModel(
            input_size=observation_shape.state,
            hidden_sizes=[512],
            output_size=256,
        )
        self.gating_mlp = MlpModel(
            input_size=256 + 256,
            hidden_sizes=[],
            output_size=256
        )
        init_log_std = 0.
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.robot_state, 1)

        state_obs_flat = observation.robot_state.view(T * B, -1)
        goal_obs_flat = observation.camera.view(T * B, -1)
        primitive_state_out = self.primitives_state_mlp(state_obs_flat)
        primitves_out = []
        for primitve_mlp in self.primitive_mlps:
            primitve_out = primitve_mlp(primitive_state_out)
            mu = fk
        robot_state = self.robot_state_mlp(robot_state_obs_flat)
        cnn_out = self.conv(camera_obs_flat).view(T * B, -1)  # apply conv and flatten afterwards
        mlp_head_in = torch.cat((robot_state, cnn_out), -1)
        mu = self.mu_head(mlp_head_in)
        log_std = self.log_std.repeat(T * B, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class QofMCPModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
    ):
        super().__init__()
        assert hasattr(observation_shape, 'camera'), "VisionFfModel requires observation to contain 'camera' attr"
        assert hasattr(observation_shape,
                       'robot_state'), "VisionFfModel requires observation to contain 'robot_state' attr"
        self.height, self.width, self.channels = observation_shape.camera
        robot_state_shape = observation_shape.robot_state[0]
        self.conv = Conv2dModel(
            in_channels=self.channels,
            channels=[8, 20],
            kernel_sizes=[5, 4],
            strides=[3, 3],
            paddings=[1, 1],
        )
        conv_out_size = self.conv.conv_out_size(self.height, self.width)
        robot_state_out = 256
        self.robot_state_mlp = MlpModel(
            input_size=robot_state_shape,
            hidden_sizes=[],
            output_size=robot_state_out
        )
        self.q_head = MlpModel(
            input_size=robot_state_out + conv_out_size + action_size,
            hidden_sizes=[256, ],
            output_size=1
        )
        init_log_std = 0.
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))

    def forward(self, observation, prev_action, prev_reward, action):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation.robot_state, 1)

        robot_state_obs_flat = observation.robot_state.view(T * B, -1)
        camera_obs_flat = observation.camera.view(T * B, self.channels, self.height, self.width)
        robot_state = self.robot_state_mlp(robot_state_obs_flat)
        cnn_out = self.conv(camera_obs_flat).view(T * B, -1)  # apply conv and flatten afterwards
        # q_input = torch.cat(
        #     [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        mlp_head_in = torch.cat((robot_state, cnn_out, action.view(T * B, -1)), -1)
        q = self.q_head(mlp_head_in).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q
