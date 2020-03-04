from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler, CpuResetCollector
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.envs.gym import make as gym_make
from typing import Dict
# from rlpyt.algos.qpg.sac import SAC
from torch.optim.sgd import SGD
from mcp_sac import SAC
from mcp_sac_agent import MCPSacAgent
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
# from rlpyt.algos.pg.ppo import PPO
from ppo_seperate_learning_rates import PPO
from rlpyt.algos.qpg.td3 import TD3
from rlpyt.agents.qpg.td3_agent import Td3Agent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent, MujocoFfAgent
import gym
from sac_agent_safe_load import SacAgentSafeLoad
import torch
import GPUtil
import multiprocessing
from mcp_model import PiMCPModel, QofMCPModel
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from robot_traj_info import RobotTrajInfo
import argparse
from mcp_model import PPOMcpModel


def make(*args, info_example=None, **kwargs):
    import gym_parkour
    import pybulletgym
    import parkour_learning
    info_example = {'timeout': 0}
    return GymEnvWrapper(EnvInfoWrapper(
        gym.make(*args, **kwargs), info_example))


def build_and_train(slot_affinity_code=None, log_dir='./data', run_ID=0,
                    serial_mode=True,
                    snapshot: Dict=None,
                    config_update: Dict=None):
    config = dict(
        sac_kwargs=dict(reward_scale=10, min_steps_learn=0, learning_rate=3e-4, batch_size=256, replay_size=1e6, discount=0.95),
        ppo_kwargs=dict(minibatches=4, learning_rate=2e-5, discount=0.95, linear_lr_schedule=False,
                        OptimCls=SGD, optim_kwargs=dict(momentum=0.9), gae_lambda=0.95, ratio_clip=0.2),
        td3_kwargs=dict(),
        sampler_kwargs=dict(batch_T=32, batch_B=24, TrajInfoCls=RobotTrajInfo,
                            env_kwargs=dict(id="TrackEnv-v0"),
                            eval_n_envs=4,
                            eval_max_steps=1e5,
                            eval_max_trajectories=10),
        agent_kwargs=dict(ModelCls=PiMCPModel, QModelCls=QofMCPModel),
        runner_kwargs=dict(n_steps=1e9, log_interval_steps=1e5),
        snapshot=snapshot,
        algo='td3'
    )

    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()  # divide by two due to hyperthreading
        num_gpus = len(GPUtil.getGPUs())
        if config['algo'] == 'sac' and not serial_mode:
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, async_sample=True)
        elif config['algo'] == 'ppo' and not serial_mode:
            affinity = dict(
                        alternating=True,
                        cuda_idx=0,
                        workers_cpus=2 * list(range(num_cpus)),
                        async_sample=True
                    )

        else:
            affinity = make_affinity(n_cpu_core=num_cpus // 2, n_gpu=num_gpus)
    else:
        affinity = affinity_from_code(slot_affinity_code)

    try:
        variant = load_variant(log_dir)
        config = update_config(config, variant)
    except FileNotFoundError:
        if config_update is not None:
            config = update_config(config, config_update)

    agent_state_dict = optimizer_state_dict = None
    if config['snapshot'] is not None:
        # snapshot = torch.load(config['snapshot_file'], map_location=torch.device('cpu'))
        agent_state_dict = config['snapshot']['agent_state_dict']
        optimizer_state_dict = config['snapshot']['optimizer_state_dict']

    if config['algo'] == 'ppo':
        AgentClass = MujocoFfAgent
        AlgoClass = PPO
        RunnerClass = MinibatchRlEval
        if serial_mode:
            SamplerClass = CpuSampler
        else:
            SamplerClass = AlternatingSampler
        algo_kwargs = config['ppo_kwargs']
    elif config['algo'] == 'sac':
        AgentClass = MCPSacAgent
        AlgoClass = SAC
        algo_kwargs = config['sac_kwargs']
        if not serial_mode:
            SamplerClass = AsyncCpuSampler
            RunnerClass = AsyncRlEval
            affinity['cuda_idx'] = 0
    elif config['algo'] == 'td3':
        AgentClass = Td3Agent
        AlgoClass = TD3
        algo_kwargs = config['td3_kwargs']

    if serial_mode:
        SamplerClass = SerialSampler
        RunnerClass = MinibatchRlEval
        config['runner_kwargs']['log_interval_steps'] = 1e3
        config['sac_kwargs']['min_steps_learn'] = 1000

    sampler = SamplerClass(
        **config['sampler_kwargs'],
        EnvCls=make,
        eval_env_kwargs=config['sampler_kwargs']['env_kwargs']
    )
    algo = AlgoClass(**algo_kwargs, initial_optim_state_dict=optimizer_state_dict)
    agent = AgentClass(initial_model_state_dict=agent_state_dict, **config['agent_kwargs'])
    runner = RunnerClass(
        **config['runner_kwargs'],
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity
    )
    config_logger(log_dir, name='parkour-training', snapshot_mode='best', log_params=config)
    runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('slot_affinity_code', nargs='?', default=None,
                        help='using all possible resources when not specified')
    parser.add_argument('log_dir_positional', nargs='?', help='required for automatic launching')
    parser.add_argument('run_id', nargs='?', help='required for automatic launching')
    parser.add_argument('--serial_mode', dest='serial_mode', action='store_true',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--no_serial_mode', dest='serial_mode', action='store_false',
                        help='flag to run in serial mode is easier for debugging')
    parser.add_argument('--log_dir', required=False,
                        help='path to directory where log folder will be; Overwrites log_dir_positional')
    parser.add_argument('--snapshot_file', help='path to snapshot params.pkl containing state_dicts',
                        default=None)  # '/home/alex/reinforcement/deepmove/logs/run_13/params.pkl')

    args = parser.parse_args()
    log_dir = args.log_dir or args.log_dir_positional or './data'
    print("training started with parameters: " + str(args))
    snapshot = None
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))

    config_update = dict(sampler_kwargs=dict(env_kwargs=dict(id='HumanoidPrimitivePretraining-v0')))

    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    snapshot=snapshot,
                    config_update=config_update,
                    serial_mode=args.serial_mode)
