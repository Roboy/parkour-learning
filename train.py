from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler, CpuResetCollector
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.envs.gym import make as gym_make
from typing import Dict
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from logger_context import config_logger
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.algos.pg.ppo import PPO
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
        sac_kwargs=dict(learning_rate=3e-4, batch_size=1024, replay_size=1e6, discount=0.95),
        ppo_kwargs=dict(minibatches=4, learning_rate=0.0001, value_loss_coeff=0.01, linear_lr_schedule=False),
        sampler_kwargs=dict(batch_T=5, batch_B=9, TrajInfoCls=RobotTrajInfo,
                            env_kwargs=dict(id="TrackEnv-v0"),
                            eval_n_envs=9,
                            eval_max_steps=1e5,
                            eval_max_trajectories=10),
        agent_kwargs=dict(ModelCls=PiMCPModel, QModelCls=QofMCPModel, model_kwargs=dict(freeze_primitives=False)),
        runner_kwargs=dict(n_steps=1e9, log_interval_steps=1e5),
        snapshot=snapshot,
        algo='sac'
    )

    if slot_affinity_code is None:
        num_cpus = multiprocessing.cpu_count()  # divide by two due to hyperthreading
        num_gpus = len(GPUtil.getAvailable())
        if config['algo'] == 'sac' and not serial_mode:
            affinity = make_affinity(n_cpu_core=num_cpus, n_gpu=num_gpus, async_sample=True)
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
        SamplerClass = CpuSampler
        algo_kwargs = config['ppo_kwargs']
    elif config['algo'] == 'sac':
        AgentClass = SacAgentSafeLoad
        AlgoClass = SAC
        algo_kwargs = config['sac_kwargs']
        if not serial_mode:
            SamplerClass = AsyncCpuSampler
            RunnerClass = AsyncRlEval
            affinity['cuda_idx'] = 0
            # create new affinity because async_sample has to be set
            print('affinity: '  + str(affinity))
            # affinity= make_affinity(
            #     n_cpu_core=len(affinity['all_cpus'])//2,  # Use 16 cores across all experiments.
            #     n_gpu=1,
            #     async_sample=True,
            # )
            # print('affinity after: '  + str(affinity))

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
