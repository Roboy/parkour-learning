"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from logger_context import config_logger
import os
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper
from rlpyt.utils.logging import logger
from torch.utils.tensorboard.writer import SummaryWriter
import gym
import torch
import GPUtil
import multiprocessing


def make(*args, info_example=None, **kwargs):
    import gym_parkour  # necessary to register yumi_reacher envs
    import pybulletgym
    info_example = {'timeout': 0}
    return GymEnvWrapper(EnvInfoWrapper(
        gym.make(*args, **kwargs), info_example))


def build_and_train(env_id="Hopper-v3", snapshot_file: str=None, run_ID=0, cuda_idx=None):
    agent_state_dict = optimizer_state_dict = None
    if snapshot_file is not None:
        snapshot = torch.load(snapshot_file)
        agent_state_dict = snapshot['agent_state_dict']
        optimizer_state_dict = snapshot['optimizer_state_dict']

    num_cpus = multiprocessing.cpu_count()
    num_gpus = len(GPUtil.getAvailable())

    if num_gpus > 0:
        SamplerClass = AsyncCpuSampler
        RunnerClass = AsyncRlEval
        affinity = make_affinity(
            run_slot=0,
            n_cpu_core=8,  # Use 16 cores across all experiments.
            n_gpu=1,  # Use 8 gpus across all experiments.
            gpu_per_run=1,
            sample_gpu_per_run=0,
            async_sample=True,
            optim_sample_share_gpu=False,
            # hyperthread_offset=16,  # If machine has 24 cores.
            # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
            # gpu_per_run=2,  # How many GPUs to parallelize one run across.
            # cpu_per_run=1,
        )
    else:
        affinity=dict() # ;dict(cuda_idx=cuda_idx),
        SamplerClass = SerialSampler
        RunnerClass = MinibatchRlEval

    sampler = SamplerClass(
        EnvCls=make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=5,  # One time-step per sampler iteration.
        batch_B=7,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=10,
    )
    algo = SAC(initial_optim_state_dict=optimizer_state_dict)  # Run with defaults.
    agent = SacAgent(initial_model_state_dict=agent_state_dict)
    runner = RunnerClass(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e7,
        log_interval_steps=1e5,
        affinity=affinity
    )
    config_logger('./data', name='parkour-training', snapshot_mode='last')
    runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='ParkourChallenge-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        # snapshot_file='./data/run_5/params.pkl'
    )
