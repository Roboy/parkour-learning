import multiprocessing
import GPUtil
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel


# script to launch hyperparameter search

num_cpus = multiprocessing.cpu_count()
num_gpus = len(GPUtil.getAvailable())

affinity_code = encode_affinity(
    n_cpu_core=num_cpus,
    n_gpu=num_gpus,
    set_affinity=True,  # it can help to restrict workers to individual CPUs
)

runs_per_setting = 1
experiment_title = "parkour_challenge"

variants = [
    {
        'algo': 'ppo',
        'ppo_kwargs': dict(minibatches=4),
        'sampler_kwargs': dict(batch_B=32)
    },
    {
        'algo': 'ppo',
        'ppo_kwargs': dict(minibatches=32),
        'sampler_kwargs': dict(batch_B=32)
    },
]
log_dirs = ["pc_ppo" + str(id) for id in range(len(variants))]

# run_experiments(
#     script="train.py",
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
# )
num_gpus = 1
affinity_code = encode_affinity(
    n_cpu_core=num_cpus//2,
    cpu_per_run=2,
    n_gpu=2,
    async_sample=True,  # for sac
    set_affinity=True,  # it can help to restrict workers to individual CPUs
)


####SAC
runs_per_setting = 1
experiment_title = "parkour_challenge"

variants = [
    {
        'algo': 'sac',
        'sac_kwargs': dict(batch_size=256)
    },
    {
        'algo': 'sac',
        'sac_kwargs': dict(batch_size=1024)
    },
]
log_dirs = ["pc_sac_" + str(id) for id in range(len(variants))]
run_experiments(
    script="train.py",
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
)
