from train import build_and_train
import torch
import argparse
from mcp_model import PiMCPModel
from mcp_vision_model import PiMCPModel, QofMCPModel, PPOMcpModel


algo = 'sac'

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
    parser.add_argument('--primitives_snapshot',
                        help='path to snapshot from pretraining; old gating params will be automatically removed')

    args = parser.parse_args()
    log_dir = args.log_dir or args.log_dir_positional or './data'
    print("training started with parameters: " + str(args))
    snapshot = None
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))
    elif args.primitives_snapshot is not None:
        snapshot = torch.load(args.primitives_snapshot, map_location=torch.device('cpu'))
        # only keep primitives
        if algo == 'sac':
            model_snapshot_dict = snapshot['agent_state_dict']['model']
            snapshot['agent_state_dict'] = dict()
            snapshot['agent_state_dict']['model'] = PiMCPModel.remove_gating_from_snapshot(model_snapshot_dict)
        elif algo == 'ppo':
            model_snapshot_dict = snapshot['agent_state_dict']
            snapshot['agent_state_dict'] = dict()
            snapshot['agent_state_dict'] = PPOMcpModel.remove_gating_from_snapshot(model_snapshot_dict)
        snapshot['optimizer_state_dict'] = None

    if algo == 'sac':
        config_update = dict(sac_agent_kwargs=dict(ModelCls=PiMCPModel, QModelCls=QofMCPModel, model_kwargs=dict(freeze_primitives=True)),
                         sampler_kwargs=dict(env_kwargs=dict(id='TrackEnv-v0')),
                         sac_kwargs=dict(discount=0.99),
                         algo='sac')
    elif algo == 'ppo':
        config_update = dict(ppo_agent_kwargs=dict(ModelCls=PPOMcpModel, model_kwargs=dict(freeze_primitives=True)),
                             sampler_kwargs=dict(env_kwargs=dict(id='TrackEnv-v0')),
                             ppo_kwargs=dict(discount=0.99, learning_rate=5e-5),
                             algo='ppo')


    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    serial_mode=args.serial_mode,
                    snapshot=snapshot,
                    config_update=config_update)
