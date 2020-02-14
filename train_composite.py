from train import build_and_train
import torch
import argparse
from mcp_model import PiMCPModel

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
    snapshot= torch.load('data/params.pkl', map_location=torch.device('cpu'))
    if args.snapshot_file is not None:
        snapshot = torch.load(args.snapshot_file, map_location=torch.device('cpu'))
    elif args.primitives_snapshot is not None:
        snapshot = torch.load(args.primitives_snapshot, map_location=torch.device('cpu'))
        # only keep primitives
        model_snapshot_dict = snapshot['agent_state_dict']['model']
        snapshot['agent_state_dict'] = dict()
        snapshot['agent_state_dict']['model'] = PiMCPModel.remove_gating_from_snapshot(model_snapshot_dict)



    config_update = dict(agent_kwargs=dict(model_kwargs=dict(freeze_primitives=True)))

    build_and_train(slot_affinity_code=args.slot_affinity_code,
                    log_dir=log_dir,
                    run_ID=args.run_id,
                    serial_mode=args.serial_mode,
                    snapshot=snapshot,
                    config_update=config_update)
