# Parkour learning repository

The goal of this project is to train a humanoid robot in simulation on an obstacle course. 
This is done by first training primitive policies on mimicking motion
capture data from real humans (see https://xbpeng.github.io/projects/DeepMimic/index.html).
We use the multiplicate compositional policies scheme to combine these policies (https://xbpeng.github.io/projects/MCP/index.html).

## Train policies
train.py contains code to train the primitives. train_composite.py is a script that trains the composite policy using
the pretrained primitives.

## visualize trained policies
### Policies for the TrackEnv
There are 3 policies trained with the Soft Actor Critic in classical RL style
To visualize these policies run:  
`python simulate_policy.py --algo sac --path policies_archive/sac_vision_track.pkl --env TrackEnv-v0`  
or  
`python simulate_policy.py --algo sac --path policies_archive/sac_vision_track.pkl --env TrackEnv-v0`  
or  
`python simulate_policy.py --algo sac --path policies_archive/sac_vision_track.pkl --env TrackEnv-v0`  

There are also policies trained using the multiplicative compositional policy scheme. To visualize the composed policy
trained on the TrackEnv run (This one is however not working so well):  
python simulate_policy.py --algo ppo --path policies_archive/ppo_mcp_composite.pkl --env TrackEnv-v0

You can also view some pretrained MCP policies. They were each only trained on one motion
capture file because otherwise it learned very poorly. To visualize the one only trained
on the run.txt mocap file, you have to edit the parkour_learning/gym_env/primitive_pretraining_env.py file.
Make sure that the class attribute mocap_files is set to ['run.txt']. Then execute:  
 python simulate_policy.py --algo ppo --path policies_archive/ppo_mcp_pretraining_run.pkl
If you want to view the walking policy set mocap_files to ['walk.txt'] and execute (This one is not very good):  
 python simulate_policy.py --algo ppo --path policies_archive/ppo_mcp_pretraining_walking.pkl
If you want to view the vaulting policy set mocap_files to ['vaulting.txt'] and execute (This one is not very good):  
 python simulate_policy.py --algo ppo --path policies_archive/ppo_mcp_pretraining_vaulting.pkl

