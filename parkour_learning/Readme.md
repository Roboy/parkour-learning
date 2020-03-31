This is an OpenAI gym environment for training primitive and compositional policies.
When this module is imported, it registers 3 environments:  
'HumanoidPrimitivePretraining-v0': This is the training environment for the primitives. The goal is to control the humanoid so that it moves similar to the humanoid from motion capture data. Reward = joint_angle_similarity + joint_velocity_similarity + end_effector_position_similarity + center_of_mass_similarity
'TrackEnv-v0': The track environment is used to train the compositional policiy. Reward is given for velocity towards
the end of the track.   

Observations for both environments are a depth map from the point of view of the humanoid in direction of the target and the joint states. Actions are inputs to pd controller of the joints.

test_env.py can be used to test these two environments.
