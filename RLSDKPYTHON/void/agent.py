# In agent.py

import os
import numpy as np
import torch
from .discrete_policy import DiscreteFF
from .your_act import LookupAction

# CustomObs1v1 - Ultra optimisé 1v1 avec ball prediction simple
# Total: 9 + 6 + 18 + 7 + 6 + 8 + 4 + 2 + 34 + 8 + 21 + 38 = 161 features
OBS_SIZE = 147  # Match EXACT du nouveau CustomObs1v1
SHARED_LAYER_SIZES = [2048, 1024, 1024]
POLICY_LAYER_SIZES = [2048, 1024, 1024, 1024]

class Agent:
    def __init__(self):
        self.action_parser = LookupAction()
        self.num_actions = len(self.action_parser._lookup_table)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        
        device = torch.device("cpu")
        
        # 🔍 DEBUG: Check actual model input size
        print(f"🔍 Creating policy with OBS_SIZE = {OBS_SIZE}")
        self.policy = DiscreteFF(OBS_SIZE, self.num_actions, SHARED_LAYER_SIZES, POLICY_LAYER_SIZES, device)
        
        shared_head_path = os.path.join(cur_dir, "SHARED_HEAD.LT")
        policy_path = os.path.join(cur_dir, "POLICY.LT")

        # Load the models
        shared_model = torch.load(shared_head_path, map_location=device, weights_only=False)
        policy_model = torch.load(policy_path, map_location=device, weights_only=False)
        
        # 🔍 DEBUG: Check what input size the loaded model expects
        if hasattr(shared_model, 'state_dict'):
            first_layer_key = list(shared_model.state_dict().keys())[0]
            first_layer_weights = shared_model.state_dict()[first_layer_key]
            if len(first_layer_weights.shape) == 2:
                actual_input_size = first_layer_weights.shape[1]
                print(f"🔍 Loaded model expects input size: {actual_input_size}")
                if actual_input_size != OBS_SIZE:
                    print(f"⚠️  WARNING: Model expects {actual_input_size} but OBS_SIZE is {OBS_SIZE}!")
                    print(f"⚠️  Please update OBS_SIZE in agent.py to {actual_input_size}")
        
        self.policy.shared_head.load_state_dict(shared_model.state_dict())
        self.policy.policy.load_state_dict(policy_model.state_dict())

        torch.set_num_threads(1)

    def act(self, state):
        with torch.no_grad():
            action_idx, probs = self.policy.get_action(state, True)
        
        action = np.array(self.action_parser.parse_actions([action_idx]))
        if len(action.shape) == 2:
            if action.shape[0] == 1:
                action = action[0]
        
        if len(action.shape) != 1:
            raise Exception("Invalid action:", action)
        
        return action