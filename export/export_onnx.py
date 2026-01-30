import os
import sys
import numpy as np

# Robust Monkeypatch for Numpy 2.x -> 1.x compatibility
if not hasattr(np, '_core'):
    import numpy.core as core
    import types
    _core = types.ModuleType('numpy._core')
    _core.numeric = core.numeric
    _core.multiarray = core.multiarray
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.numeric'] = core.numeric
    sys.modules['numpy._core.multiarray'] = core.multiarray
    print("Numpy 2.x -> 1.x Monkeypatch applied.")

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
import onnx

# Add the project root to sys.path so we can find 'training.model'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from training.model import CustomCNN


import torch.nn.functional as F

class DeterministicPPOPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        
        # V10: Reconstructed Architecture is ONNX-native.
        # No more AdaptiveAvgPool2d workarounds needed.
        pass
    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Action is dist.mode() in SB3 for deterministic output, 
        # but get_distribution(obs).distribution.mean is equivalent for PPO.
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean

def export_to_onnx(model_path, onnx_path, features_dim=512):
    # Load the model
    # V14: uses features_dim=512 and net_arch=[]
    model = PPO.load(
        model_path,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[]
        ),
    )
    
    # Fixed at 80x160 (Native resolution of simulator training)
    from gymnasium import spaces
    model.observation_space = spaces.Box(low=0, high=1, shape=(12, 80, 160), dtype=np.float32)
    print(f"Setting Observation Space to {model.observation_space.shape} for native export.")
    
    # We only need the policy (the neural network)
    policy = model.policy.to("cpu")
    policy.eval()

    export_model = DeterministicPPOPolicy(policy)
    export_model.eval()
    
    # Create dummy input based on the native observation space.
    dummy_input = th.randn(1, 12, 80, 160)
    
    # Export the model
    print(f"Exporting to {onnx_path} (Native 80px resolution)...")
    
    # 1. Clean up old files
    if os.path.exists(onnx_path): os.remove(onnx_path)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file): os.remove(data_file)
    
    # 2. Export to a temporary file first
    temp_onnx = onnx_path + ".temp"
    if os.path.exists(temp_onnx): os.remove(temp_onnx)
    
    print(f"Phase 1: Exporting to {temp_onnx}...")
    th.onnx.export(
        export_model,
        (dummy_input,),
        temp_onnx, 
        opset_version=15, 
        input_names=["input"],
        output_names=["action"],
        dynamic_axes={"input": {0: "batch_size"}, "action": {0: "batch_size"}},
    )
    
    # Phase 2: Load and Save as External Data (Splitting weights)
    print(f"Phase 2: Splitting weights into {data_file}...")
    loaded_model = onnx.load(temp_onnx)
    onnx.save_model(
        loaded_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(onnx_path) + ".data",
        size_threshold=1024, # Force split
        convert_attribute=False
    )
    
    # Cleanup temp files
    if os.path.exists(temp_onnx): os.remove(temp_onnx)
    # torch might create a .temp.data file too
    temp_data = temp_onnx + ".data"
    if os.path.exists(temp_data): os.remove(temp_data)
    
    print(f"Model exported successfully (Native 80px + External Data).")

if __name__ == "__main__":
    # Use project_root to build absolute paths to models
    # V14: Export the NEW 1.25M step checkpoint
    best_file = os.path.join(project_root, "rl_model_1250000_steps.zip")
    output_onnx = os.path.join(project_root, "line_follower.onnx")

    if os.path.exists(best_file):
        print(f"Using New V14 Checkpoint: {best_file}")
        export_to_onnx(best_file, output_onnx, features_dim=512)
    else:
        print(f"Model file not found: {best_file}")
