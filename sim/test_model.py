import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Add the project root to sys.path so SB3 can find the 'training' module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from sim.environment import RCLineFollowerEnv
except ImportError:
    from environment import RCLineFollowerEnv

def test():
    # Create the environment in human (GUI) mode
    # For testing, we need to wrap it the same way as in training
    def make_env():
        return RCLineFollowerEnv(render_mode='human')
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='first')
    
    # Load the trained model
    # Priority: 1. best_model.zip (Root) 2. line_follower_final.zip 3. Models subdir
    model_path = "best_model.zip"
    
    if not os.path.exists(model_path):
        model_path = "line_follower_final.zip"
        
    if not os.path.exists(model_path) and os.path.exists("./models/"):
        checkpoints = [f for f in os.listdir("./models/") if f.endswith(".zip")]
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join("./models/", x)), reverse=True)
            model_path = os.path.join("./models/", checkpoints[0])
    
    print(f"Targeting model: {model_path}")
    
    try:
        # V12: Dual-Line model uses CustomCNN with features_dim=512
        # V11/10: Used features_dim=128
        # We try 512 first for V12, then fall back
        from training.model import CustomCNN
        
        try:
            model = PPO.load(model_path, env=env, policy_kwargs=dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),
            ))
            print("Successfully loaded V12 architecture (dim=512)")
        except Exception:
            print("Failed V12 load, falling back to V11 architecture (dim=128)...")
            model = PPO.load(model_path, env=env, policy_kwargs=dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
            ))
            print("Successfully loaded V11 architecture (dim=128)")
            
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return

    # Run a few test episodes
    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"Starting Episode {episode + 1}")
        
        while not done:
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment (VecEnv returns done as a boolean array)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            
            # Add a small sleep to make it watchable
            time.sleep(0.02)
            
            if dones[0]:
                done = True
                print(f"Episode finished. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test()
