import os
import sys

# Add project root to sys.path to allow sibling imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sim.environment import RCLineFollowerEnv
from training.model import CustomCNN


def train():
    n_envs = 16

    env = make_vec_env(RCLineFollowerEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    eval_env = make_vec_env(RCLineFollowerEnv, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="first")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[],
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.00001,
        n_steps=2048,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.005,
        tensorboard_log="./tensorboard_logs/",
        device="cuda",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/",
        eval_freq=max(1, 50000 // n_envs),
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 250000 // n_envs),
        save_path="./models/checkpoints/",
        name_prefix="rl_model",
    )

    print("----------------- Starting Training -----------------")
    model.learn(
        total_timesteps=5000000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save("./models/line_follower_final")
    print("Training complete. Best model is in ./models/best_model/best_model.zip")


if __name__ == "__main__":
    train()