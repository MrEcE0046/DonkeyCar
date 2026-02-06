""" train.py ansvarar för att driva inlärningsprocessen genom Reinforcement Learning. 
Skriptet använder algoritmen PPO (Proximal Policy Optimization) och kör 16 simuleringar parallellt för att samla in data extremt snabbt. 
Den kopplar samman en CustomCNN-hjärna med simulatorn och använder ett belöningssystem för att stegvis förbättra bilens förmåga att hålla sig på banan. 
Genom att använda automatiska sparfiler säkerställer skriptet att den bästa versionen av modellen alltid sparas, även om träningen skulle avbrytas."""

import os
import sys

# --- PATH SETUP ---
# Lägger till rotmappen i sys.path så att Python hittar sim/ och training/ 
# oavsett varifrån skriptet körs.
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
    # --- PARALLELLISERING ---
    # n_envs=16 gör att vi kör 16 bilar samtidigt i bakgrunden.
    # Detta snabbar upp träningen enormt och ger stabilare inlärning.
    n_envs = 16

    # Skapa träningsmiljön och stacka 4 bilder (n_stack=4)
    # n_stack ger modellen 'syn på tid' så den ser rörelse.
    env = make_vec_env(RCLineFollowerEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    # En separat miljö för att utvärdera hur bra modellen faktiskt är.
    eval_env = make_vec_env(RCLineFollowerEnv, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="first")

    # --- MODELLKONFIGURATION ---
    # Här kopplar vi på din CustomCNN med 512 dimensioner.
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[], # Tom lista betyder att vi går direkt från CNN till beslut
    )

    # Initiera PPO-algoritmen
    # learning_rate=0.00001 (låg fart) gör att modellen inte glömmer vad den lärt sig.
    # device="cuda" använder ditt grafikkort för blixtsnabb beräkning.
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.00001,
        n_steps=2048,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99, # Framtidsfokus: hur mycket vi värderar framtida belöningar
        ent_coef=0.005, # Entropi: tvingar modellen att utforska och inte fastna i ett spår
        tensorboard_log="./tensorboard_logs/",
        device="cuda",
    )

    # --- AUTOMATISK SPARNING (CALLBACKS) ---
    # EvalCallback sparar den modell som faktiskt kör bäst, inte bara den senaste.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/",
        eval_freq=max(1, 50000 // n_envs),
        deterministic=True,
        render=False,
    )

    # CheckpointCallback sparar med jämna mellanrum som en back-up.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 250000 // n_envs),
        save_path="./models/checkpoints/",
        name_prefix="rl_model",
    )

    # --- TRÄNINGSSTART ---
    # total_timesteps=5 000 000 innebär att AI får se 5 miljoner bilder.
    print("----------------- Starting Training -----------------")
    model.learn(
        total_timesteps=5000000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Spara den slutgiltiga modellen
    model.save("./models/line_follower_final")
    print("Training complete. Best model is in ./models/best_model/best_model.zip")

if __name__ == "__main__":
    train()