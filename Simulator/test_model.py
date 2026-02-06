""" test_model.py fungerar som en slutbesiktning för modellen. Skriptet skapar en testmiljö som exakt efterliknar träningsmiljön, 
inklusive bildstackningen på fyra bilder som ger modellen tidsuppfattning. 
Dess huvudsakliga uppgift är att leta rätt på den bäst tränade modellen och ladda in den med rätt hjärnarkitektur (CustomCNN). 
"""

import cv2
import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# --- MILJÖ- OCH PATH-INSTÄLLNINGAR ---
# Lägger till projektets rotmapp i Python-sökvägen för att kunna importera CustomCNN
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from sim.environment import RCLineFollowerEnv
except ImportError:
    from environment import RCLineFollowerEnv

def test():
    # Skapa simulator-miljön i visuellt läge (GUI)
    def make_env():
        return RCLineFollowerEnv(render_mode='human')
    
    # Packa in miljön med 4 bilders minne (n_stack=4)
    # Detta måste matcha hur modellen tränades för att den ska förstå rörelse
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='first')
    
    # --- LOGIK FÖR ATT HITTA MODELLFIL ---
    # Prioritetsordning: 1. best_model.zip | 2. line_follower_final.zip | 3. Senaste i /models/
    model_path = "best_model.zip"
    
    if not os.path.exists(model_path):
        model_path = "line_follower_final.zip"
        
    if not os.path.exists(model_path) and os.path.exists("./models/"):
        checkpoints = [f for f in os.listdir("./models/") if f.endswith(".zip")]
        if checkpoints:
            # Sorterar så att den nyaste filen hamnar först i listan
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join("./models/", x)), reverse=True)
            model_path = os.path.join("./models/", checkpoints[0])
    
    print(f"Laddar modell: {model_path}")
    
    # --- LADDNING AV 512 dimensioner ---
    try:
        from training.model import CustomCNN
        
        # Här tvingar vi programmet att använda 512 dimensioner för features
        model = PPO.load(model_path, env=env, policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        ))
        print("512 dimensioner laddad utan problem.")
            
    except Exception as e:
        print(f"FEL vid laddning av modell: {e}")
        print("Kontrollera att modellen verkligen är tränad med 512 dimensioner.")
        return

    # --- KÖR TESTER ---
    num_episodes = 5 # Kör 5 varv för att se hur stabil AI är
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"Startar testomgång {episode + 1}")
        
        while not done:
            # deterministic=True gör att AI:n kör så stadigt som möjligt utan att 'chansa'
            action, _ = model.predict(obs, deterministic=True)
            # Hämta den senaste bilden från stacken (den sista kanalen/bilden)
            # Vi konverterar från (Kanaler, Höjd, Bredd) till (Höjd, Bredd, Kanaler) för OpenCV
            current_frame = obs[0][-3:].transpose(1, 2, 0) 
            
            # Visa bilden i ett fönster som heter "AI Vision"
            cv2.imshow("AI Vision", cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1) # Krävs för att fönstret ska uppdateras            
            # Uppdatera simulatorn med AI:ns beslut
            obs, rewards, dones, infos = env.step(action)

            total_reward += rewards[0]
            
            # Paus på 20ms för att göra simuleringen behaglig att titta på
            time.sleep(0.02)
            
            if dones[0]:
                done = True
                print(f"Omgång klar. Total belöning: {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    test()