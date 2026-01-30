import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from sim.environment import RCLineFollowerEnv
from sim.utils import create_custom_spline, get_line_point

def clear_track(env_setup):
    """Deep cleanup of all track-related items."""
    p.removeAllUserDebugItems()
    if hasattr(env_setup, 'line_ids'):
        for lid in env_setup.line_ids:
            try:
                p.removeBody(lid)
            except:
                pass
        env_setup.line_ids = []
    print("Track and visuals cleared.")

def sandbox():
    print("\n" + "="*40)
    print("      RC CAR SANDBOX V3 (KEYBOARD MODE)")
    print("="*40)
    print("SÅ HÄR BYGGER DU DIN BANA:")
    print("  PILAR      : Flytta den blå markören")
    print("  SPACE      : Sätt ut en vägpunkt vid markören")
    print("  'S'        : Starta/Stoppa körningen")
    print("  'C'        : Rensa allt och börja om")
    print("  'G'        : Ändra markens färg")
    print("  'L'        : Slå av/på skuggor")
    print("  'T'        : Växla tejpfärg (Vit/Gul)")
    print("  'Q'        : Avsluta")
    print("="*40)

    # Setup environment
    env_setup = RCLineFollowerEnv(render_mode='human')
    def make_env(): return env_setup
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order='first')

    # Load latest sophisticated model
    model_path = "line_follower_final.zip"
    if os.path.exists("./models/"):
        checkpoints = [f for f in os.listdir("./models/") if f.endswith(".zip") and "sophisticated" in f]
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join("./models/", x)), reverse=True)
            model_path = os.path.join("./models/", checkpoints[0])
    
    try:
        model = PPO.load(model_path, env=env)
        print(f"Modell laddad: {model_path}")
    except:
        print("Kunde inte ladda modellen. Se till att ha tränat först.")
        return

    waypoints = []
    simulation_running = False
    
    # Cursor state
    cursor_pos = [0.0, 0.0]
    cursor_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere_small.urdf"), [0,0,0], globalScaling=2)
    p.changeVisualShape(cursor_id, -1, rgbaColor=[0, 0.5, 1, 0.8]) # Blue transparent cursor

    # Top-down view
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[10, 0, 0])

    while True:
        keys = p.getKeyboardEvents()
        
        # 'Q'
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break
            
        # 'C'
        if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
            waypoints = []
            simulation_running = False
            clear_track(env_setup)
            cursor_pos = [0.0, 0.0]
            print("Allt rensat.")
            
        # Keyboard Cursor Movement
        move_speed = 0.2
        if p.B3G_LEFT_ARROW in keys: cursor_pos[1] += move_speed
        if p.B3G_RIGHT_ARROW in keys: cursor_pos[1] -= move_speed
        if p.B3G_UP_ARROW in keys: cursor_pos[0] += move_speed
        if p.B3G_DOWN_ARROW in keys: cursor_pos[0] -= move_speed
        
        # Update cursor visual
        p.resetBasePositionAndOrientation(cursor_id, [cursor_pos[0], cursor_pos[1], 0.1], [0,0,0,1])

        # SPACE to Add Waypoint
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
            if not simulation_running:
                if not waypoints:
                    clear_track(env_setup)
                
                # Simple constraint: must move forward in X
                if not waypoints or cursor_pos[0] > waypoints[-1][0] + 0.1:
                    new_pt = [cursor_pos[0], cursor_pos[1]]
                    waypoints.append(new_pt)
                    p.addUserDebugText(f"#{len(waypoints)}", [new_pt[0], new_pt[1], 0.3], textColorRGB=[1, 1, 0])
                    if len(waypoints) > 1:
                        p.addUserDebugLine([waypoints[-2][0], waypoints[-2][1], 0.05], 
                                         [new_pt[0], new_pt[1], 0.05], 
                                         lineColorRGB=[1, 0, 0], lineWidth=2)
                    print(f"Vägpunkt #{len(waypoints)} satt vid {new_pt}")
                else:
                    print("FEK: Flytta markören längre fram (höger) innan du sätter nästa punkt.")

        # 'S' Start/Stop
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            if not simulation_running:
                if len(waypoints) < 4:
                    print("Behöver minst 4 punkter för en bra bana!")
                else:
                    print("Bygger bana och startar...")
                    try:
                        # Hide cursor during sim
                        p.resetBasePositionAndOrientation(cursor_id, [0, 0, -10], [0,0,0,1])
                        cs, x_range = create_custom_spline(waypoints)
                        env_setup.spline = cs
                        env_setup.x_points = x_range
                        env_setup.reset()
                        obs = env.reset()
                        simulation_running = True
                    except Exception as e:
                        print(f"Fel vid banbygge: {e}")
            else:
                simulation_running = False
                p.resetBasePositionAndOrientation(cursor_id, [cursor_pos[0], cursor_pos[1], 0.1], [0,0,0,1])
                print("Simulation stoppad.")

        # Toggles
        if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
            clr = np.random.uniform(0.1, 0.4, size=3).tolist() + [1]
            p.changeVisualShape(env_setup.plane_id, -1, rgbaColor=clr)

        if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
            current = p.getDebugVisualizerCamera()[8]
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 - current)

        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            env_setup.line_color = [1, 1, 0, 1] if env_setup.line_color[1] > 0.9 and env_setup.line_color[2] > 0.9 else [1, 1, 1, 1]
            for lid in env_setup.line_ids: p.removeBody(lid)
            env_setup.line_ids = []
            env_setup._draw_line()

        if simulation_running:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            if dones[0]:
                simulation_running = False
                print(f"Mål nått! Belöning: {rewards[0]:.1f}")
            time.sleep(1/30.0)
        else:
            p.stepSimulation()
            time.sleep(1/60.0)

    env.close()

if __name__ == "__main__":
    sandbox()
