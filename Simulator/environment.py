""" Simuleringsmiljö fungerar som en digital träning. 
Genom att använda en realistisk fysikmodell och slumpmässigt genererade banor tvingas bilen att lära sig exakt hur den ska navigera kurvor. 
Miljön är medvetet fylld med visuella störningar som reflexer och skuggor, som tränar modellen att ignorera detaljer och 
fokusera på linjen oavsett ljusförhållanden.

Tekniska begränsningar simuleras som fördröjningar och mekaniskt glapp, vilket skapar en stabil och förutseende körstil. 
Allt styrs av ett belöningssystem som belönar mjukhet och centrering framför ryckiga rörelser. 
Detta säkerställer att AI inte bara presterar i den perfekta digitala världen, utan även levererar en stabil körning."""

import os
import sys
# Lägg till projektets rotmapp i sys.path för att kunna importera utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import collections

# Försök importera simuleringsverktyg; hanterar olika mappstrukturer
try:
    from sim.utils import generate_random_spline, preprocess_image, get_line_point
except ImportError:
    from utils import generate_random_spline, preprocess_image, get_line_point

class RCLineFollowerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(RCLineFollowerEnv, self).__init__()
        self.render_mode = render_mode
        
        # Action space: AI:n ger ett värde mellan -1 (höger) och 1 (vänster) för styrning
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: Bilddata i formatet (Kanaler, Höjd, Bredd) -> (3, 80, 160)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 80, 160), dtype=np.float32)
        
        # Anslut till PyBullet-fysikmotorn (GUI för visuell kontroll, DIRECT för snabb träning)
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Miljöparametrar för "Digital Twin V4"
        self.max_steps = 2500
        self.speed = 1.0 
        self.dt = 1./20. # 20 Hz styrfrekvens
        
        self.line_width = 0.05 # 5cm bred tejp
        self.line_ids = []
        self.distractor_ids = [] # Lista för att hålla koll på möbler/hinder
        
        # Buffert för att simulera eftersläpning (latency) i styrningen
        self.steering_queue = collections.deque(maxlen=3) # Max 150ms delay
        self.last_steering = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        """ Återställer miljön till startläget """
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self.line_ids = []
        self.distractor_ids = []
        
        # 1. Domain Randomization: Markplan (Fokus på ljusa/vita golv)
        self.plane_id = p.loadURDF("plane.urdf")
        # Slumpar golvfärg mot ljusa nyanser (0.6 - 1.0)
        ground_color = np.random.uniform(0.6, 1.0, size=3).tolist() + [1.0]
        specular = [np.random.uniform(0.5, 1.0)] * 3 # Slumpar golvets glansighet
        p.changeVisualShape(self.plane_id, -1, 
                            rgbaColor=ground_color,
                            specularColor=specular)
        
        # 2. Ban-generering via slumpmässig spline
        self.spline, self.x_points = generate_random_spline(length=30.0, num_points=10, complexity=3.5)
        self.line_progress = 0.0
        
        # 3. Domain Randomization: Linjen (Gul tejp)
        self.line_color = [1.0, 0.8, 0.0, 1.0] 
        self.line_width = 0.05 
        
        self._draw_line()
        
        # 4. Aktivera skuggor för realism
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        
        # 5. Skapa bilmodellen
        self.car_id = self._create_car()
        
        # 6. Domain Randomization: Hinder (Lådor/möbler)
        # Placerar ut 15 objekt som AI:n måste lära sig att ignorera
        for _ in range(15):
            obj_pos = [np.random.uniform(0, 30), np.random.uniform(-3, 3), 0.1]
            # Kontrollera att hindret inte hamnar mitt på spåret
            if abs(obj_pos[1] - self.spline(obj_pos[0])) < 0.4:
                continue
            
            size = [np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5)]
            color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
            v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
            c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            d_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id, basePosition=obj_pos)
            self.distractor_ids.append(d_id)

        # Startposition med lite brus (noise) för att öka robustheten
        start_point = get_line_point(self.spline, self.x_points, 0.0)
        start_y_offset = np.random.uniform(-0.06, 0.06)
        p.resetBasePositionAndOrientation(self.car_id, [start_point[0], start_point[1] + start_y_offset, 0.02], [0, 0, 0, 1])
        
        # Återställ styrnings-kön
        self.steering_queue.clear()
        for _ in range(3): self.steering_queue.append(0.0)
        
        self.current_step = 0
        self.last_steering = 0.0
        return self._get_observation(), {}

    def _create_car(self):
        """ Skapar den fysiska representationen av bilen (23cm hjulbas) """
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.115, 0.05, 0.02], rgbaColor=[1, 0, 0, 1])
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.115, 0.05, 0.02])
        car_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0.05])
        return car_id

    def _draw_line(self):
        """ Ritar ut spåret i simulatorn som en serie segment """
        num_segments = 120
        points = []
        for t in np.linspace(0, 1, num_segments + 1):
            pt = get_line_point(self.spline, self.x_points, t)
            points.append(pt)
            
        track_width = 0.4 # 40cm bred bana totalt
        
        for i in range(num_segments):
            p1, p2 = np.array(points[i]), np.array(points[i+1])
            center = (p1 + p2) / 2.0
            diff = p2 - p1
            length = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            normal = np.array([-np.sin(angle), np.cos(angle)])
            
            # Ritar linjer på båda sidor om mittpunkten
            for side in [-1, 1]:
                offset_pos = center + side * (track_width / 2.0) * normal
                visual_id = p.createVisualShape(p.GEOM_BOX, 
                                                halfExtents=[length/2.1, self.line_width/2.0, 0.0001],
                                                rgbaColor=self.line_color)
                line_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, 
                                            basePosition=[offset_pos[0], offset_pos[1], 0.001],
                                            baseOrientation=p.getQuaternionFromEuler([0, 0, angle]))
                self.line_ids.append(line_id)

    def step(self, action):
        """ Utför ett tidsteg (20Hz) """
        # 1. Simulering av fördröjning (Latency)
        self.steering_queue.append(action[0])
        # Varierar fördröjning mellan 1-3 tidssteg för att hantera jitter
        lag = np.random.choice([1, 2, 3])
        steering = self.steering_queue[-lag]
        
        # 2. Simulering av mekaniskt glapp (Jitter) i styrstag
        steering += np.random.normal(0, 0.03)
        steering = np.clip(steering, -1, 1)

        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        # Kinematisk cykelmodell (Bicycle Model)
        wheelbase = 0.23
        max_wheel_angle = 0.52 # ~30 grader
        delta = steering * max_wheel_angle
        
        vx, vy = self.speed * np.cos(yaw), self.speed * np.sin(yaw)
        omega = (self.speed * np.tan(delta)) / wheelbase
        
        # Uppdatera bilens hastighet och kör simulatorn
        p.resetBaseVelocity(self.car_id, [vx, vy, 0], [0, 0, omega])
        for _ in range(int(self.dt / (1./240.))): p.stepSimulation()
            
        self.current_step += 1
        observation = self._get_observation()
        reward, terminated = self._compute_reward(pos, yaw, steering)
        truncated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        """ Skapar kamerabilden och lägger på avancerade visuella störningar """
        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        # Simulera kameraskakningar (Pitch/Yaw noise)
        pitch_noise = np.random.uniform(-0.02, 0.02)
        yaw_noise = np.random.uniform(-0.02, 0.02)
        
        cam_yaw = p.getEulerFromQuaternion(ori)[2] + yaw_noise
        # Placera kameran ovanför framaxeln
        cam_pos = [pos[0] + 0.115 * np.cos(cam_yaw), pos[1] + 0.115 * np.sin(cam_yaw), 0.23]
        target_pos = [pos[0] + 0.615 * np.cos(cam_yaw), pos[1] + 0.615 * np.sin(cam_yaw), 0.0 + pitch_noise]
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(65, 2.0, 0.01, 10) # Matchar vidvinkelkamera
        
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view_matrix, proj_matrix)
        rgb = np.array(rgb).reshape(240, 320, 4)
        processed = preprocess_image(rgb, target_size=(160, 80), grayscale=False)
        
        # --- AVANCERAD LJUSSÄTTNING (STÖRNINGAR) ---
        
        # 1. Reflexer (Specular Hotspots) från t.ex. taklampor
        if np.random.rand() > 0.15:
            for _ in range(np.random.randint(1, 3)):
                glare_x = np.random.randint(0, 160)
                glare_y = np.random.randint(0, 80)
                channel = np.ascontiguousarray(processed[0])
                axes = (np.random.randint(15, 35), np.random.randint(8, 15))
                angle = np.random.randint(0, 180)
                cv2.ellipse(channel, (glare_x, glare_y), axes, angle, 0, 360, (0.9, 0.9, 0.9), -1)
                processed[0] = channel
        
        # 2. Skuggor från fönster/persienner
        if np.random.rand() > 0.4:
            num_bars = np.random.randint(3, 8)
            bar_width = np.random.randint(5, 15)
            spacing = np.random.randint(10, 30)
            offset = np.random.randint(0, 160)
            for i in range(num_bars):
                x_start = (offset + i * spacing) % 160
                processed[:, :, x_start:x_start+bar_width] *= np.random.uniform(0.4, 0.6)

        # 3. Motion Blur (Rörelseoskärpa)
        if np.random.rand() > 0.4:
            k = np.random.choice([3, 5, 7])
            for i in range(3):
                kernel = np.zeros((k, k))
                if np.random.rand() > 0.5:
                    kernel[int((k-1)/2), :] = np.ones(k) # Horisontell oskärpa
                else:
                    kernel[:, int((k-1)/2)] = np.ones(k) # Vertikal oskärpa
                kernel /= k
                channel = np.ascontiguousarray(processed[i])
                processed[i] = cv2.filter2D(channel, -1, kernel)

        # 4. Färgtemperatur (Varmt vs Kallt ljus)
        if np.random.rand() > 0.4:
            processed[0] *= np.random.uniform(0.9, 1.1) # Röd-bias (Varmt)
            processed[2] *= np.random.uniform(0.9, 1.1) # Blå-bias (Kallt)
            
        # 5. Global ljusstyrka, kontrast och kornigt brus
        alpha = np.random.uniform(0.4, 1.6) 
        beta = np.random.uniform(-0.3, 0.3) 
        processed = np.clip(alpha * processed + beta, 0, 1)
        
        noise = np.random.normal(0, 0.05, processed.shape).astype(np.float32)
        processed = np.clip(processed + noise, 0, 1)
        
        return processed

    def _compute_reward(self, car_pos, car_yaw, steering):
        """ Beräknar belöning baserat på position och stabilitet """
        # Straff om bilen kör utanför banans gränser
        if car_pos[0] < -0.5 or car_pos[0] > self.x_points[-1] + 0.5: return -20.0, True
        target_y = self.spline(car_pos[0])
        dist = abs(car_pos[1] - target_y)
        if dist > 0.4: return -20.0, True # Avbryt om avståndet är > 40cm
            
        reward = 0.1 
        # Bonus för att vara nära mitten (Gausisk kurva)
        centering_reward = 3.0 * np.exp(-(dist**2) / (2 * 0.1**2))
        
        # Belöna rätt vinkel i förhållande till banans kurva
        slope = self.spline.derivative()(car_pos[0])
        target_yaw = np.arctan(slope)
        angle_diff = (target_yaw - car_yaw + np.pi) % (2 * np.pi) - np.pi
        angle_reward = 1.0 * np.exp(-(angle_diff**2) / (2 * 0.2**2))
        
        # Straff för hög styransträngning
        steering_penalty = -0.1 * abs(steering) 
        
        # ANTI-WOBBLE: Straffar snabba styrförändringar för att få mjuk körning
        steering_diff = abs(steering - self.last_steering)
        smoothing_penalty = -2.0 * (steering_diff ** 2) 
        
        # Bonus för att hålla sig extremt centrerad (inom 5cm)
        persistence_bonus = 0.0
        if dist < 0.05: 
            persistence_bonus = 0.5
        self.last_steering = steering
        
        return reward + centering_reward + angle_reward + steering_penalty + smoothing_penalty + persistence_bonus, False

    def close(self): p.disconnect(self.physics_client)

if __name__ == "__main__":
    # Testkör miljön med slumpmässiga rörelser om filen körs direkt
    env = RCLineFollowerEnv(render_mode='human')
    obs, info = env.reset()
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated: obs, info = env.reset()
        time.sleep(0.02)
    env.close()