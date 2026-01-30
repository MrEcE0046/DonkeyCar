import os
import sys
# Add project root to sys.path
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
try:
    from sim.utils import generate_random_spline, preprocess_image, get_line_point
except ImportError:
    from utils import generate_random_spline, preprocess_image, get_line_point

import collections
class RCLineFollowerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(RCLineFollowerEnv, self).__init__()
        self.render_mode = render_mode
        
        # Action space: [-1, 1] for steering
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: 3x80x160 RGB image (C, H, W)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 80, 160), dtype=np.float32)
        
        # Connect to PyBullet
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Environment parameters (V4 Digital Twin)
        self.max_steps = 2500
        self.speed = 1.0 
        self.dt = 1./20. # 20 Hz
        
        self.line_width = 0.05 # 5cm wide tape
        self.line_ids = []
        self.distractor_ids = [] # To clear furniture-like objects
        
        # V5 Dirty Reality Buffers
        self.steering_queue = collections.deque(maxlen=3) # Max 150ms delay
        self.last_steering = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self.line_ids = []
        self.distractor_ids = []
        
        # 1. Domain Randomization: Plane (Focus on Bright/White floors)
        self.plane_id = p.loadURDF("plane.urdf")
        # Skewed towards lighter colors (0.6 - 1.0) to simulate bright floors
        ground_color = np.random.uniform(0.6, 1.0, size=3).tolist() + [1.0]
        specular = [np.random.uniform(0.5, 1.0)] * 3 # Handle shininess
        p.changeVisualShape(self.plane_id, -1, 
                            rgbaColor=ground_color,
                            specularColor=specular)
        
        # 2. Track Generation
        self.spline, self.x_points = generate_random_spline(length=30.0, num_points=10, complexity=3.5)
        self.line_progress = 0.0
        
        # 3. Domain Randomization: Line
        # Matches user's Yellow tape with randomization for robustness
        self.line_color = [1.0, 0.8, 0.0, 1.0] # Yellow
        self.line_width = 0.05 # 5cm wide tape
        
        self._draw_line()
        
        # 4. Domain Randomization: Lighting
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        
        # 5. Create Car
        self.car_id = self._create_car()
        
        # 6. Domain Randomization: Distractor Objects (Boxes, furniture-like)
        for _ in range(15):
            obj_pos = [np.random.uniform(0, 30), np.random.uniform(-3, 3), 0.1]
            # Ensure it's not directly on the track roughly
            if abs(obj_pos[1] - self.spline(obj_pos[0])) < 0.4:
                continue
            
            size = [np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5)]
            color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
            v_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
            c_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            d_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c_id, baseVisualShapeIndex=v_id, basePosition=obj_pos)
            self.distractor_ids.append(d_id)

        # Initial position with noise
        start_point = get_line_point(self.spline, self.x_points, 0.0)
        start_y_offset = np.random.uniform(-0.06, 0.06)
        p.resetBasePositionAndOrientation(self.car_id, [start_point[0], start_point[1] + start_y_offset, 0.02], [0, 0, 0, 1])
        
        # 6. Domain Randomization: V5 Buffers
        self.steering_queue.clear()
        for _ in range(3): self.steering_queue.append(0.0)
        
        self.current_step = 0
        self.last_steering = 0.0
        return self._get_observation(), {}

    def _create_car(self):
        # 23cm wheelbase = 0.115m halfExtents
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.115, 0.05, 0.02], rgbaColor=[1, 0, 0, 1])
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.115, 0.05, 0.02])
        car_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0.05])
        return car_id

    def _draw_line(self):
        num_segments = 120
        points = []
        for t in np.linspace(0, 1, num_segments + 1):
            pt = get_line_point(self.spline, self.x_points, t)
            points.append(pt)
            
        # 40cm wide track (20cm offset each side)
        track_width = 0.4 
        
        for i in range(num_segments):
            p1, p2 = np.array(points[i]), np.array(points[i+1])
            center = (p1 + p2) / 2.0
            diff = p2 - p1
            length = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            
            # Normal vector to the segment
            normal = np.array([-np.sin(angle), np.cos(angle)])
            
            # Draw Left and Right boundaries
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
        # 1. Latency simulation (IMMUNE TO WIFI/CPU LAGG)
        self.steering_queue.append(action[0])
        # Varied lag [1, 3] frames to make model robust to timing jitter
        lag = np.random.choice([1, 2, 3])
        steering = self.steering_queue[-lag]
        
        # 2. Mechanical Slack (Jitter)
        # Added +/- 0.03 noise to simulate loose steering rods
        steering += np.random.normal(0, 0.03)
        steering = np.clip(steering, -1, 1)

        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        # Kinematic Bicycle Model (Digital Twin V4)
        wheelbase = 0.23
        max_wheel_angle = 0.52 # ~30 degrees
        delta = steering * max_wheel_angle
        
        vx, vy = self.speed * np.cos(yaw), self.speed * np.sin(yaw)
        omega = (self.speed * np.tan(delta)) / wheelbase
        
        p.resetBaseVelocity(self.car_id, [vx, vy, 0], [0, 0, omega])
        for _ in range(int(self.dt / (1./240.))): p.stepSimulation()
            
        self.current_step += 1
        observation = self._get_observation()
        reward, terminated = self._compute_reward(pos, yaw, steering)
        truncated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.car_id)
        # Cam angle noise
        pitch_noise = np.random.uniform(-0.02, 0.02)
        yaw_noise = np.random.uniform(-0.02, 0.02)
        
        cam_yaw = p.getEulerFromQuaternion(ori)[2] + yaw_noise
        # Offset 0.115 puts camera directly above the front axle
        cam_pos = [pos[0] + 0.115 * np.cos(cam_yaw), pos[1] + 0.115 * np.sin(cam_yaw), 0.23]
        target_pos = [pos[0] + 0.615 * np.cos(cam_yaw), pos[1] + 0.615 * np.sin(cam_yaw), 0.0 + pitch_noise]
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        # FOV 65 matches the Pi Camera 3 Wide perspective better after vertical crop
        proj_matrix = p.computeProjectionMatrixFOV(65, 2.0, 0.01, 10)
        
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view_matrix, proj_matrix)
        rgb = np.array(rgb).reshape(240, 320, 4)
        processed = preprocess_image(rgb, target_size=(160, 80), grayscale=False)
        
        # 1. Advanced Lighting (Glares, Specular Hotspots, and Shadow Strips)
        
        # Specular Hotspots (Sunlight reflections on shiny floors)
        # REDUCED complexity for CPU speedup: Fewer loops, faster ops
        if np.random.rand() > 0.15:
            for _ in range(np.random.randint(1, 3)): # Max 2 hotspots
                glare_x = np.random.randint(0, 160)
                glare_y = np.random.randint(0, 80)
                # Elliptical hotspots with high intensity core
                # We ensure contiguous slice to prevent OpenCV layout errors
                channel = np.ascontiguousarray(processed[0])
                axes = (np.random.randint(15, 35), np.random.randint(8, 15))
                angle = np.random.randint(0, 180)
                cv2.ellipse(channel, (glare_x, glare_y), axes, angle, 0, 360, (0.9, 0.9, 0.9), -1)
                processed[0] = channel
        
        # 2. Window-Blind Shadows (Alternating light/dark bars)
        if np.random.rand() > 0.4:
            num_bars = np.random.randint(3, 8)
            bar_width = np.random.randint(5, 15)
            spacing = np.random.randint(10, 30)
            offset = np.random.randint(0, 160)
            for i in range(num_bars):
                x_start = (offset + i * spacing) % 160
                processed[:, :, x_start:x_start+bar_width] *= np.random.uniform(0.4, 0.6)

        # 3. Dynamic Motion Blur (Based on car's turn rate/speed)
        # We simulate the shutter speed of the Pi Camera
        if np.random.rand() > 0.4:
            # Kernel size based on 'shakiness'
            k = np.random.choice([3, 5, 7])
            for i in range(3):
                # Apply blur in a random direction to simulate motion
                kernel = np.zeros((k, k))
                if np.random.rand() > 0.5:
                    kernel[int((k-1)/2), :] = np.ones(k) # Horizontal blur
                else:
                    kernel[:, int((k-1)/2)] = np.ones(k) # Vertical blur
                kernel /= k
                # V12: Fix OpenCV layout error by ensuring contiguous slice
                channel = np.ascontiguousarray(processed[i])
                processed[i] = cv2.filter2D(channel, -1, kernel)

        # 4. Hue & Saturation Shifts (Light Temperature)
        if np.random.rand() > 0.4:
            # Shift colors slightly to handle Warm LED vs Cold Daylight
            # Since processed is [3, 80, 160], we convert to HLS-like space
            processed[0] *= np.random.uniform(0.9, 1.1) # R-bias (Warm)
            processed[2] *= np.random.uniform(0.9, 1.1) # B-bias (Cool)
            
        # 5. Global Brightness/Contrast/Noise
        alpha = np.random.uniform(0.4, 1.6) 
        beta = np.random.uniform(-0.3, 0.3) 
        processed = np.clip(alpha * processed + beta, 0, 1)
        
        noise = np.random.normal(0, 0.05, processed.shape).astype(np.float32)
        processed = np.clip(processed + noise, 0, 1)
        
        return processed

    def _compute_reward(self, car_pos, car_yaw, steering):
        if car_pos[0] < -0.5 or car_pos[0] > self.x_points[-1] + 0.5: return -20.0, True
        target_y = self.spline(car_pos[0])
        dist = abs(car_pos[1] - target_y)
        if dist > 0.4: return -20.0, True
            
        reward = 0.1 
        centering_reward = 3.0 * np.exp(-(dist**2) / (2 * 0.1**2))
        
        slope = self.spline.derivative()(car_pos[0])
        target_yaw = np.arctan(slope)
        angle_diff = (target_yaw - car_yaw + np.pi) % (2 * np.pi) - np.pi
        angle_reward = 1.0 * np.exp(-(angle_diff**2) / (2 * 0.2**2))
        
        steering_penalty = -0.1 * abs(steering) # Base effort penalty
        
        # ANTI-WOBBLE: Heavy penalty for rapid steering oscillations
        steering_diff = abs(steering - self.last_steering)
        smoothing_penalty = -2.0 * (steering_diff ** 2) # Reduced for stability
        
        # Persistence Bonus: High reward for staying near center for consecutive steps
        persistence_bonus = 0.0
        if dist < 0.05: # Within 5cm of center
            persistence_bonus = 0.5
        self.last_steering = steering
        
        return reward + centering_reward + angle_reward + steering_penalty + smoothing_penalty + persistence_bonus, False

    def close(self): p.disconnect(self.physics_client)

if __name__ == "__main__":
    env = RCLineFollowerEnv(render_mode='human')
    obs, info = env.reset()
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated: obs, info = env.reset()
        time.sleep(0.02)
    env.close()
