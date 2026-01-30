import time
import numpy as np
try:
    from adafruit_servokit import ServoKit
    PCA9685_AVAILABLE = True
except ImportError:
    PCA9685_AVAILABLE = False

class RCControl:
    def __init__(self, steering_channel=0, throttle_channel=1):
        """
        RC Control via PCA9685 using standard Servo angles for maximum compatibility.
        """
        self.steering_channel = steering_channel
        self.throttle_channel = throttle_channel
        self.hardware_active = False
        
        # RC-Dockers Calibration (Brushed Motor)
        # Neutral: 145. Right: 90. Left: 180.
        # Range Right = 145-90 = 55. Range Left = 180-145 = 35.
        # V11: Maximize symmetric range for better curve handling
        self.steer_neutral = 145
        self.steer_range = 40 
        
        self.throttle_neutral = 120 # Verified STOP value for Brushed ESC
        self.throttle_max = 45      # 120 + (0.2 * 45) = 129 (Exact 'Slow Forward')
        
        if PCA9685_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16)
                # Using DEFAULT pulse width range (usually 500-2500) to match user tests
                
                self.mode = "pca9685"
                self.hardware_active = True
                print(f"Using PCA9685 (Precision Angle mode).")
                print(f"Steer Neutral: {self.steer_neutral}, Throttle Neutral: {self.throttle_neutral}")
            except Exception as e:
                print(f"PCA9685 initialization failed: {e}")
                self.hardware_active = False
        
        if not self.hardware_active:
            print("No PCA9685 found. Debug mode only.")
            self.mode = "debug"

    def arm(self):
        """
        Arms the ESC by sending a steady neutral signal.
        """
        if self.mode == "pca9685":
            print(f"Holding neutral for ESC arming ({self.throttle_neutral})...")
            self.kit.servo[self.throttle_channel].angle = self.throttle_neutral
            time.sleep(3)
            print("ESC should be ARMED.")
        else:
            print("Debug mode: Arming skipped.")

    def set_steering(self, action):
        """
        Action: [-1, 1] 
        Hardware: Neutral 145, Left 180 (+35), Right 90 (-55)
        
        V9 SYMMETRY: Using 35 for BOTH sides to match simulator training.
        """
        if self.mode == "pca9685":
            if action >= 0:
                # Steering Left: Map [0, 1] to [145, 180]
                angle = self.steer_neutral + (action * self.steer_range) # 40
            else:
                # Steering Right: Map [-1, 0] to [110, 145] (Using symmetric 40)
                angle = self.steer_neutral + (action * self.steer_range) # 40
            
            self.kit.servo[self.steering_channel].angle = max(0, min(180, angle))
        else:
            print(f"  [STEER] -> {action:5.2f}", end="\r")

    def set_throttle(self, action):
        """
        Action: [0, 1] 
        """
        if self.mode == "pca9685":
            # Map [0, 1] to throttle values starting from neutral
            angle = self.throttle_neutral + (action * self.throttle_max)
            self.kit.servo[self.throttle_channel].angle = max(0, min(180, angle))
        else:
            pass

    def stop(self):
        if self.mode == "pca9685":
            self.kit.servo[self.steering_channel].angle = self.steer_neutral
            self.kit.servo[self.throttle_channel].angle = self.throttle_neutral
            print("\nHardware stop (Neutral sent).")
