import cv2
import numpy as np
import threading
import time
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class Camera:
    def __init__(self, resolution=(640, 480), target_size=(160, 80)):
        self.target_size = target_size
        self.resolution = resolution
        
        # V9: Color & Exposure Overrides
        # Set to True if the yellow line looks Cyan/Blue in debug images
        self.ENABLE_BGR_FLIP = False 
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        if PICAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                # Use VIDEO configuration for continuous capture stability
                config = self.picam2.create_video_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
                
                # V9.1: Restore Auto Exposure
                # Fixed exposure was too dark for indoor testing.
                self.picam2.set_controls({
                    "AeEnable": True
                })
                
                self.picam2.start()
                self.use_picamera = True
                print("Camera initialized: Picamera2 (RGB888 Video Mode)")
                # CRITICAL: Wait for hardware to fully sync before starting thread
                time.sleep(2.0)
            except Exception as e:
                print(f"Failed to initialize Picamera2: {e}. Falling back to OpenCV.")
                self.use_picamera = False
        else:
            self.use_picamera = False

        if not self.use_picamera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("Camera initialized: OpenCV")

        # Start background capture thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while not self.stopped:
            try:
                if self.use_picamera:
                    # capture_array is synchronous but called in background thread
                    f = self.picam2.capture_array()
                else:
                    ret, f = self.cap.read()
                    if not ret: 
                        time.sleep(0.01)
                        continue
                
                if f is not None:
                    with self.lock:
                        self.frame = f
            except Exception as e:
                # Log error but keep thread alive
                print(f"Capture thread error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame

    def preprocess(self, frame):
        """
        V3 Hyper-Turbo: Input is RGB888. 
        """
        if frame is None: return None
        
        # 1. Fast Crop: Picamera2 is already RGB. 
        # Match model exactly 2:1 ratio (with updated 85px trick).
        h, w = frame.shape[:2]
        crop_top = 160 # Bottom 2/3 of 480
        crop_bottom = 480
        
        # Remove BGR flip (::-1) and use adjusted crop
        crop = frame[crop_top:crop_bottom, :]
        
        if self.ENABLE_BGR_FLIP:
            # V9 Color Fix: If yellow looks cyan, flip BGR to RGB (or vice versa)
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        
        # 2. Fast Resize
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. Transpose & Normalize
        processed = resized.transpose(2, 0, 1).astype(np.float32)
        processed *= (1.0 / 255.0)
        
        return np.expand_dims(processed, axis=0)

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.use_picamera:
            try:
                self.picam2.stop()
            except:
                pass
        else:
            self.cap.release()

    def save_image(self, filename):
        f = self.get_frame()
        if f is not None:
            # AI sees flipped, so we save what the AI sees for debug
            proc = self.preprocess(f)[0] # (3, 80, 160)
            vis = (proc.transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imwrite(filename, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
