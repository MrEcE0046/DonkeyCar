import cv2
import numpy as np
import threading
import time

# Försök ladda Picamera2 (Raspberry Pi 5s kamera-bibliotek)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class Camera:
    def __init__(self, resolution=(640, 480), target_size=(160, 80)):
        """
        resolution: Kamerans råa upplösning.
        target_size: Den storlek AI-modellen förväntar sig (160x80).
        """
        self.target_size = target_size
        self.resolution = resolution
        
        # --- FÄRGKORRIGERING ---
        # Om färger ser konstiga ut i debug-bilderna (t.ex. gul linje blir blå),
        # kan denna sättas till True för att kasta om färgkanalerna.
        self.ENABLE_BGR_FLIP = False 
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock() # Förhindrar att trådar krockar vid bildläsning
        
        if PICAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                # Konfigurera kameran för video-läge (snabbare än stillbild)
                config = self.picam2.create_video_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
                
                # Auto-exponering på: Viktigt för att hantera skuggor/ljus inomhus
                self.picam2.set_controls({"AeEnable": True})
                self.picam2.start()
                self.use_picamera = True
                print("Kamera: Picamera2 startad (RGB888)")
                time.sleep(2.0) # Låt hårdvaran stabiliseras
            except Exception as e:
                print(f"Picamera2 misslyckades: {e}. Fallback till OpenCV.")
                self.use_picamera = False
        else:
            self.use_picamera = False

        # Fallback om Picamera2 inte finns (t.ex. vanlig USB-webbkamera)
        if not self.use_picamera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("Kamera: OpenCV startad")

        # --- BAKGRUNDSTRÅD ---
        # Startar _update i en egen tråd så att kameran aldrig behöver vänta på AI:n.
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        """ Loop som konstant hämtar den senaste bilden från hårdvaran. """
        while not self.stopped:
            try:
                if self.use_picamera:
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
                print(f"Fel i kameratråden: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """ Returnerar den senaste bilden säkert. """
        with self.lock:
            return self.frame

    def preprocess(self, frame):
        """ 
        Gör om den stora kamerabilden till vad AI:n vill ha.
        Viktigt: AI:n behöver (3, 80, 160) i flyttal mellan 0 och 1.
        """
        if frame is None: return None
        
        # 1. CROP (Beskärning)
        # Vi tar bort den översta tredjedelen (himlen/väggarna) 
        # och fokuserar på marken (160 till 480 pixlar ner).
        h, w = frame.shape[:2]
        crop_top = 160 
        crop_bottom = 480
        crop = frame[crop_top:crop_bottom, :]
        
        if self.ENABLE_BGR_FLIP:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        
        # 2. RESIZE
        # Skalar ner bilden till AI:ns upplösning (160x80).
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 3. FORMATTERING (NCHW)
        # Flyttar färgkanalen först: (Höjd, Bredd, Kanaler) -> (Kanaler, Höjd, Bredd)
        processed = resized.transpose(2, 0, 1).astype(np.float32)
        
        # Normalisering: Gör om pixlar (0-255) till värden mellan (0.0 - 1.0)
        processed *= (1.0 / 255.0)
        
        # Lägger till en extra dimension för "batch size" (1, 3, 80, 160)
        return np.expand_dims(processed, axis=0)

    def release(self):
        """ Stänger ner kameran snyggt. """
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