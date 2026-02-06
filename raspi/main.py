import numpy as np
import onnxruntime as ort
import time
from camera import Camera
from control import RCControl
import os

def main():
    # --- MODELLADRESSERING ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_candidates = [
        os.path.join(os.path.dirname(script_dir), "line_follower.onnx"),
        os.path.join(script_dir, "line_follower.onnx")
    ]
    
    onnx_path = None
    for candidate in onnx_candidates:
        if os.path.exists(candidate):
            onnx_path = candidate
            break
            
    if onnx_path is None:
        print(f"Error: Modell saknas i {onnx_candidates}")
        return

    # --- AI OPTIMERING (För RPi 5) ---
    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4 
        session = ort.InferenceSession(onnx_path, sess_options)
        print(f"Modell laddad: {onnx_path}")
    except Exception as e:
        print(f"Modellfel: {e}")
        return

    # --- HARDWARE SETUP ---
    cam = Camera()
    control = RCControl()
    
    # Armeringsfas: Håller neutralgas i 3 sekunder så ESC aktiveras.
    print("\n--- STARTA BILENS BATTERI NU ---")
    control.arm()

    # --- TRIM-PARAMETRAR (Justera dessa efter testkörning) ---
    # 1. THROTTLE: Höj för fart, sänk om bilen kör av i kurvor.
    THROTTLE_VALUE = 0.15 

    # Enkel automatisk inbromsning i kurvor:
    dynamic_throttle = THROTTLE_VALUE * (1.0 - abs(action) * 0.5) # TEST!!!

    
    # 2. STEER_GAIN: Höj (>1.0) om den svänger för lite. Sänk (<1.0) om den vobblar.
    STEER_GAIN = 1.0
    
    # 3. STRIDE: Hur ofta AI:n "tittar".  
    # Vid högre fart (Throttle > 0.25), sänk STRIDE till 2 eller 1.
    STRIDE = 3
    
    # Buffer för Frame Stacking (AI:ns tidsminne)
    stacked_buffer = np.zeros((1, 12, 80, 160), dtype=np.float32)
    stride_timer = 0
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            frame = cam.get_frame()
            if frame is None: continue
            
            # --- BILDBEHANDLING & STACKING ---
            stride_timer += 1
            if stride_timer >= STRIDE:
                stride_timer = 0
                processed_img = cam.preprocess(frame)
                
                if frame_count == 0:
                    for i in range(4):
                        stacked_buffer[0, i*3:(i+1)*3] = processed_img[0]
                else:
                    # Skiftar bufferten: [Gammal, Gammal, Gammal, NY]
                    stacked_buffer[0, 0:9] = stacked_buffer[0, 3:12]
                    stacked_buffer[0, 9:12] = processed_img[0]
            
            # --- AI INFERENCE ---
            inputs = {session.get_inputs()[0].name: stacked_buffer}
            outputs = session.run(None, inputs)
            raw_ai = float(outputs[0][0][0])
            
            # Applicera Gain och klipp värdet mellan -1 (Höger) och 1 (Vänster)
            action = max(-1.0, min(1.0, raw_ai * STEER_GAIN))
            
            # --- SMOOTHING FILTER ---
            # alpha 0.85 = 85% ny data, 15% gammal. 
            # Sänk alpha (t.ex. 0.7) för ännu mjukare (men segare) styrning.
            alpha = 0.85
            if 'smooth_steering' not in locals():
                smooth_steering = action
            else:
                smooth_steering = (alpha * action) + ((1 - alpha) * smooth_steering)
            
            # Skicka kommandon till servon/motor
            control.set_steering(smooth_steering)
            # control.set_throttle(THROTTLE_VALUE) 
            control.set_throttle(dynamic_throttle) # TEST!!!

            # --- TIMING ---
            # Ser till att loopen håller exakt 60 FPS (0.0166s per varv)
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.0166 - elapsed))
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"AI: {action:5.2f} | Smooth: {smooth_steering:5.2f}", end="\r")

    except KeyboardInterrupt:
        print("\nAvbryter...")
    finally:
        control.stop()
        cam.release()

if __name__ == "__main__":
    main()