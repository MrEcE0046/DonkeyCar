import cv2
import numpy as np
import onnxruntime as ort
import time
from camera import Camera
from control import RCControl

import os

def main():
    # 1. Initialize models and hardware
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try to find the model in current or parent directory
    # Prioritize parent directory as that's where we usually scp fresh models
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
        print(f"Error: line_follower.onnx not found in {onnx_candidates}")
        return

    try:
        # Optimize for Pi 5 ARM cores
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options)
        print(f"Successfully loaded model: {onnx_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Create a unique session folder for debugging
    session_id = time.strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(script_dir, "debug_sessions", f"session_{session_id}")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Debug images will be saved to: {debug_dir}")

    cam = Camera()
    control = RCControl()
    
    # 2. ARMING PHASE (Crucial for ESC)
    print("\n--- ARMING ESC ---")
    print("Action: TURN ON THE CAR BATTERY NOW!")
    control.arm()
    print("\nESC should be ARMED. Starting autonomous loop...")

    # V11: Lowered throttle to give steering more time in curves
    THROTTLE_VALUE = 0.15 
    # V14: Lowered gain to 1.0 to handle early-stage policy sensitivity
    STEER_GAIN = 1.0
    
    # Pre-allocated stacked buffer for performance (12 channels for 4x RGB)
    # V6: Model expects 160x80 (Native resolution)
    stacked_buffer = np.zeros((1, 12, 80, 160), dtype=np.float32)
    
    # Temporal Stride: Skip frames for the AI memory to get a longer time horizon
    # 60 FPS + Stride 3 = 20 Hz temporal updates (Matches simulator training)
    stride_timer = 0
    STRIDE = 3
    
    frame_count = 0
    print(f"\nRC Line Follower V11. Autonomous drive (Throttle: {THROTTLE_VALUE}).")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            # 2. Get image from threaded camera (Fast call)
            # Camera thread is already pushing at 60 FPS
            frame = cam.get_frame()
            # V14: DISABLE BGR_FLIP. Simulator trains on RGB. 
            cam.ENABLE_BGR_FLIP = False 
            if frame is None: continue
            
            # 3. Handle Stacking with Temporal Stride
            stride_timer += 1
            if stride_timer >= STRIDE:
                stride_timer = 0
                processed_img = cam.preprocess(frame)
                
                if frame_count == 0:
                    for i in range(4):
                        stacked_buffer[0, i*3:(i+1)*3] = processed_img[0]
                else:
                    # SB3 VecFrameStack expects: [Oldest, ..., Newest]
                    # Shift left (remove oldest) and push newest to the end
                    stacked_buffer[0, 0:9] = stacked_buffer[0, 3:12]
                    stacked_buffer[0, 9:12] = processed_img[0]
            
            # 4. Inference (Full frequency)
            inputs = {session.get_inputs()[0].name: stacked_buffer}
            outputs = session.run(None, inputs)
            raw_ai = float(outputs[0][0][0])
            
            # V11: Steering Boost & Clipping
            action = max(-1.0, min(1.0, raw_ai * STEER_GAIN))
            
            # 5. Control with Smoothing Filter (V4 Pro)
            # V9: alpha=0.85 (Less smoothing, more responsiveness to curves)
            alpha = 0.85
            if 'smooth_steering' not in locals():
                smooth_steering = action
            else:
                smooth_steering = (alpha * action) + ((1 - alpha) * smooth_steering)
            
            control.set_steering(smooth_steering)
            control.set_throttle(THROTTLE_VALUE) 
            
            frame_count += 1
            # Debug saves help us see what's happening
            if frame_count % 30 == 0:
                p_v = cam.preprocess(frame)[0]
                proc_v = (p_v.transpose(1, 2, 0) * 255).astype(np.uint8)
                fname = os.path.join(debug_dir, f"step{frame_count:05d}_ai{action:+.3f}.jpg")
                cv2.imwrite(fname, cv2.cvtColor(proc_v, cv2.COLOR_RGB2BGR))
            
            # 6. Maintain Maximum Stability (60 FPS)
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.0166 - elapsed))
            
            if frame_count % 10 == 0:
                dt = time.time() - start_time
                fps = 1.0 / dt if dt > 0 else 0
                # Telemetry: AI is raw*gain, Smooth is filtered
                print(f"FPS: {fps:4.1f} | AI: {action:5.2f} | Smooth: {smooth_steering:5.2f}", end="\r")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        control.stop()
        cam.release()

if __name__ == "__main__":
    main()
