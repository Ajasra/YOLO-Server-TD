import time
import cv2
from pythonosc import udp_client

from src.config import parse_arguments
from src.model_loader import load_model
from src.video_handler import VideoHandler

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
def main():
    args = parse_arguments()

    # --- CONFIGURATION ---
    SOURCE_TYPE = args.source
    CAM_ID = args.cam_id
    TARGET_FPS = args.fps
    WIDTH = args.width
    HEIGHT = args.height
    MAX_FPS = args.max_fps
    if MAX_FPS is not None and MAX_FPS <= 0:
        MAX_FPS = None
    VIDEO_FILE = args.video_file
    NDI_SOURCE_NAME = args.ndi_name
    OSC_IP = args.osc_ip
    OSC_PORT = args.osc_port
    OSC_ADDRESS = args.osc_address
    MODEL_BASE = args.model
    USE_ONNX = not args.no_onnx
    CONFIDENCE = args.conf
    CLASSES = args.classes
    CPU_CORES = args.cpu_cores
    DEBUG_MODE = args.debug

    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    
    try:
        model, device = load_model(MODEL_BASE, USE_ONNX, CPU_CORES)
        video = VideoHandler(SOURCE_TYPE, CAM_ID, TARGET_FPS, WIDTH, HEIGHT, VIDEO_FILE, NDI_SOURCE_NAME)
    except Exception as e:
        print(f"Startup Error: {e}")
        return

    print(f"📡 Streaming OSC to {OSC_IP}:{OSC_PORT}")
    if DEBUG_MODE:
        print("👀 Debug Mode: ON (Press 'q' to exit preview)")
    print("Press CTRL+C to stop.")

    prev_time = 0
    fps = 0
    last_process_time = 0

    try:
        while True:
            # --- RATE LIMITING CHECK ---
            curr_time = time.time()
            should_process = True
            if MAX_FPS is not None:
                if (curr_time - last_process_time) < (1.0 / MAX_FPS):
                    should_process = False

            # Get frame (skip decoding if we are just rate limiting to save CPU)
            success, frame = video.get_frame(decode=should_process)
            
            if not success:
                # Sleep briefly to prevent CPU spike when waiting for NDI frames
                # time.sleep(0.001) # NDI recv now handles wait
                if DEBUG_MODE and video.type == "NDI":
                     # Print status occasionally if stuck
                     if time.time() % 2.0 < 0.1:
                         print("⚠️ Waiting for NDI Video frames...")
                continue

            if not should_process:
                # We skipped this frame to maintain target FPS
                continue

            last_process_time = curr_time

            if frame is None:
                continue

            # --- INFERENCE ---
            # persist=True: Vital for tracking IDs
            # verbose=False: Keeps console clean
            # half=True: Use FP16 if available (mostly GPU)
            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=CONFIDENCE, 
                classes=CLASSES,
                device=device,
                half=(device != 'cpu')
            )

            # --- PARSING & SENDING (BATCHED) ---
            if results[0].boxes.id is not None:
                # Extract Data (on CPU)
                boxes = results[0].boxes.xywhn.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                osc_batch = []
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # Y-Flip: OpenCV Top-Left (0,0) vs TD Bottom-Left (0,0)
                    # We flip Y here so it makes sense in TD immediately
                    # Format: [id, x, y, w, h, id, x, y, w, h, ...]
                    osc_batch.extend([int(track_id), float(x), float(1.0-y), float(w), float(h)])

                if osc_batch:
                    client.send_message(OSC_ADDRESS, osc_batch)

            # Optional: Visualize (Disable for max FPS)
            if DEBUG_MODE:
                annotated_frame = results[0].plot()
                
                # Calculate FPS (Rolling Average)
                curr_time = time.time()
                if prev_time > 0:
                    current_fps = 1 / (curr_time - prev_time)
                    fps = 0.9 * fps + 0.1 * current_fps # Smoothing
                else:
                    fps = 0
                prev_time = curr_time
                
                # Draw FPS on frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        if 'video' in locals():
            video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
