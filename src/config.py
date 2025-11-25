import argparse

# ==========================================
# ⚙️ CONFIGURATION (DEFAULTS)
# ==========================================

# INPUT SOURCE: "WEBCAM", "NDI", or "FILE"
DEFAULT_SOURCE_TYPE = "NDI" 

# WEBCAM SETTINGS
DEFAULT_CAM_ID = 0                  # 0 = Default, 1 = External
DEFAULT_TARGET_FPS = 60             # Request Hardware FPS
DEFAULT_MAX_FPS = 30                # Limit processing FPS (None = Unlimited)

# FILE SETTINGS (If using FILE source)
DEFAULT_VIDEO_FILE = "data/vid1.mp4"

# NDI SETTINGS (If using NDI source)
DEFAULT_NDI_SOURCE_NAME = "TD_OUTPUT"  # Name of sender in TouchDesigner

# OSC OUTPUT (To TouchDesigner)
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9000
DEFAULT_OSC_ADDRESS = "/face"

# AI MODEL SETTINGS
DEFAULT_MODEL_BASE = 'yolo11n'      # The model version (Nano is fastest)
DEFAULT_USE_ONNX = True             # True = Convert to ONNX for speed boost
DEFAULT_CONFIDENCE = 0.5            # Sensitivity (0.1 = Low, 0.9 = High)
DEFAULT_CLASSES = [0]               # 0 = Person. Set to None for all objects.
DEFAULT_CPU_CORES = None            # List of CPU cores to use (e.g. [0, 1, 2, 3]). None = All cores.

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8/11 Object Detection with OSC Output")
    
    # General
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with preview window")
    parser.add_argument("--max-fps", type=int, default=DEFAULT_MAX_FPS, help="Limit processing FPS (reduce load)")
    
    # Source
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_TYPE, choices=["WEBCAM", "NDI", "FILE"], help="Input source type")
    
    # Webcam
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="Webcam ID (if source is WEBCAM)")
    parser.add_argument("--fps", type=int, default=DEFAULT_TARGET_FPS, help="Target FPS for webcam")
    
    # File
    parser.add_argument("--video-file", type=str, default=DEFAULT_VIDEO_FILE, help="Path to video file (if source is FILE)")
    
    # NDI
    parser.add_argument("--ndi-name", type=str, default=DEFAULT_NDI_SOURCE_NAME, help="NDI Source Name")
    
    # OSC
    parser.add_argument("--osc-ip", type=str, default=DEFAULT_OSC_IP, help="OSC Destination IP")
    parser.add_argument("--osc-port", type=int, default=DEFAULT_OSC_PORT, help="OSC Destination Port")
    parser.add_argument("--osc-address", type=str, default=DEFAULT_OSC_ADDRESS, help="OSC Address")
    
    # Model
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_BASE, help="YOLO model base name (e.g. yolo11n)")
    parser.add_argument("--no-onnx", action="store_true", help="Disable ONNX optimization (force PyTorch)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--classes", type=int, nargs='+', default=DEFAULT_CLASSES, help="Class IDs to detect (space separated)")
    parser.add_argument("--cpu-cores", type=int, nargs='+', default=DEFAULT_CPU_CORES, help="Specific CPU cores to use (space separated)")

    return parser.parse_args()

