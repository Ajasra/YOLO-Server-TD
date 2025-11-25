

# Project: YOLO11 Real-Time Tracking Bridge
**Version:** 2.0 (High-Performance / ONNX Enabled)
**Target Platform:** Python 3.10+ & TouchDesigner
**Package Manager:** uv

## 1. Executive Summary
This project is a high-performance middleware application that bridges computer vision with interactive rendering. It captures video feeds (via physical **Webcam** or networked **NDI**), processes them using the **YOLO11 Nano** neural network, and streams tracking data to **TouchDesigner** via **OSC**.

To achieve **60+ FPS** and low latency:
*   We utilize **YOLO11n** (Nano) for state-of-the-art speed/accuracy balance.
*   We auto-convert models to **ONNX** format to bypass Python overhead and utilize optimized C++ inference runtimes.
*   We leverage **GPU Acceleration** (CUDA) where available.
*   Tracking is **stateless**: Users who leave and re-enter are assigned new IDs.

---

## 2. System Architecture

1.  **Input Layer:**
    *   **Mode A (Low Latency):** Direct USB/PCIe Camera access via OpenCV.
    *   **Mode B (Networked):** NDI Video Stream (received from TouchDesigner or other sources).
    *   **Mode C (File):** Load file for testing
2.  **Processing Layer (Python):**
    *   **Inference:** YOLO11 (ONNX Runtime) detects "Person" class.
    *   **Tracking:** BoT-SORT algorithm assigns IDs and maintains continuity across frames.
    *   **Normalization:** Coordinates are converted to 0.0–1.0 range (independent of resolution).
    *   **Debug:** Debug parameter to show teh preview of the output.
3.  **Output Layer:**
    *   **Protocol:** UDP OSC (Open Sound Control).
    *   **Payload:** `/face [ID, CenterX, CenterY, Width, Height]`.
4.  **Client Layer (TouchDesigner):**
    *   Parses OSC.
    *   Manages object lifecycle (Creation, Updates, Deletion).
    *   Drives Instanced Geometry.

---

## 3. Prerequisites

### Hardware
*   **GPU:** NVIDIA GeForce GTX 1060 or higher recommended for 60FPS (requires CUDA drivers).
    *   *Note: ONNX Runtime also accelerates CPU inference if no GPU is present.*
*   **Camera:** Standard USB Webcam or NDI Source.

### Software
*   **OS:** Windows 10/11 (Recommended for NDI/TouchDesigner compatibility) or macOS.
*   **Python:** Version 3.10 or higher.
*   **TouchDesigner:** Any recent build.
*   **NDI Tools:** (Optional but recommended) Install the NDI Runtime if NDI detection fails. [Download here](https://ndi.video/tools/).

---

## 4. Installation Guide (using `uv`)

We use `uv` for lightning-fast environment setup and dependency resolution.

### Step 1: Install uv
Open PowerShell (Windows) or Terminal (Mac/Linux):
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Set up Project Environment
Navigate to your desired project folder:
```bash
mkdir yolo_td_bridge
cd yolo_td_bridge

# Create virtual environment
uv venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### Step 3: Install High-Performance Libraries
We install standard libraries plus `onnxruntime-gpu` for maximum speed.

```bash
uv pip install ultralytics opencv-python python-osc ndi-python onnx onnxruntime-gpu
```

---

## 5. The Application Code (`tracker.py`)

Create a file named `tracker.py` in your folder and paste the code below.
This script features an **Auto-Optimizer** that converts YOLO models to ONNX on the first run.

```python
import cv2
import numpy as np
import NDIlib as ndi
from ultralytics import YOLO
from pythonosc import udp_client
import time
import os
import torch
import sys

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

# INPUT SOURCE: "WEBCAM" or "NDI"
SOURCE_TYPE = "WEBCAM" 

# WEBCAM SETTINGS
CAM_ID = 0                  # 0 = Default, 1 = External
TARGET_FPS = 60             # Request Hardware FPS

# NDI SETTINGS (If using NDI source)
NDI_SOURCE_NAME = "TD_OUTPUT"  # Name of sender in TouchDesigner

# OSC OUTPUT (To TouchDesigner)
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
OSC_ADDRESS = "/face"

# AI MODEL SETTINGS
MODEL_BASE = 'yolo11n'      # The model version (Nano is fastest)
USE_ONNX = True             # True = Convert to ONNX for speed boost
CONFIDENCE = 0.5            # Sensitivity (0.1 = Low, 0.9 = High)
CLASSES = [0]               # 0 = Person. Set to None for all objects.

# ==========================================
# 🧠 MODEL OPTIMIZER
# ==========================================
def load_model():
    """
    Loads YOLO11. Automatically exports to ONNX if enabled and missing.
    Returns: model object, device string
    """
    pt_file = f"{MODEL_BASE}.pt"
    onnx_file = f"{MODEL_BASE}.onnx"
    
    # Check Hardware
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ No GPU detected. Running on CPU.")

    if USE_ONNX:
        # If ONNX file doesn't exist, create it
        if not os.path.exists(onnx_file):
            print(f"⚙️ First Run Optimization: Converting {pt_file} to ONNX...")
            print("   (This takes ~20 seconds, please wait...)")
            model = YOLO(pt_file)
            # Dynamic=True allows different input resolutions (e.g., resizing windows)
            model.export(format='onnx', dynamic=True, simplify=True)
            print("✅ Export Complete.")
        
        print(f"🚀 Loading Optimized Model: {onnx_file}")
        # 'task=detect' is required for ONNX loading in Ultralytics
        return YOLO(onnx_file, task='detect'), device
    else:
        print(f"🐌 Loading Standard Model (PyTorch): {pt_file}")
        return YOLO(pt_file), device

# ==========================================
# 📹 VIDEO HANDLER
# ==========================================
class VideoHandler:
    def __init__(self, source_type):
        self.type = source_type
        self.cap = None
        self.ndi_recv = None
        
        print(f"--- Initializing Source: {self.type} ---")

        if self.type == "WEBCAM":
            self.cap = cv2.VideoCapture(CAM_ID)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            # Disable buffer to ensure lowest latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Error: Cannot open Webcam ID {CAM_ID}")

        elif self.type == "NDI":
            if not ndi.initialize(): raise RuntimeError("❌ NDI Init Failed")
            self.ndi_recv = ndi.recv_create_v3()
            if self.ndi_recv is None: raise RuntimeError("❌ NDI Recv Create Failed")
            
            print("🔎 Scanning for NDI Sources (3s)...")
            sources = []
            for _ in range(3):
                sources = ndi.find_get_current_sources(ndi.find_create_v2())
                if sources: break
                time.sleep(1)
            
            # Find specific source or default to first
            target = next((s for s in sources if NDI_SOURCE_NAME in s.ndi_name), None)
            if not target and sources:
                print(f"⚠️ '{NDI_SOURCE_NAME}' not found. Using '{sources[0].ndi_name}'")
                target = sources[0]
            
            if target:
                ndi.recv_connect(self.ndi_recv, target)
                print(f"✅ Connected to NDI: {target.ndi_name}")
            else:
                raise RuntimeError("❌ No NDI Sources Found.")

    def get_frame(self):
        if self.type == "WEBCAM":
            return self.cap.read()
        elif self.type == "NDI":
            t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 0) # 0ms poll
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = np.copy(np.frombuffer(v.data, dtype=np.uint8))
                # Reshape and Drop Alpha Channel (BGRA -> BGR)
                frame = frame.reshape((v.yres, v.xres, 4))[:, :, :3]
                ndi.recv_free_video_v2(self.ndi_recv, v)
                return True, frame
            return False, None

    def release(self):
        if self.type == "WEBCAM": self.cap.release()
        elif self.type == "NDI": ndi.recv_destroy(self.ndi_recv); ndi.destroy()

# ==========================================
# 🏁 MAIN LOOP
# ==========================================
def main():
    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    
    try:
        model, device = load_model()
        video = VideoHandler(SOURCE_TYPE)
    except Exception as e:
        print(e)
        return

    print(f"📡 Streaming OSC to {OSC_IP}:{OSC_PORT}")
    print("Press CTRL+C to stop.")

    try:
        while True:
            success, frame = video.get_frame()
            
            if not success or frame is None:
                # Sleep briefly to prevent CPU spike when waiting for NDI frames
                time.sleep(0.001) 
                continue

            # --- INFERENCE ---
            # persist=True: Vital for tracking IDs
            # verbose=False: Keeps console clean
            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=CONFIDENCE, 
                classes=CLASSES,
                device=device
            )

            # --- PARSING & SENDING ---
            if results[0].boxes.id is not None:
                # Extract Data (on CPU)
                boxes = results[0].boxes.xywhn.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # Y-Flip: OpenCV Top-Left (0,0) vs TD Bottom-Left (0,0)
                    # We flip Y here so it makes sense in TD immediately
                    client.send_message(OSC_ADDRESS, [int(track_id), float(x), float(1.0-y), float(w), float(h)])

            # Optional: Visualize (Disable for max FPS)
            # cv2.imshow("Preview", results[0].plot())
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 6. TouchDesigner Integration

To use this data for interaction, we need to capture the stream and maintain a list of active users.

### Step 1: Receiving Data
1.  Create an **OSC In DAT**.
2.  Set **Network Port** to `9000`.
3.  Set **Pulse** on `Callbacks` to ensure it's active.

### Step 2: The Logic (Callbacks)
Since the Python script sends data *per person*, we need to compile this into a table.
1.  Create a **Table DAT** named `faces`.
    *   Add headers in the first row: `id`, `tx`, `ty`, `sx`, `sy`, `life`.
2.  Click the **Python** icon (blue/yellow) on the **OSC In DAT**.
3.  Click **Edit** to open the callback editor.
4.  Paste this code:

```python
# oscin1_callbacks
def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    if address == "/face":
        # args = [id, x, y, w, h]
        f_id = int(args[0])
        
        # Get our storage table
        tab = op('faces')
        
        # Check if this ID is already in the table
        if tab.row(str(f_id)):
            # Update existing user
            tab[str(f_id), 'tx'] = args[1]
            tab[str(f_id), 'ty'] = args[2]
            tab[str(f_id), 'sx'] = args[3]
            tab[str(f_id), 'sy'] = args[4]
            tab[str(f_id), 'life'] = 10  # Reset "Time To Live"
        else:
            # Create new user
            tab.appendRow([f_id, args[1], args[2], args[3], args[4], 10])
    return
```

### Step 3: Cleanup (Garbage Collection)
We need to remove users who stop being detected (leave the frame).
1.  Create an **Execute DAT**.
2.  Toggle **Start** (Frame Start) to On.
3.  Edit the code:

```python
def onFrameStart(frame):
    tab = op('faces')
    # Loop backwards to safely delete rows
    for i in range(tab.numRows-1, 0, -1):
        life = int(tab[i, 'life'])
        if life <= 0:
            tab.deleteRow(i)
        else:
            tab[i, 'life'] = life - 1
    return
```

### Step 4: Visualization (Instancing)
1.  Connect the `faces` Table DAT to a **DAT to CHOP**.
    *   *Select Rows:* 2-End (Skip header).
    *   *Output:* Select `tx`, `ty`, `sx`, `sy`.
2.  Create a **Circle SOP** and a **Geometry COMP**.
3.  Turn **Instancing** to **On** in the Geometry COMP.
4.  Drag the **DAT to CHOP** onto the **Default Instance OP**.
5.  Map X -> `tx`, Y -> `ty`, Scale X -> `sx`, Scale Y -> `sy`.

---

## 7. Troubleshooting

*   **"No NDI Sources Found":**
    *   Ensure TouchDesigner has an **NDI Out TOP** active.
    *   Ensure the Python script's `NDI_SOURCE_NAME` matches the NDI name in TD.
    *   Firewall might be blocking the connection.
*   **"DLL load failed":**
    *   If using NDI, download the [NDI Tools](https://ndi.video/tools/).
    *   If using GPU, ensure you have [NVIDIA CUDA Toolkit 11.8 or 12.x](https://developer.nvidia.com/cuda-downloads) installed.
*   **Slow Performance:**
    *   Check the terminal output. If it says `No GPU detected`, check your NVIDIA drivers.
    *   Ensure `cv2.imshow` is commented out in the loop (it is by default in the script provided).
*   **Detection is erratic:**
    *   Lower the `CONFIDENCE` variable in `tracker.py` to `0.3` for better detection in low light, or raise to `0.7` to reduce false positives.