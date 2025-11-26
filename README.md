# YOLO11 Real-Time Tracking Bridge

This project acts as a middleware between a camera source (Webcam, NDI, or File) and TouchDesigner. It uses YOLO11 for person detection and tracking, sending the data via OSC.

## Features
- **High Performance:** Uses YOLO11 Nano and ONNX Runtime (GPU accelerated if available), now with **FP16** support.
- **Smart Resource Management:** Skips expensive video decoding when exceeding target FPS to minimize CPU usage.
- **Batched OSC:** Sends all detections in a single bundled message to reduce network overhead.
- **Resolution Control:** Dedicated flags to reduce camera resolution at the hardware level for lower CPU load.
- **Flexible Input:** Supports Webcam, NDI (Network), and Video Files.
- **Tracking:** Implements BoT-SORT for persistent ID tracking.
- **Output:** Sends normalized coordinates via OSC to TouchDesigner.

## Prerequisites
- Windows 10/11
- [UV Package Manager](https://github.com/astral-sh/uv) (Required)
- NVIDIA GPU (Highly Recommended for real-time performance)
    - **CUDA 12.1** capable GPU
    - Drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
- NDI Tools (Optional, for NDI support)
    - **Note:** You must install the [NDI SDK](https://ndi.video/tech/) and the Python wrapper to use NDI.

## Installation & Setup

This project is managed by `uv` and requires **Python 3.10+**.

1.  **Install `uv` (if not installed):**
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Sync Dependencies:**
    
    Select the installation mode that matches your hardware needs:

    **Option A: GPU Accelerated (Recommended for NVIDIA GPUs)**
    *Requires CUDA 12.1 capable GPU.*
    ```powershell
    uv sync --extra gpu
    ```

    **Option B: CPU Only (Slower, but universal)**
    *Use this if you don't have a dedicated NVIDIA GPU.*
    ```powershell
    uv sync --extra cpu
    ```

    **Option C: With NDI Support**
    *Add `--extra ndi` to any command above.*
    ```powershell
    # Example: GPU + NDI
    uv sync --extra gpu --extra ndi
    ```

    *Note:* For NDI support, ensure you have the [NDI SDK](https://ndi.video/tech/) installed. If `ndi-python` fails to install, ensure `cmake` is installed (`winget install Kitware.CMake` or `pip install cmake`).

### Advanced: Lightweight CPU-Only Installation
The default configuration includes CUDA-enabled libraries (~2.5GB) to ensure compatibility. If you need a **lightweight** CPU-only installation (~200MB):

1. Open `pyproject.toml`.
2. Find the `[tool.uv.index]` section at the bottom.
3. Change the URL to: `https://download.pytorch.org/whl/cpu`.
4. Run: `uv sync --extra cpu`

## Usage

1.  **Run with Defaults:**
    Simply run the batch file or use `uv run`:
    ```powershell
    ./run.bat
    # OR
    uv run main.py
    ```

2.  **Command Line Arguments:**
    The server is fully configurable via CLI arguments.
    
    **Common Examples:**
    ```powershell
    # Enable Debug Preview
    uv run main.py --debug

    # Limit to 30 FPS (Reduces load)
    uv run main.py --max-fps 30

    # Use Webcam (ID 0) with lower resolution
    uv run main.py --source WEBCAM --cam-id 0 --width 640 --height 480

    # Use specific NDI Source Name (Requires NDI installation)
    uv run main.py --source NDI --ndi-name "MySource"

    # Detect specific classes (0=Person, 67=Cell Phone)
    uv run main.py --classes 0 67
    ```

3.  **Full Argument List:**
    | Argument | Default | Description |
    | :--- | :--- | :--- |
    | `--source` | `WEBCAM` | Input source: `NDI`, `WEBCAM`, or `FILE` |
    | `--debug` | `False` | Enable preview window |
    | `--max-fps` | `None` | Limit processing FPS (e.g. 30). `0` or `None` for unlimited. |
    | `--cam-id` | `0` | Webcam Device ID |
    | `--width` | `640` | Request Webcam Width (Lower = Faster) |
    | `--height` | `480` | Request Webcam Height |
    | `--ndi-name` | `TD_OUTPUT` | NDI Sender Name to look for |
    | `--osc-ip` | `127.0.0.1` | OSC Target IP |
    | `--osc-port` | `9000` | OSC Target Port |
    | `--conf` | `0.5` | Detection Confidence Threshold |
    | `--classes` | `0` | Class IDs to detect (space separated) |
    | `--precision` | `fp16` | `fp16` (Half) or `fp32` (Full). FP16 is faster on GPU. |
    | `--no-onnx` | `False` | Force PyTorch instead of ONNX |

4.  **TouchDesigner Setup:**
    - **Example File:** Check `TD/yolo.tox` included in the project for a complete example.
    - Create an **OSC In DAT** listening on port `9000`.
    - **Data Format (Batched):** The server sends a single OSC message to `/yolo` containing a flat list of all detections for that frame.
    - **Structure:** `[id, x, y, w, h, id, x, y, w, h, ...]`
    - **Parsing Example (Python in TD):**
      ```python
      # Script in OSC In DAT Callbacks
      # Requires a Table DAT named 'data'
      
      tab = op('data')
      
      def onReceiveOSC(dat, rowIndex, message, byteData, timeStamp, address, args, peer):
          if address == "/yolo":
              # Initialize table if empty
              if tab.numRows == 0:
                  tab.appendRow(['id', 'tx', 'ty', 'sx', 'sy', 'life'])

              # args = [id, x, y, w, h, id, x, y, w, h, ...]
              for i in range(0, len(args), 5):
                  f_id = int(args[i])
                  x = args[i+1]
                  y = args[i+2]
                  w = args[i+3]
                  h = args[i+4]
                  
                  # Check if this ID is already in the table
                  if tab.row(str(f_id)):
                      # Update existing user
                      tab[str(f_id), 'tx'] = x
                      tab[str(f_id), 'ty'] = y
                      tab[str(f_id), 'sx'] = w
                      tab[str(f_id), 'sy'] = h
                      tab[str(f_id), 'life'] = 10
                  else:
                      # Create new user
                      tab.appendRow([f_id, x, y, w, h, 10])
      ```

## Troubleshooting

- **Reshape Error (819...):** If you see a massive number in a reshape error, ensure `simplify=False` is set in the export options in `main.py`.
- **Falling back to CPU:** Check that your NVIDIA drivers are up to date. The project is configured for CUDA 12.1.
- **ONNX Runtime not found:** Run `uv sync --extra gpu` (or `cpu`) to install the inference engine.
- **NDI Issues:** If you see `ImportError` or NDI failures, make sure you ran `uv sync --extra ndi` and have the NDI SDK installed.
- **Performance:** If running slow, ensure `onnxruntime-gpu` is effectively using your GPU (check CUDA drivers).
