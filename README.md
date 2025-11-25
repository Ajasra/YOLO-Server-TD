# YOLO11 Real-Time Tracking Bridge

This project acts as a middleware between a camera source (Webcam, NDI, or File) and TouchDesigner. It uses YOLO11 for person detection and tracking, sending the data via OSC.

## Features
- **High Performance:** Uses YOLO11 Nano and ONNX Runtime (GPU accelerated if available).
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
    - **Note:** You must install the [NDI SDK](https://ndi.video/tech/) for NDI support to work.

## Installation & Setup

This project is managed by `uv` and requires **Python 3.10+**.

1.  **Install `uv` (if not installed):**
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Sync Dependencies:**
    In the project folder, run:
    ```powershell
    uv sync
    ```
    This creates the virtual environment and installs all required packages, including **PyTorch with CUDA 12.1 support** and **ONNX Runtime GPU**.

    *Note:* If `ndi-python` fails to install:
    1. Ensure you have the NDI SDK installed (check `$env:NDI_SDK_DIR`).
    2. Ensure `cmake` is installed (`winget install Kitware.CMake` or `pip install cmake`).

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

    # Use Webcam (ID 0)
    uv run main.py --source WEBCAM --cam-id 0

    # Use specific NDI Source Name
    uv run main.py --source NDI --ndi-name "MySource"

    # Detect specific classes (0=Person, 67=Cell Phone)
    uv run main.py --classes 0 67
    ```

3.  **Full Argument List:**
    | Argument | Default | Description |
    | :--- | :--- | :--- |
    | `--source` | `NDI` | Input source: `NDI`, `WEBCAM`, or `FILE` |
    | `--debug` | `False` | Enable preview window |
    | `--max-fps` | `None` | Limit processing FPS (e.g. 30). `0` or `None` for unlimited. |
    | `--cam-id` | `0` | Webcam Device ID |
    | `--ndi-name` | `TD_OUTPUT` | NDI Sender Name to look for |
    | `--osc-ip` | `127.0.0.1` | OSC Target IP |
    | `--osc-port` | `9000` | OSC Target Port |
    | `--conf` | `0.5` | Detection Confidence Threshold |
    | `--classes` | `0` | Class IDs to detect (space separated) |
    | `--no-onnx` | `False` | Force PyTorch instead of ONNX |

4.  **TouchDesigner Setup:**
    - Create an **OSC In DAT** listening on port `9000`.
    - Use the callback script provided in the docs (or PDD) to parse data.

## Troubleshooting

- **Reshape Error (819...):** If you see a massive number in a reshape error, ensure `simplify=False` is set in the export options in `main.py`.
- **Falling back to CPU:** Check that your NVIDIA drivers are up to date. The project is configured for CUDA 12.1.
- **NDI Issues:** If `NDIlib` is missing, NDI source will be disabled. ensure NDI SDK is installed.
- **Performance:** If running slow, ensure `onnxruntime-gpu` is effectively using your GPU (check CUDA drivers).
