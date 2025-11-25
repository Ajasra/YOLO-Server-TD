import os
import torch
from ultralytics import YOLO

def load_model(model_base, use_onnx=True, cpu_cores=None):
    """
    Loads YOLO11. Automatically exports to ONNX if enabled and missing.
    Returns: model object, device string
    """
    pt_file = f"{model_base}.pt"
    onnx_file = f"{model_base}.onnx"
    
    # Check Hardware
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ No GPU detected. Running on CPU.")

        # CPU Affinity Logic
        if cpu_cores:
            try:
                import psutil
                proc = psutil.Process()
                proc.cpu_affinity(cpu_cores)
                print(f"✅ CPU Affinity set to cores: {cpu_cores}")
            except ImportError:
                 print("⚠️ 'psutil' library not found. Cannot set CPU affinity.")
                 print("   Run 'pip install psutil' to enable this feature.")
            except Exception as e:
                 print(f"❌ Failed to set CPU affinity: {e}")

    if use_onnx:
        # Check ONNX Runtime Providers
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            print(f"🔎 ONNX Runtime Providers: {available_providers}")
            
            # Check for package conflict
            from importlib.metadata import distributions
            installed = [d.metadata['Name'] for d in distributions()]
            if 'onnxruntime' in installed and 'onnxruntime-gpu' in installed:
                print("⚠️ CONFLICT: Both 'onnxruntime' and 'onnxruntime-gpu' are installed.")
                print("   This forces CPU mode. Please run:")
                print("   pip uninstall -y onnxruntime onnxruntime-gpu")
                print("   pip install onnxruntime-gpu")

            if 'CUDAExecutionProvider' not in available_providers and torch.cuda.is_available():
                print("⚠️ WARNING: CUDA is available but ONNX Runtime is running on CPU!")
                print("   Possible causes:")
                print("   1. 'onnxruntime' is installed instead of (or alongside) 'onnxruntime-gpu'")
                print("   2. CUDA/cuDNN libraries are missing or incompatible with this ONNX Runtime version.")
        except ImportError:
            pass

        # If ONNX file doesn't exist, create it
        if not os.path.exists(onnx_file):
            print(f"⚙️ First Run Optimization: Converting {pt_file} to ONNX...")
            print("   (This takes ~20 seconds, please wait...)")
            try:
                model = YOLO(pt_file)
                # Dynamic=True allows different input resolutions (e.g., resizing windows)
                # simplify=False: sometimes breaks dynamic shapes on newer YOLO/ONNX versions
                model.export(format='onnx', dynamic=True, simplify=False, opset=17)
                print("✅ Export Complete.")
            except Exception as e:
                print(f"❌ Export Failed: {e}")
                print("   Falling back to PyTorch model.")
                return YOLO(pt_file), device
        
        print(f"🚀 Loading Optimized Model: {onnx_file}")
        # 'task=detect' is required for ONNX loading in Ultralytics
        try:
            return YOLO(onnx_file, task='detect'), device
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            print("   Falling back to PyTorch model.")
            return YOLO(pt_file), device
    else:
        print(f"🐌 Loading Standard Model (PyTorch): {pt_file}")
        return YOLO(pt_file), device

