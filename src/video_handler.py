import cv2
import numpy as np
import time
import os

try:
    import NDIlib as ndi
except ImportError:
    ndi = None
    print("⚠️ NDIlib not found. NDI source will not work. Install ndi-python.")

class VideoHandler:
    def __init__(self, source_type, cam_id=0, target_fps=60, video_file="data/vid1.mp4", ndi_source_name="TD_OUTPUT"):
        self.type = source_type
        self.cam_id = cam_id
        self.target_fps = target_fps
        self.video_file = video_file
        self.ndi_source_name = ndi_source_name
        
        self.cap = None
        self.ndi_recv = None
        
        print(f"--- Initializing Source: {self.type} ---")

        if self.type == "WEBCAM":
            self.cap = cv2.VideoCapture(self.cam_id)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Disable buffer to ensure lowest latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Error: Cannot open Webcam ID {self.cam_id}")

        elif self.type == "FILE":
            if not os.path.exists(self.video_file):
                raise RuntimeError(f"❌ Error: File not found: {self.video_file}")
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                raise RuntimeError(f"❌ Error: Cannot open video file {self.video_file}")
            print(f"✅ Loaded Video File: {self.video_file}")

        elif self.type == "NDI":
            if ndi is None:
                raise RuntimeError("❌ NDI library not installed.")
            if not ndi.initialize(): raise RuntimeError("❌ NDI Init Failed")
            self.ndi_recv = ndi.recv_create_v3()
            if self.ndi_recv is None: raise RuntimeError("❌ NDI Recv Create Failed")
            
            print("🔎 Scanning for NDI Sources (3s)...")
            ndi_find = ndi.find_create_v2()
            if ndi_find is None: raise RuntimeError("❌ NDI Find Create Failed")

            sources = []
            for _ in range(5): # Try 5 times
                ndi.find_wait_for_sources(ndi_find, 1000) # Wait up to 1s for sources to appear
                sources = ndi.find_get_current_sources(ndi_find)
                if sources: 
                    print(f"   Found {len(sources)} source(s).")
                    break
                print("   Waiting for sources...")
            
            # Find specific source or default to first
            target = next((s for s in sources if self.ndi_source_name in s.ndi_name), None)
            
            if not target and sources:
                print(f"⚠️ '{self.ndi_source_name}' not found. Using '{sources[0].ndi_name}'")
                target = sources[0]
            
            if target:
                ndi.recv_connect(self.ndi_recv, target)
                print(f"✅ Connected to NDI: {target.ndi_name}")
                
                # Cleanup finder AFTER connecting
                ndi.find_destroy(ndi_find)

                # Just in case, wait a bit for connection to stabilize
                time.sleep(1.0) 
            else:
                ndi.find_destroy(ndi_find)
                raise RuntimeError("❌ No NDI Sources Found.")

    def get_frame(self):
        if self.type == "WEBCAM" or self.type == "FILE":
            ret, frame = self.cap.read()
            if self.type == "FILE" and not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame
        elif self.type == "NDI":
            if ndi is None: return False, None
            
            # Wait up to 50ms for a frame (avoid CPU spin, ensure we catch data)
            t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 50) 
            
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = np.copy(np.frombuffer(v.data, dtype=np.uint8))
                
                # UYVY (default NDI) to BGR
                if v.FourCC == ndi.FOURCC_VIDEO_TYPE_UYVY:
                    frame = frame.reshape((v.yres, v.xres, 2))  # UYVY is 2 bytes per pixel -> (H, W, 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
                
                # UYVA (NDI with Alpha)
                elif v.FourCC == ndi.FOURCC_VIDEO_TYPE_UYVA:
                    frame = frame.reshape((v.yres, v.xres, 2))  # Treat as UYVY (H, W, 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)

                # BGRA (some sources)
                elif v.FourCC == ndi.FOURCC_VIDEO_TYPE_BGRA:
                     frame = frame.reshape((v.yres, v.xres, 4))
                     frame = frame[:, :, :3] # Drop alpha
                
                # BGRX (some sources)
                elif v.FourCC == ndi.FOURCC_VIDEO_TYPE_BGRX:
                     frame = frame.reshape((v.yres, v.xres, 4))
                     frame = frame[:, :, :3] # Drop alpha

                # Fallback: Assume UYVY if unknown (commonest NDI format) or try standard reshape
                else:
                    try:
                         # Try basic 4-channel reshape first (old assumption)
                         frame = frame.reshape((v.yres, v.xres, 4))[:, :, :3]
                    except ValueError:
                         # If that fails, assume UYVY (2 bytes per pixel)
                         frame = frame.reshape((v.yres, v.xres, 2))
                         frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)

                ndi.recv_free_video_v2(self.ndi_recv, v)
                return True, frame
            
            return False, None
        return False, None

    def release(self):
        if (self.type == "WEBCAM" or self.type == "FILE") and self.cap: self.cap.release()
        elif self.type == "NDI" and self.ndi_recv: 
            if ndi:
                ndi.recv_destroy(self.ndi_recv)
                ndi.destroy()

