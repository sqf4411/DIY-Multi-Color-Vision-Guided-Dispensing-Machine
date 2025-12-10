import sys
import os
import time
import ctypes
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# Import MvImport SDK
# Assuming MvImport is in the project root or accessible via sys.path
# In main_camera_ui.py there was a sys.path append, we might need to handle it or assume it's set.
# For now, we assume MvImport is importable.
try:
    from MvImport.MvCameraControl_class import *
except ImportError:
    # If standard import fails, try appending the path relative to this file
    # But usually it's better to rely on the environment being efficient.
    # Let's try to match what main_camera_ui.py did but cleanly.
    pass

class CameraWorker(QThread):
    frame_received = pyqtSignal(np.ndarray)
    
    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.running = True

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        st_param = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            return

        data_size = st_param.nCurValue
        data_buff = (ctypes.c_ubyte * data_size)()
        frame_info = MV_FRAME_OUT_INFO_EX()

        while self.running:
            ret = self.cam.MV_CC_GetOneFrameTimeout(data_buff, data_size, frame_info, 1000)
            if ret == 0:
                w, h = frame_info.nWidth, frame_info.nHeight
                img = np.frombuffer(data_buff, dtype=np.uint8)[: w * h]
                img = img.reshape((h, w))
                self.frame_received.emit(img)
            else:
                pass 
                # potentially handle timeout/error, but keep loop for robustness? 
                # or just ignore timeouts.

class CameraManager:
    """
    Encapsulates MvCamera operations.
    """
    def __init__(self):
        self.cam = None
        self.worker = None
        
        # Ranges
        self.exp_range = (100.0, 10000.0)
        self.gain_range = (0.0, 10.0)
        self.gamma_range = (0.5, 2.0)
        
        # Info
        self.is_open = False

    def open_camera(self) -> bool:
        if self.cam:
            return True

        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

        if deviceList.nDeviceNum == 0:
            return False

        dev_info = ctypes.cast(
            deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)
        ).contents

        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            self.cam = None
            return False

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            return False
            
        self.is_open = True
        
        # Init defaults
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)
        try:
            self.cam.MV_CC_SetBoolValue("GammaEnable", True)
        except:
            pass
            
        self._read_ranges()
        return True

    def start_grabbing(self):
        if not self.cam or not self.is_open:
            return False
            
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            return False
            
        self.worker = CameraWorker(self.cam)
        self.worker.start()
        return True

    def stop_grabbing(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            
        if self.cam and self.is_open:
            self.cam.MV_CC_StopGrabbing()

    def close_camera(self):
        self.stop_grabbing()
        if self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
            self.is_open = False

    def get_frame_signal(self):
        if self.worker:
            return self.worker.frame_received
        return None

    def set_exposure(self, val: float):
        if not self.cam: return
        self.cam.MV_CC_SetFloatValue("ExposureTime", float(val))

    def set_gain(self, val: float):
        if not self.cam: return
        self.cam.MV_CC_SetFloatValue("Gain", float(val))

    def set_gamma(self, val: float):
        if not self.cam: return
        try:
            self.cam.MV_CC_SetFloatValue("Gamma", float(val))
        except:
            pass

    def get_param_ranges(self):
        return self.exp_range, self.gain_range, self.gamma_range

    def _read_ranges(self):
        if not self.cam: return
        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("ExposureTime", v)
            self.exp_range = (v.fMin, v.fMax)
        except: pass
        
        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gain", v)
            self.gain_range = (v.fMin, v.fMax)
        except: pass

        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gamma", v)
            self.gamma_range = (v.fMin, v.fMax)
        except: pass
    
    def get_params(self):
        # Read current params from camera
        res = {}
        if not self.cam: return res
        
        v = MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue("ExposureTime", v) == 0:
            res['ExposureTime'] = v.fCurValue
            
        if self.cam.MV_CC_GetFloatValue("Gain", v) == 0:
            res['Gain'] = v.fCurValue
            
        if self.cam.MV_CC_GetFloatValue("Gamma", v) == 0:
            res['Gamma'] = v.fCurValue
            
        return res

    def get_cam_info(self):
        """
        Returns a dict with: 'width', 'height', 'fps', 'pixel_format_str'
        """
        info = {'width': 0, 'height': 0, 'fps': 0.0, 'pixel_format_str': '-'}
        if not self.cam: return info
        
        try:
            w_val = MVCC_INTVALUE()
            self.cam.MV_CC_GetIntValue("Width", w_val)
            info['width'] = w_val.nCurValue
            
            h_val = MVCC_INTVALUE()
            self.cam.MV_CC_GetIntValue("Height", h_val)
            info['height'] = h_val.nCurValue
        except: pass
        
        try:
            fps_val = MVCC_FLOATVALUE()
            ret = self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", fps_val)
            if ret != 0:
                self.cam.MV_CC_GetFloatValue("ResultingFrameRate", fps_val)
            info['fps'] = fps_val.fCurValue
        except: pass
        
        try:
            enum_val = MVCC_ENUMVALUE()
            ret = self.cam.MV_CC_GetEnumValue("PixelFormat", enum_val)
            if ret == 0:
                if enum_val.nCurValue == PixelType_Gvsp_Mono8:
                    info['pixel_format_str'] = "Mono8"
                else:
                    info['pixel_format_str'] = f"0x{enum_val.nCurValue:X}"
        except: pass
        
        return info
