import json
import os
import shutil
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Centralized configuration manager for the application.
    Handles loading and saving of:
    1. App Settings (app_settings.json) - UI state, motion params, etc.
    2. Camera Params (config/params.json) - Exposure, Gain, Gamma.
    """
    
    APP_SETTINGS_FILE = "config/app_settings.json"
    CAM_PARAMS_FILE = "config/params.json"
    
    DEFAULTS = {
        "feedrate_xy": 3000.0,
        "feedrate_z": 500.0,
        "accel": 500.0,
        "safe_z_cavity": 5.0,
        "safe_z_product": 20.0,
        "work_z": 0.0,
        "jog_step": 1.0,
        "z_base": 0.0
    }

    def __init__(self):
        self.app_settings: Dict[str, Any] = self.DEFAULTS.copy()
        self.cam_params: Dict[str, Any] = {}
        
        self.ensure_config_dir()
        self.load_all()

    def ensure_config_dir(self):
        os.makedirs("config", exist_ok=True)

    def load_all(self):
        self.load_app_settings()
        self.load_cam_params()

    def load_app_settings(self):
        if not os.path.exists(self.APP_SETTINGS_FILE):
            return

        try:
            with open(self.APP_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Update existing settings with loaded data
                self.app_settings.update(data)
                print(f"[ConfigManager] Loaded app settings: {self.APP_SETTINGS_FILE}")
        except Exception as e:
            print(f"[ConfigManager] Failed to load app settings: {e}")
            # Optional: Backup corrupted file
            if os.path.exists(self.APP_SETTINGS_FILE):
                shutil.copy(self.APP_SETTINGS_FILE, self.APP_SETTINGS_FILE + ".bak")
                print(f"[ConfigManager] Backed up corrupted file to .bak")

    def save_app_settings(self):
        try:
            with open(self.APP_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.app_settings, f, indent=4, ensure_ascii=False)
            print("[ConfigManager] App settings saved.")
        except Exception as e:
            print(f"[ConfigManager] Failed to save app settings: {e}")

    def get(self, key: str, default=None):
        return self.app_settings.get(key, default)

    def set(self, key: str, value: Any):
        self.app_settings[key] = value

    # --- Camera Params ---
    def load_cam_params(self):
        if not os.path.exists(self.CAM_PARAMS_FILE):
            return

        try:
            with open(self.CAM_PARAMS_FILE, 'r', encoding='utf-8') as f:
                self.cam_params = json.load(f)
            print(f"[ConfigManager] Loaded camera params: {self.CAM_PARAMS_FILE}")
        except Exception as e:
            print(f"[ConfigManager] Failed to load camera params: {e}")

    def save_cam_params(self, params: Dict[str, Any]):
        """
        Save camera parameters.
        :param params: dict containing 'ExposureTime', 'Gain', 'Gamma' etc.
        """
        self.cam_params.update(params)
        try:
            with open(self.CAM_PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cam_params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ConfigManager] Failed to save camera params: {e}")

    def get_cam_param(self, key: str, default=None):
        return self.cam_params.get(key, default)
