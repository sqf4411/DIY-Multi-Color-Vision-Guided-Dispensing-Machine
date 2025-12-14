import json
import os
import shutil
from typing import Dict, Any, Optional

class ConfigManager:
    """
    应用程序的集中式配置管理器。
    处理以下内容的加载和保存：
    1. 应用对设置 (app_settings.json) - UI 状态、运动参数等。
    2. 相机参数 (config/params.json) - 曝光、增益、Gamma。
    """
    
    # Calculate absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project root (core/../)
    CONFIG_DIR = os.path.join(BASE_DIR, "config")
    
    APP_SETTINGS_FILE = os.path.join(CONFIG_DIR, "app_settings.json")
    CAM_PARAMS_FILE = os.path.join(CONFIG_DIR, "params.json")
    
    DEFAULTS = {
        "feedrate_xy": 3000.0,
        "feedrate_z": 500.0,
        "accel": 500.0,
        "safe_z_cavity": 5.0,
        "safe_z_product": 20.0,
        "work_z": 0.0,
        "jog_step": 1.0,
        "z_base": 0.0,
        # Nozzle offset parameters
        "nozzle_offset_x": 0.0,
        "nozzle_offset_y": 0.0,
        "nozzle_offset_z": 0.0,
        # Vision/Camera offset parameters
        "vision_offset_x": 0.0,
        "vision_offset_y": 0.0,
        # Camera-specific Z heights
        "cam_safe_z_cavity": 5.0,
        "cam_safe_z_product": 20.0,
        "cam_work_z": 0.0
    }

    def __init__(self):
        self.app_settings: Dict[str, Any] = self.DEFAULTS.copy()
        self.cam_params: Dict[str, Any] = {}
        
        self.ensure_config_dir()
        self.load_all()

    def ensure_config_dir(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)

    def load_all(self):
        self.load_app_settings()
        self.load_cam_params()

    def load_app_settings(self):
        if not os.path.exists(self.APP_SETTINGS_FILE):
            return

        try:
            with open(self.APP_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 使用加载的数据更新现有设置
                self.app_settings.update(data)
                print(f"[ConfigManager] 已加载应用设置: {self.APP_SETTINGS_FILE}")
        except Exception as e:
            print(f"[ConfigManager] 加载应用设置失败: {e}")
            # 可选：备份损坏的文件
            if os.path.exists(self.APP_SETTINGS_FILE):
                shutil.copy(self.APP_SETTINGS_FILE, self.APP_SETTINGS_FILE + ".bak")
                print(f"[ConfigManager] 已将损坏的文件备份到 .bak")

    def save_app_settings(self):
        try:
            with open(self.APP_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.app_settings, f, indent=4, ensure_ascii=False)
            print("[ConfigManager] 应用设置已保存。")
        except Exception as e:
            print(f"[ConfigManager] 保存应用设置失败: {e}")

    def get(self, key: str, default=None):
        return self.app_settings.get(key, default)

    def set(self, key: str, value: Any):
        self.app_settings[key] = value

    # --- 相机参数 ---
    def load_cam_params(self):
        if not os.path.exists(self.CAM_PARAMS_FILE):
            return

        try:
            with open(self.CAM_PARAMS_FILE, 'r', encoding='utf-8') as f:
                self.cam_params = json.load(f)
            print(f"[ConfigManager] 已加载相机参数: {self.CAM_PARAMS_FILE}")
        except Exception as e:
            print(f"[ConfigManager] 加载相机参数失败: {e}")

    def save_cam_params(self, params: Dict[str, Any]):
        """
        保存相机参数。
        :param params: 包含 'ExposureTime', 'Gain', 'Gamma' 等的字典
        """
        self.cam_params.update(params)
        try:
            with open(self.CAM_PARAMS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cam_params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ConfigManager] 保存相机参数失败: {e}")

    def get_cam_param(self, key: str, default=None):
        return self.cam_params.get(key, default)
