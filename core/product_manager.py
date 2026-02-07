import os
import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

class ProductManager:
    """
    Manages products (configurations).
    Structure:
    produces/
      product_A/
        config.json  (Params, Offsets, Model Path)
        path.npy     (Dispensing Path)
        path_poly.npy (Poly Path)
        product_A.shm (Halcon Model)
        reference.png (Training Image)
    """

    def __init__(self, products_dir="produces"):
        # 确保使用绝对路径
        if not os.path.isabs(products_dir):
            # 相对于项目根目录（product_manager.py在core/目录下）
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.products_dir = os.path.join(base_dir, products_dir)
        else:
            self.products_dir = products_dir
        
        if not os.path.exists(self.products_dir):
            os.makedirs(self.products_dir)

    def get_product_path(self, name):
        """获取产品目录的绝对路径"""
        return os.path.abspath(os.path.join(self.products_dir, name))

    def list_products(self):
        """Returns a list of available product names."""
        if not os.path.exists(self.products_dir):
            return []
        return [d for d in os.listdir(self.products_dir) 
                if os.path.isdir(os.path.join(self.products_dir, d))]

    def save_product(self, name, config_data, path_points=None, path_polys=None, source_model_path=None):
        """
        Save a product.
        :param name: Product Name (Folder Name)
        :param config_data: Dict of parameters
        :param path_points: List or Array of DOT points (path.npy)
        :param path_polys: List or Array of LINE segments (path_poly.npy)
        """
        product_dir = self.get_product_path(name)
        if not os.path.exists(product_dir):
            os.makedirs(product_dir)

        # 2. Load current valve parameters and add to config
        valve_params_file = Path("config") / "valve_dispense_params.json"
        if valve_params_file.exists():
            try:
                with open(valve_params_file, 'r', encoding='utf-8') as f:
                    valve_params = json.load(f)
                # Add valve parameters to config_data
                params = valve_params.get("parameters", {})
                config_data["valve_prefeed_time"] = params.get("prefeed_time", 0.10)
                config_data["valve_a_dispense_time"] = params.get("valve_a_dispense_time", 0.50)
                config_data["valve_b_dispense_time"] = params.get("valve_b_dispense_time", 0.50)
                config_data["line_speed_xy"] = params.get("line_speed_xy", 800)  # [NEW] 画线速度
            except Exception as e:
                print(f"[Warn] Could not load valve params, using defaults: {e}")
                config_data["valve_prefeed_time"] = 0.10
                config_data["valve_a_dispense_time"] = 0.50
                config_data["valve_b_dispense_time"] = 0.50
                config_data["line_speed_xy"] = 800
        else:
            # Use default values
            config_data["valve_prefeed_time"] = 0.10
            config_data["valve_a_dispense_time"] = 0.50
            config_data["valve_b_dispense_time"] = 0.50
            config_data["line_speed_xy"] = 800
        
        # 3. Save Config JSON
        config_path = os.path.join(product_dir, "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return False, f"Failed to save config: {e}"

        # 3. Save Path (Dots -> path.npy)
        if path_points is not None:
            npy_path = os.path.join(product_dir, "path.npy")
            try:
                np.save(npy_path, np.array(path_points))
            except Exception as e:
                return False, f"Failed to save path(dots): {e}"
                
        # 4. Save Path (Polys -> path_poly.npy)
        if path_polys is not None:
            poly_path = os.path.join(product_dir, "path_poly.npy")
            try:
                np.save(poly_path, np.array(path_polys, dtype=object))
            except Exception as e:
                return False, f"Failed to save path(poly): {e}"

        return True, f"Product '{name}' saved successfully."

    def load_product(self, name):
        """
        Load product data.
        :return: (config_dict, dots_data, polys_data, msg)
        """
        product_dir = self.get_product_path(name)
        if not os.path.exists(product_dir):
            return None, None, None, f"Product '{name}' not found."

        # 1. Load Config
        config_path = os.path.join(product_dir, "config.json")
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                return None, None, None, f"Error loading config.json: {e}"
        else:
            return None, None, None, "config.json missing."
        
        # 1.5 Apply valve parameters from config to global valve params file
        valve_params = {
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "prefeed_time": config_data.get("valve_prefeed_time", 0.10),
                "valve_a_dispense_time": config_data.get("valve_a_dispense_time", 0.50),
                "valve_b_dispense_time": config_data.get("valve_b_dispense_time", 0.50),
                "line_speed_xy": config_data.get("line_speed_xy", 800)  # [NEW] 画线速度
            },
            "notes": f"Loaded from product '{name}'"
        }
        
        print(f"[ProductManager] 加载产品 '{name}' 的点胶参数:")
        print(f"[ProductManager]   供料提前: {valve_params['parameters']['prefeed_time']}")
        print(f"[ProductManager]   阀A时间: {valve_params['parameters']['valve_a_dispense_time']}")
        print(f"[ProductManager]   阀B时间: {valve_params['parameters']['valve_b_dispense_time']}")
        print(f"[ProductManager]   画线速度: {valve_params['parameters']['line_speed_xy']} mm/min")
        
        valve_params_dir = Path("config")
        valve_params_dir.mkdir(exist_ok=True)
        valve_params_file = valve_params_dir / "valve_dispense_params.json"
        
        print(f"[ProductManager] 写入参数文件: {valve_params_file.absolute()}")
        
        try:
            with open(valve_params_file, 'w', encoding='utf-8') as f:
                json.dump(valve_params, f, indent=2, ensure_ascii=False)
            print(f"[ProductManager] ✅ 已应用产品'{name}'的阀门参数到全局配置")
        except Exception as e:
            print(f"[Warn] Could not save valve params: {e}")

        # 2. Load Path (Dots)
        npy_path = os.path.join(product_dir, "path.npy")
        dots_data = None
        if os.path.exists(npy_path):
            try:
                dots_data = np.load(npy_path)
            except Exception as e:
                print(f"[Warn] Failed to load path.npy: {e}")
                
        # 3. Load Path (Polys)
        poly_path = os.path.join(product_dir, "path_poly.npy")
        poly_data = None
        if os.path.exists(poly_path):
            try:
                poly_data = np.load(poly_path, allow_pickle=True)
            except Exception as e:
                print(f"[Warn] Failed to load path_poly.npy: {e}")

        return config_data, dots_data, poly_data, f"Loaded '{name}'."
    
    def load_product_config(self, name):
        """
        Load only config data from product
        :return: config_dict or None
        """
        product_dir = self.get_product_path(name)
        config_path = os.path.join(product_dir, "config.json")
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warn] Failed to load config: {e}")
            return None
    
    def load_product_paths(self, name):
        """
        Load only paths from product
        :return: (dots_data, poly_data)
        """
        product_dir = self.get_product_path(name)
        
        # Load dots
        dots_path = os.path.join(product_dir, "path.npy")
        dots_data = None
        if os.path.exists(dots_path):
            try:
                dots_data = np.load(dots_path)
            except Exception as e:
                print(f"[Warn] Failed to load path.npy: {e}")
        
        # Load polys
        poly_path = os.path.join(product_dir, "path_poly.npy")
        poly_data = None
        if os.path.exists(poly_path):
            try:
                poly_data = np.load(poly_path, allow_pickle=True)
            except Exception as e:
                print(f"[Warn] Failed to load path_poly.npy: {e}")
        
        return dots_data, poly_data

    def get_product_config(self, name):
        """
        Get product configuration dict
        """
        conf = self.load_product_config(name)
        return conf if conf else {}

    def update_product_config(self, name, new_config):
        """
        Update specific keys in product config
        """
        # 1. Load existing
        current_conf = self.get_product_config(name)
        
        # 2. Update (Merge)
        current_conf.update(new_config)
        
        # 3. Save back (Preserve paths by not passing them, wait, save_product overwrites?)
        # save_product writes config.json. 
        # CAUTION: save_product implementation above writes config.json efficiently 
        # but also writes path.npy if provided. If path is None, it ignores it.
        # So we can just call save_product with config only.
        
        return self.save_product(name, current_conf, path_points=None, path_polys=None)
