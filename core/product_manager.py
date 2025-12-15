import os
import json
import shutil
import numpy as np
from datetime import datetime

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

        # 2. Save Config JSON
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
