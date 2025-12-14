import os
import json
import shutil
import numpy as np
from datetime import datetime

class RecipeManager:
    """
    Manages product recipes (configurations).
    Structure:
    recipes/
      Product_A/
        config.json  (Params, Offsets, Model Path)
        path.npy     (Dispensing Path)
        model.shm    (Halcon Model, optional if shared)
        image.png    (Training Image)
    """

    def __init__(self, recipes_dir="recipes"):
        self.recipes_dir = recipes_dir
        if not os.path.exists(self.recipes_dir):
            os.makedirs(self.recipes_dir)

    def get_recipe_path(self, name):
        return os.path.join(self.recipes_dir, name)

    def list_recipes(self):
        """Returns a list of available recipe names."""
        if not os.path.exists(self.recipes_dir):
            return []
        return [d for d in os.listdir(self.recipes_dir) 
                if os.path.isdir(os.path.join(self.recipes_dir, d))]

    def save_recipe(self, name, config_data, path_points=None, path_polys=None, source_model_path=None):
        """
        Save a recipe.
        :param name: Recipe Name (Folder Name)
        :param config_data: Dict of parameters
        :param path_points: List or Array of DOT points (path.npy)
        :param path_polys: List or Array of LINE segments (path_poly.npy)
        """
        recipe_dir = self.get_recipe_path(name)
        if not os.path.exists(recipe_dir):
            os.makedirs(recipe_dir)

        # 2. Save Config JSON
        config_path = os.path.join(recipe_dir, "config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return False, f"Failed to save config: {e}"

        # 3. Save Path (Dots -> path.npy)
        if path_points is not None:
            npy_path = os.path.join(recipe_dir, "path.npy")
            try:
                np.save(npy_path, np.array(path_points))
            except Exception as e:
                return False, f"Failed to save path(dots): {e}"
                
        # 4. Save Path (Polys -> path_poly.npy)
        if path_polys is not None:
            poly_path = os.path.join(recipe_dir, "path_poly.npy")
            try:
                np.save(poly_path, np.array(path_polys, dtype=object))
            except Exception as e:
                return False, f"Failed to save path(poly): {e}"

        return True, f"Recipe '{name}' saved successfully."

    def load_recipe(self, name):
        """
        Load recipe data.
        :return: (config_dict, dots_data, polys_data, msg)
        """
        recipe_dir = self.get_recipe_path(name)
        if not os.path.exists(recipe_dir):
            return None, None, None, f"Recipe '{name}' not found."

        # 1. Load Config
        config_path = os.path.join(recipe_dir, "config.json")
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
        npy_path = os.path.join(recipe_dir, "path.npy")
        dots_data = None
        if os.path.exists(npy_path):
            try:
                dots_data = np.load(npy_path)
            except Exception as e:
                print(f"[Warn] Failed to load path.npy: {e}")
                
        # 3. Load Path (Polys)
        poly_path = os.path.join(recipe_dir, "path_poly.npy")
        poly_data = None
        if os.path.exists(poly_path):
            try:
                poly_data = np.load(poly_path, allow_pickle=True)
            except Exception as e:
                print(f"[Warn] Failed to load path_poly.npy: {e}")

        return config_data, dots_data, poly_data, f"Loaded '{name}'."
