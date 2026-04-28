import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


class ProductManager:
    """
    Manages products (configurations).
    Structure:
    produces/
      product_A/
        config.json   (Params, Offsets, Model Path)
        path.npy      (Dispensing dots)
        path_poly.json (Dispensing polylines, safe JSON format)
        product_A.shm (Halcon Model)
        reference.png (Training Image)
    """

    PRODUCT_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")

    def __init__(self, products_dir: str = "produces"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = os.path.abspath(base_dir)
        self.config_dir = os.path.join(self.base_dir, "config")

        # Ensure absolute, normalized products root.
        if not os.path.isabs(products_dir):
            products_dir = os.path.join(self.base_dir, products_dir)
        self.products_dir = os.path.realpath(os.path.abspath(products_dir))
        os.makedirs(self.products_dir, exist_ok=True)

    # ---------- Name/Path safety ----------
    def normalize_product_name(self, name: str) -> str:
        return (name or "").strip()

    def is_valid_product_name(self, name: str) -> bool:
        return bool(self.PRODUCT_NAME_RE.fullmatch(self.normalize_product_name(name)))

    def _assert_valid_product_name(self, name: str):
        if not self.is_valid_product_name(name):
            raise ValueError(
                "Invalid product name. Use 1-64 chars from: letters, numbers, _, -, Chinese."
            )

    def get_product_path(self, name: str) -> str:
        """Get safe absolute product directory path."""
        normalized = self.normalize_product_name(name)
        self._assert_valid_product_name(normalized)

        product_path = os.path.realpath(os.path.join(self.products_dir, normalized))
        if os.path.commonpath([self.products_dir, product_path]) != self.products_dir:
            raise ValueError("Unsafe product path.")
        return product_path

    # ---------- IO helpers ----------
    def _safe_load_npy(self, path: str):
        """Load numeric .npy safely (pickle disabled)."""
        try:
            return np.load(path, allow_pickle=False)
        except ValueError as e:
            if "Object arrays cannot be loaded" in str(e):
                import logging
                logging.warning(f"兼容加载包含对象的旧格式文件: {path}")
                return np.load(path, allow_pickle=True)
            raise

    def _normalize_poly_segments(self, path_polys) -> List[List[List[float]]]:
        segments: List[List[List[float]]] = []
        if path_polys is None:
            return segments

        for seg in path_polys:
            if seg is None:
                continue
            arr = np.asarray(seg, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                if arr.shape[0] != 2:
                    continue
                arr = arr.reshape(1, 2)
            elif arr.ndim != 2 or arr.shape[1] != 2:
                continue
            segments.append(arr.tolist())
        return segments

    def _load_poly_segments(self, product_dir: str):
        """
        Safe poly path loader:
        1) Prefer JSON path format.
        2) Fallback to legacy .npy ONLY when it is numeric (pickle disabled).
        """
        json_path = os.path.join(product_dir, "path_poly.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_segments = data.get("segments", [])
            return [np.asarray(seg, dtype=np.float32) for seg in raw_segments if seg]

        legacy_path = os.path.join(product_dir, "path_poly.npy")
        if not os.path.exists(legacy_path):
            return None

        arr = self._safe_load_npy(legacy_path)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2:
            return [arr.astype(np.float32)]

        raise ValueError(
            "Unsafe or unsupported legacy path_poly.npy format. "
            "Please re-save product to migrate to path_poly.json."
        )

    # ---------- Public APIs ----------
    def list_products(self):
        """Returns a list of available (valid) product names."""
        if not os.path.exists(self.products_dir):
            return []

        names = []
        for d in os.listdir(self.products_dir):
            full = os.path.join(self.products_dir, d)
            if os.path.isdir(full) and self.is_valid_product_name(d):
                names.append(d)
        return sorted(names)

    def save_product(self, name, config_data, path_points=None, path_polys=None, source_model_path=None):
        """
        Save a product.
        :param name: Product Name (Folder Name)
        :param config_data: Dict of parameters
        :param path_points: List or Array of DOT points (path.npy)
        :param path_polys: List/Array of LINE segments (saved as path_poly.json)
        """
        try:
            product_dir = self.get_product_path(name)
        except ValueError as e:
            return False, str(e)

        os.makedirs(product_dir, exist_ok=True)

        # Load current valve parameters and add to config
        valve_params_file = Path(self.config_dir) / "valve_dispense_params.json"
        if valve_params_file.exists():
            try:
                with open(valve_params_file, "r", encoding="utf-8") as f:
                    valve_params = json.load(f)
                params = valve_params.get("parameters", {})
                config_data["valve_prefeed_time"] = params.get("prefeed_time", 0.10)
                config_data["valve_a_dispense_time"] = params.get("valve_a_dispense_time", 0.50)
                config_data["valve_b_dispense_time"] = params.get("valve_b_dispense_time", 0.50)
                config_data["line_speed_xy"] = params.get("line_speed_xy", 800)
            except Exception as e:
                print(f"[Warn] Could not load valve params, using defaults: {e}")
                config_data["valve_prefeed_time"] = 0.10
                config_data["valve_a_dispense_time"] = 0.50
                config_data["valve_b_dispense_time"] = 0.50
                config_data["line_speed_xy"] = 800
        else:
            config_data["valve_prefeed_time"] = 0.10
            config_data["valve_a_dispense_time"] = 0.50
            config_data["valve_b_dispense_time"] = 0.50
            config_data["line_speed_xy"] = 800

        # Save Config JSON
        # [FIX] Load existing config first and MERGE, to preserve training-time fields
        # (e.g. roi_in_reference, recognized_coordinates, laser_data) that are NOT
        # included in config_data passed by the UI save button.
        config_path = os.path.join(product_dir, "config.json")
        try:
            existing_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            # Merge: new values overwrite, existing-only fields are preserved
            merged_config = {**existing_config, **config_data}
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return False, f"Failed to save config: {e}"


        # Save dots path
        if path_points is not None:
            npy_path = os.path.join(product_dir, "path.npy")
            try:
                np.save(npy_path, np.asarray(path_points, dtype=np.float32))
            except Exception as e:
                return False, f"Failed to save path(dots): {e}"

        # Save polylines as JSON (safe, no pickle required)
        if path_polys is not None:
            poly_path = os.path.join(product_dir, "path_poly.json")
            try:
                payload = {"version": 2, "segments": self._normalize_poly_segments(path_polys)}
                with open(poly_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                return False, f"Failed to save path(poly): {e}"

        return True, f"Product '{self.normalize_product_name(name)}' saved successfully."

    def load_product(self, name):
        """
        Load product data.
        :return: (config_dict, dots_data, polys_data, msg)
        """
        try:
            product_dir = self.get_product_path(name)
        except ValueError as e:
            return None, None, None, str(e)

        if not os.path.exists(product_dir):
            return None, None, None, f"Product '{name}' not found."

        # Load config
        config_path = os.path.join(product_dir, "config.json")
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                return None, None, None, f"Error loading config.json: {e}"
        else:
            return None, None, None, "config.json missing."

        # Apply valve params to global config
        valve_params = {
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "prefeed_time": config_data.get("valve_prefeed_time", 0.10),
                "valve_a_dispense_time": config_data.get("valve_a_dispense_time", 0.50),
                "valve_b_dispense_time": config_data.get("valve_b_dispense_time", 0.50),
                "line_speed_xy": config_data.get("line_speed_xy", 800),
            },
            "notes": f"Loaded from product '{name}'",
        }

        valve_params_dir = Path(self.config_dir)
        valve_params_dir.mkdir(exist_ok=True)
        valve_params_file = valve_params_dir / "valve_dispense_params.json"
        try:
            with open(valve_params_file, "w", encoding="utf-8") as f:
                json.dump(valve_params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warn] Could not save valve params: {e}")

        # Load paths
        dots_data, poly_data = self.load_product_paths(name)
        return config_data, dots_data, poly_data, f"Loaded '{name}'."

    def load_product_config(self, name):
        """
        Load only config data from product.
        :return: config_dict or None
        """
        try:
            product_dir = self.get_product_path(name)
        except ValueError as e:
            print(f"[Warn] Invalid product name: {e}")
            return None

        config_path = os.path.join(product_dir, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warn] Failed to load config: {e}")
            return None

    def load_product_paths(self, name):
        """
        Load only paths from product.
        :return: (dots_data, poly_data)
        """
        try:
            product_dir = self.get_product_path(name)
        except ValueError as e:
            print(f"[Warn] Invalid product name: {e}")
            return None, None

        # Dots
        dots_path = os.path.join(product_dir, "path.npy")
        dots_data = None
        if os.path.exists(dots_path):
            try:
                dots_data = self._safe_load_npy(dots_path)
            except Exception as e:
                print(f"[Warn] Failed to load path.npy: {e}")

        # Polylines
        poly_data = None
        try:
            poly_data = self._load_poly_segments(product_dir)
        except Exception as e:
            print(f"[Warn] Failed to load poly path: {e}")

        return dots_data, poly_data
