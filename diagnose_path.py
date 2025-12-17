import os
import json
import numpy as np

def diagnose(product_name="opt1"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    product_dir = os.path.join(base_dir, "produces", product_name)
    
    print(f"=== Diagnosing Product: {product_name} ===")
    print(f"Dir: {product_dir}")
    
    if not os.path.exists(product_dir):
        print("Product directory not found!")
        return

    # 1. Check Config
    config_path = os.path.join(product_dir, "product_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"\n[Config] model_row: {config.get('model_row')}")
            print(f"[Config] model_col: {config.get('model_col')}")
        except Exception as e:
            print(f"Error reading config: {e}")
    else:
        print("\n[Config] product_config.json not found!")

    # 2. Check JSON Path
    json_path = os.path.join(product_dir, "multicolor_path.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get('segments', [])
            print(f"\n[JSON Path] Segments count: {len(segments)}")
            for i, seg in enumerate(segments):
                pts = seg.get('points', [])
                if pts:
                    print(f"  Seg {i} ({seg.get('type')}): First point {pts[0]}")
                else:
                    print(f"  Seg {i} ({seg.get('type')}): No points")
        except Exception as e:
            print(f"Error reading path json: {e}")
    else:
        print("\n[JSON Path] multicolor_path.json not found!")

    # 3. Check Legacy NPY
    npy_path_dots = os.path.join(product_dir, "path.npy")
    if os.path.exists(npy_path_dots):
        try:
            dots = np.load(npy_path_dots)
            print(f"\n[NPY Dots] Shape: {dots.shape}")
            if len(dots) > 0:
                print(f"  First dot: {dots[0]}")
        except Exception as e:
            print(f"Error reading path.npy: {e}")
            
    npy_path_poly = os.path.join(product_dir, "path_poly.npy")
    if os.path.exists(npy_path_poly):
        try:
            polys = np.load(npy_path_poly, allow_pickle=True)
            print(f"\n[NPY Polys] Count: {len(polys)}")
            if len(polys) > 0:
                print(f"  First poly shape: {len(polys[0])}")
        except Exception as e:
            print(f"Error reading path_poly.npy: {e}")

if __name__ == "__main__":
    diagnose()
