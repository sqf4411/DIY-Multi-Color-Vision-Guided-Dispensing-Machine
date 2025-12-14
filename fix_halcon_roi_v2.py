import os
import numpy as np
import cv2
import re

def get_combinations(pts):
    # Sort by x
    pts = pts[np.argsort(pts[:, 0]), :]
    # Left two, Right two
    left = pts[:2, :]
    right = pts[2:, :]
    # Sort left by y -> tl, bl
    left = left[np.argsort(left[:, 1]), :]
    tl, bl = left[0], left[1]
    # Sort right by y -> tr, br
    right = right[np.argsort(right[:, 1]), :]
    tr, br = right[0], right[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def fix_v2():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    mold2_path = os.path.join(config_dir, "mold2.png")
    slot_path = os.path.join(config_dir, "slot_templates.npy")
    script_path = os.path.join(base_dir, "scripts", "detect_mold.hdev")
    model_path = os.path.join(config_dir, "mold_model_scaled.shm")
    
    if not os.path.exists(mold2_path):
        print("Error: mold2.png not found")
        return
        
    img = cv2.imread(mold2_path)
    if img is None:
        print("Error: Failed to load mold2.png")
        return
        
    # 1. Detect Big Box to get Minv
    from core.mold_detector import detect_mold_and_cells
    
    # Use grayscale just like golden_teach.py
    gray_img = cv2.imread(mold2_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print("Error: Failed to load mold2.png as grayscale")
        return

    _, big_box, _, _ = detect_mold_and_cells(gray_img)
    
    if big_box is None:
        print("Error: Could not find 4-corner mold in mold2.png using detect_mold_and_cells")
        return

    pts = big_box.astype(np.float32)
    rect_pts = get_combinations(pts)
    
    print(f"Mold Corners (Raw): \n{rect_pts}")
    
    # Destination points (Golden)
    warp_size_path = os.path.join(config_dir, "warp_size.npy")
    if os.path.exists(warp_size_path):
         warp_wh = np.load(warp_size_path)
         goldW, goldH = int(warp_wh[0]), int(warp_wh[1])
         print(f"Loaded Warp Size: {goldW}x{goldH}")
    else:
         w_raw = np.linalg.norm(rect_pts[0] - rect_pts[1])
         h_raw = np.linalg.norm(rect_pts[0] - rect_pts[3])
         goldW, goldH = int(w_raw), int(h_raw)
         print(f"Estimated Warp Size: {goldW}x{goldH}")
         
    dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)
    
    Minv = cv2.getPerspectiveTransform(dst_pts, rect_pts)
    print(f"Minv:\n{Minv}")
    
    # 2. Transform Slot Box
    slots = np.load(slot_path, allow_pickle=True)
    if len(slots) == 0:
        print("Error: No slots")
        return
        
    box_warped = slots[0] # (4, 2)
    print(f"Slot (Warped):\n{box_warped}")
    
    # Transform points
    pts_w = box_warped.reshape(-1, 1, 2).astype(np.float32)
    pts_raw = cv2.perspectiveTransform(pts_w, Minv).reshape(-1, 2)
    print(f"Slot (Raw Transformed):\n{pts_raw}")
    
    # 3. Calculate Bounding Box of Raw Points
    r1 = np.min(pts_raw[:, 1])
    c1 = np.min(pts_raw[:, 0])
    r2 = np.max(pts_raw[:, 1])
    c2 = np.max(pts_raw[:, 0])
    
    width = c2 - c1
    height = r2 - r1
    print(f"Calculated ROI: r1={r1:.1f}, c1={c1:.1f}, r2={r2:.1f}, c2={c2:.1f} (W={width:.1f}, H={height:.1f})")
    
    if width < 5 or height < 5:
        print(f"❌ Error: Calculated ROI is too small! Aborting update.")
        print(f"DEBUG: rect_pts (Mold Corners) = \n{rect_pts}")
        print(f"DEBUG: dst_pts (Warped Dst) = \n{dst_pts}")
        return

    print(f"Corrected Raw ROI: [{r1:.1f}, {c1:.1f}, {r2:.1f}, {c2:.1f}]")
    
    # 4. Update Script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"(gen_rectangle1\s*\(\s*ModelRegion\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*\))"
    replacement = f"\\g<1>{r1:.3f}\\g<3>{c1:.3f}\\g<5>{r2:.3f}\\g<7>{c2:.3f}\\g<9>"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Updated detect_mold.hdev with Corrected ROI.")
    else:
        print("Info: Script ROI unchanged.")

    # 5. Delete Bad Model
    if os.path.exists(model_path):
        os.remove(model_path)
        print("🗑️ Deleted old model file.")

if __name__ == "__main__":
    try:
        fix_v2()
    except Exception as e:
        print(f"Exception: {e}")
