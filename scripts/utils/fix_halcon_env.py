import os
import numpy as np
import re

def fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    slot_path = os.path.join(config_dir, "slot_templates.npy")
    script_path = os.path.join(base_dir, "scripts", "detect_mold.hdev")
    model_path = os.path.join(config_dir, "mold_model_scaled.shm")

    print(f"Checking {slot_path}...")
    if not os.path.exists(slot_path):
        print("Slot templates not found.")
        return

    slots = np.load(slot_path, allow_pickle=True)
    if len(slots) == 0:
        print("Slots empty.")
        return

    # Cell 0 ROI
    box = slots[0]
    r1 = np.min(box[:, 1])
    c1 = np.min(box[:, 0])
    r2 = np.max(box[:, 1])
    c2 = np.max(box[:, 0])
    print(f"Valid ROI found: [{r1:.1f}, {c1:.1f}, {r2:.1f}, {c2:.1f}]")

    # Update Script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"(gen_rectangle1\s*\(\s*ModelRegion\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*,\s*)([\d\.]+)(\s*\))"
    replacement = f"\\g<1>{r1:.3f}\\g<3>{c1:.3f}\\g<5>{r2:.3f}\\g<7>{c2:.3f}\\g<9>"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Updated detect_mold.hdev ROI.")
    else:
        print("Info: ROI in script was already correct.")

    # Delete Model
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"🗑️ Deleted {model_path}.")
        except Exception as e:
            print(f"❌ Failed to delete model: {e}")
    else:
        print(f"Info: {model_path} does not exist (will be created).")

if __name__ == "__main__":
    fix()
