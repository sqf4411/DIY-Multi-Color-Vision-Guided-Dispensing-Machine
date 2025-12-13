
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mold_detector import _find_big_mold

def check_mold():
    path = "config/mold2.png"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    print(f"🔍 Checking {path}...")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to read image")
        return
    
    h, w = img.shape
    print(f"📏 Image size: {w}x{h}")

    rect, box = _find_big_mold(img)
    
    if box is not None:
        xs = box[:, 0]
        ys = box[:, 1]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        print(f"✅ Box Found!")
        print(f"   - Box Size: {bw} x {bh}")
        print(f"   - Center: {rect[0]}")
        print(f"   - Angle: {rect[2]}")
        
        # Save debug image
        debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug, [box], 0, (0, 0, 255), 5)
        cv2.imwrite("debug_mold_check.jpg", debug)
        print("📸 Saved debug_mold_check.jpg")
        
        if bw < 500 or bh < 500:
             print("⚠️ WARNING: Box seems too small! Likely detection failure.")
    else:
        print("❌ No Box Found!")

if __name__ == "__main__":
    check_mold()
