import halcon as ha
import numpy as np
import ctypes

try:
    # Create valid numpy image (grayscale for simplicity first)
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Fill with some data
    img[20:80, 20:80] = 255
    
    # Try different ways to call gen_image1
    print("Attempting ha.gen_image1...")
    
    # 1. Try passing array directly (some bindings magic might exist)
    try:
        himg = ha.gen_image1('byte', w, h, img)
        print("Success with direct numpy array!")
    except Exception as e:
        print(f"Direct array failed: {e}")

        # 2. Try passing data pointer
        try:
            ptr = img.ctypes.data
            himg = ha.gen_image1('byte', w, h, ptr)
            print("Success with ctypes pointer!")
        except Exception as e:
            print(f"Pointer failed: {e}")
            
except Exception as e:
    print(f"Fatal test error: {e}")
