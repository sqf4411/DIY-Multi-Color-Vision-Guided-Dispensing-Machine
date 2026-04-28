import cv2
import numpy as np
import json
import os

base = r'h:/CodeForGithub/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/produces/n_TB2'
ref_path = os.path.join(base, 'reference.png')
tmpl_path = os.path.join(base, 'roi_template.png')
config_path = os.path.join(base, 'config.json')

ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
roi_tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)

print(f'reference.png shape: {ref.shape}')
print(f'roi_template.png shape: {roi_tmpl.shape}')

# Downscale for speed (template matching on full 3072x2048 is slow)
scale = 0.25
ref_s = cv2.resize(ref, None, fx=scale, fy=scale)
tmpl_s = cv2.resize(roi_tmpl, None, fx=scale, fy=scale)

res = cv2.matchTemplate(ref_s, tmpl_s, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)

# Restore to full scale (divide scale coords back)
x = int(round(max_loc[0] / scale))
y = int(round(max_loc[1] / scale))
w = roi_tmpl.shape[1]
h = roi_tmpl.shape[0]

print(f'\nMatch score: {max_val:.4f}')
print(f'Found ROI position: x={x}, y={y}, width={w}, height={h}')

with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

model_col = cfg.get('model_col', 0)
model_row = cfg.get('model_row', 0)
print(f'model_col={model_col}, model_row={model_row}')
print(f'=> origin_in_roi: col={model_col - x:.1f}, row={model_row - y:.1f}')

if max_val < 0.7:
    print('\nWARNING: Low match score. The ROI template may not accurately match reference.png.')
    print('Consider re-training n_TB2.')
else:
    print(f'\nMatch is good (score={max_val:.4f}). Updating config.json...')
    cfg['roi_in_reference'] = {
        'x': x,
        'y': y,
        'width': w,
        'height': h
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f'config.json updated with roi_in_reference: x={x}, y={y}, w={w}, h={h}')
    print('Done! Please restart the application or reload the product n_TB2.')
