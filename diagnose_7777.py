"""
诊断工具：检查7777产品的origin和路径问题
"""

import json
import os

product_dir = r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\produces\7777"

# 1. 读取配置
with open(os.path.join(product_dir, "config.json"), 'r') as f:
    config = json.load(f)

print("=" * 60)
print("7777 产品配置诊断")
print("=" * 60)

# 2. 模型Origin
print(f"\n📍 模型Origin（在reference.png中的位置）:")
print(f"   model_row: {config.get('model_row', 'N/A')}")
print(f"   model_col: {config.get('model_col', 'N/A')}")

# 3. ROI信息
roi = config.get('roi_in_reference', {})
print(f"\n📦 训练ROI（在reference.png中）:")
print(f"   左上角: ({roi.get('x', 'N/A')}, {roi.get('y', 'N/A')})")
print(f"   尺寸: {roi.get('width', 'N/A')} x {roi.get('height', 'N/A')}")
print(f"   中心点: ({roi.get('x', 0) + roi.get('width', 0)/2:.0f}, {roi.get('y', 0) + roi.get('height', 0)/2:.0f})")

# 4. Origin应该在ROI中心附近
if roi:
    expected_row =roi.get('y', 0) + roi.get('height', 0) / 2
    expected_col = roi.get('x', 0) + roi.get('width', 0) / 2
    actual_row = config.get('model_row', 0)
    actual_col = config.get('model_col', 0)
    
    diff_row = abs(actual_row - expected_row)
    diff_col = abs(actual_col - expected_col)
    
    print(f"\n✅ Origin位置检查:")
    print(f"   预期Origin（ROI中心）: ({expected_row:.0f}, {expected_col:.0f})")
    print(f"   实际Origin: ({actual_row:.0f}, {actual_col:.0f})")
    print(f"   差异: Δrow={diff_row:.0f}, Δcol={diff_col:.0f}")
    
    if diff_row < 50 and diff_col < 50:
        print(f"   ✅ Origin位置合理")
    else:
        print(f"   ⚠️ Origin偏离ROI中心较远！")

# 5. 路径信息
if os.path.exists(os.path.join(product_dir, "path.npy")):
    import numpy as np
    path = np.load(os.path.join(product_dir, "path.npy"))
    print(f"\n📍 点胶路径:")
    print(f"   点数量: {len(path)}")
    if len(path) > 0:
        print(f"   路径范围: X[{path[:, 0].min():.0f}, {path[:, 0].max():.0f}], Y[{path[:, 1].min():.0f}, {path[:, 1].max():.0f}]")
else:
    print(f"\n📍 点胶路径: 未找到")

print("\n" + "=" * 60)
