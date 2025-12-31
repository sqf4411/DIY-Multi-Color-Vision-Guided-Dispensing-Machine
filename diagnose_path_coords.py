"""
诊断路径坐标系统
检查保存的路径坐标是绝对坐标还是相对坐标
"""

import numpy as np
import json
import os

product_dir = r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\produces\7777"

print("=" * 60)
print("7777路径坐标诊断")
print("=" * 60)

# 1. 加载配置
with open(os.path.join(product_dir, "config.json"), 'r') as f:
    config = json.load(f)

model_row = config.get('model_row', 0)
model_col = config.get('model_col', 0)
roi = config.get('roi_in_reference', {})

print(f"\n📍 模型Origin: row={model_row:.0f}, col={model_col:.0f}")
print(f"📦 训练ROI: x={roi.get('x', 0)}, y={roi.get('y', 0)}, {roi.get('width', 0)}x{roi.get('height', 0)}")

# 2. 加载路径
multicolor_path = os.path.join(product_dir, "multicolor_path.json")
if os.path.exists(multicolor_path):
    with open(multicolor_path, 'r') as f:
        path_data = json.load(f)
    
    segments = path_data.get('segments', [])
    print(f"\n📍 路径段数量: {len(segments)}")
    
    if segments:
        first_seg = segments[0]
        points = first_seg.get('points', [])
        if points:
            print(f"\n第一个路径段示例:")
            print(f"  类型: {first_seg.get('type', 'N/A')}")
            print(f"  颜色: {first_seg.get('color', 'N/A')}")
            print(f"  点数: {len(points)}")
            print(f"  第一个点: {points[0]}")
            
            # 分析坐标范围
            all_points = []
            for seg in segments:
                all_points.extend(seg.get('points', []))
            
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                
                print(f"\n📊 所有路径点统计:")
                print(f"  X范围: [{min(xs):.0f}, {max(xs):.0f}]")
                print(f"  Y范围: [{min(ys):.0f}, {max(ys):.0f}]")
                
                # 判断坐标系
                print(f"\n🔍 坐标系判断:")
                if min(xs) > roi.get('x', 0) and max(xs) < roi.get('x', 0) + roi.get('width', 0):
                    if min(ys) > roi.get('y', 0) and max(ys) < roi.get('y', 0) + roi.get('height', 0):
                        print(f"  ✅ 路径坐标在ROI范围内 → 绝对坐标（在reference.png上）")
                    else:
                        print(f"  ⚠️ X在ROI内但Y不在 → 可能是混合坐标")
                elif abs(min(xs)) < 500 and abs(max(xs)) < 500:
                    print(f"  ✅ 路径坐标接近0 → 相对坐标（相对于origin）")
                else:
                    print(f"  ❓ 坐标系不确定")
                    print(f"     ROI范围: X[{roi.get('x', 0)}, {roi.get('x', 0) + roi.get('width', 0)}]")
                    print(f"              Y[{roi.get('y', 0)}, {roi.get('y', 0) + roi.get('height', 0)}]")
else:
    print("\n❌ 未找到multicolor_path.json")

print("\n" + "=" * 60)
