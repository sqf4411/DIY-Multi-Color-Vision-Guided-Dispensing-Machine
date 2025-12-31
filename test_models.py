"""
测试使用不同模型识别7777产品
"""

import os
import sys
import cv2

# 添加Halcon路径
halcon_path = r"C:\Program Files\MVTec\HALCON-20.11-Steady\bin\x64-win64"
if halcon_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = halcon_path + os.pathsep + os.environ.get('PATH', '')

import halcon as ha

def test_recognition(image_path, model_path, model_name):
    """测试识别"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}")
    
    # 读取图像
    image = ha.read_image(image_path)
    
    # 加载模型
    model_id = ha.read_shape_model(model_path)
    
    # 识别
    try:
        row, column, angle, score = ha.find_scaled_shape_model(
            image, model_id,
            0.0, 6.29,  # angle range
            0.8, 1.2,   # scale min, max
            0.5,        # min_score
            1,          # num_matches
            0.5,        # max_overlap
            'least_squares',  # sub_pixel
            0,          # num_levels
            0.9         # greediness
        )
        
        # 转换结果
        if hasattr(row, '__iter__'):
            rows = list(row)
            cols = list(column)
            angles = list(angle)
            scores = list(score)
        else:
            rows = [row] if row else []
            cols = [column] if column else []
            angles = [angle] if angle else []
            scores = [score] if score else []
        
        print(f"\n✅ 识别结果: 找到 {len(rows)} 个目标")
        for i, (r, c, a, s) in enumerate(zip(rows, cols, angles, scores)):
            print(f"  目标{i+1}: row={r:.2f}, col={c:.2f}, angle={a:.4f}, score={s:.4f}")
        
    except Exception as e:
        print(f"❌ 识别失败: {e}")
    
    finally:
        ha.clear_shape_model(model_id)

if __name__ == "__main__":
    # 测试图像（从摄像头截图或使用reference.png）
    test_image = r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\produces\7777\reference.png"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        print("请提供一张包含4个7777产品的图像")
        sys.exit(1)
    
    # 测试两个模型
    test_recognition(test_image, 
                    r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\produces\7777\7777.shm",
                    "7777.shm (产品专用)")
    
    test_recognition(test_image,
                    r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\config\mold_model.shm",
                    "mold_model.shm (原始模板)")
