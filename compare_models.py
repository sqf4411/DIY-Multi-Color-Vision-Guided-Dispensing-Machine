"""
对比两个Halcon模型文件的参数
"""

import os
import sys

# 添加Halcon路径
halcon_path = r"C:\Program Files\MVTec\HALCON-20.11-Steady\bin\x64-win64"
if halcon_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = halcon_path + os.pathsep + os.environ.get('PATH', '')

try:
    import halcon as ha
except Exception as e:
    print(f"无法加载Halcon: {e}")
    sys.exit(1)

def compare_models(model1_path, model2_path):
    """对比两个Halcon shape model"""
    
    print("=" * 60)
    print("Halcon模型对比工具")
    print("=" * 60)
    
    # 检查文件
    for path in [model1_path, model2_path]:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return
    
    # 文件基本信息
    size1 = os.path.getsize(model1_path)
    size2 = os.path.getsize(model2_path)
    
    print(f"\n📁 文件信息:")
    print(f"  模型1: {model1_path}")
    print(f"    大小: {size1:,} bytes")
    print(f"  模型2: {model2_path}")
    print(f"    大小: {size2:,} bytes")
    print(f"  大小差异: {abs(size1 - size2):,} bytes ({(size1/size2)*100:.1f}%)")
    
    # 加载模型
    try:
        print(f"\n📥 加载模型...")
        model1_id = ha.read_shape_model(model1_path)
        model2_id = ha.read_shape_model(model2_path)
        
        # 获取模型参数
        print(f"\n🔍 模型1参数:")
        params1 = get_model_params(model1_id)
        print_params(params1)
        
        print(f"\n🔍 模型2参数:")
        params2 = get_model_params(model2_id)
        print_params(params2)
        
        # 对比
        print(f"\n📊 参数对比:")
        compare_params(params1, params2)
        
        # 清理
        ha.clear_shape_model(model1_id)
        ha.clear_shape_model(model2_id)
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()

def get_model_params(model_id):
    """获取shape model的参数"""
    params = {}
    
    try:
        # 获取模型原点
        row, col = ha.get_shape_model_origin(model_id)
        params['origin'] = (row, col)
        
        # 获取模型参数
        # get_shape_model_params返回: NumLevels, AngleStart, AngleExtent, AngleStep, Metric, MinContrast
        num_levels = ha.get_shape_model_param(model_id, 'num_levels')
        angle_start = ha.get_shape_model_param(model_id, 'angle_start')
        angle_extent = ha.get_shape_model_param(model_id, 'angle_extent')
        angle_step = ha.get_shape_model_param(model_id, 'angle_step')
        metric = ha.get_shape_model_param(model_id, 'metric')
        min_contrast = ha.get_shape_model_param(model_id, 'min_contrast')
        
        params['num_levels'] = num_levels
        params['angle_start'] = angle_start
        params['angle_extent'] = angle_extent
        params['angle_step'] = angle_step
        params['metric'] = metric
        params['min_contrast'] = min_contrast
        
    except Exception as e:
        print(f"  警告: 无法获取某些参数: {e}")
    
    return params

def print_params(params):
    """打印参数"""
    if 'origin' in params:
        print(f"  Origin: ({params['origin'][0]:.2f}, {params['origin'][1]:.2f})")
    if 'num_levels' in params:
        print(f"  Num Levels: {params['num_levels']}")
    if 'angle_start' in params:
        print(f"  Angle Start: {params['angle_start']:.4f} rad")
    if 'angle_extent' in params:
        print(f"  Angle Extent: {params['angle_extent']:.4f} rad")
    if 'angle_step' in params:
        print(f"  Angle Step: {params['angle_step']:.4f} rad")
    if 'metric' in params:
        print(f"  Metric: {params['metric']}")
    if 'min_contrast' in params:
        print(f"  Min Contrast: {params['min_contrast']}")

def compare_params(params1, params2):
    """对比两组参数"""
    all_keys = set(params1.keys()) | set(params2.keys())
    
    for key in sorted(all_keys):
        v1 = params1.get(key, 'N/A')
        v2 = params2.get(key, 'N/A')
        
        if v1 == v2:
            status = "✅ 相同"
        else:
            status = "❌ 不同"
        
        print(f"  {key}: {status}")
        if v1 != v2:
            print(f"    模型1: {v1}")
            print(f"    模型2: {v2}")

if __name__ == "__main__":
    model1 = r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\produces\7777\7777.shm"
    model2 = r"h:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\config\mold_model.shm"
    
    compare_models(model1, model2)
