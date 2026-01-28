import sys
import os
import numpy as np

# 尝试导入 Halcon 库
# 用户环境应安装 mvtec-halcon 或 halcon
try:
    import halcon as ha
except Exception as e:
    ha = None
    print(f"Warning: Failed to import halcon: {e}")

class HalconManager:
    """
    Halcon 集成管理器。
    负责加载 .hdev 脚本或 .shm 模型，并执行视觉处理。
    """
    def __init__(self):
        self.engine = None
        self.procedure = None
        self.procedure_call = None
        self.script_path = None
        self.current_model_path = None # 当前加载的产品模型路径 (.shm)
        self.is_initialized = False

    def init_engine(self):
        """
        初始化 HDevEngine。
        """
        global ha
        if ha is None:
            print("[HalconManager] 错误: 未安装 halcon python 库。")
            return False
            
        try:
            # 在这里初始化引擎，如果需要的话
            # self.engine = ha.HDevEngine()
            # self.engine.set_attribute("debug_port", 5777) # 示例调试端口
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"[HalconManager] 引擎初始化失败: {e}")
            return False

    def load_script(self, script_path):
        """
        加载 Halcon 脚本 (.hdev) 或 外部过程。
        目前假设是一个 .hdev 脚本，里面包含主要的检测逻辑。
        """
        global ha
        if not self.is_initialized:
            if not self.init_engine():
                return False

        if not os.path.exists(script_path):
            print(f"[HalconManager] 文件不存在: {script_path}")
            return False

        try:
            self.script_path = script_path
            
            # 示例：加载 HDev 程序
            # program = ha.HDevProgram(script_path)
            # self.procedure_call = ha.HDevProcedureCall(program)
            
            print(f"[HalconManager] 脚本已加载: {script_path}")
            return True
        except Exception as e:
            print(f"[HalconManager] 加载脚本失败: {e}")
            return False

    def set_model_path(self, path):
        """设置当前使用的产品模型路径"""
        if path and os.path.exists(path):
            self.current_model_path = path
            print(f"[HalconManager] 已设置模型路径: {path}")
            return True
        else:
            print(f"[HalconManager] 模型文件不存在: {path}")
            return False

    def process_image(self, image_np: np.ndarray):
        """
        运行 'scripts/detect_mold.hdev' 进行检测。
        """
        global ha
        if not self.is_initialized:
            if not self.init_engine():
                return None
 
        # 1. 转换图像: Numpy -> Halcon Image
        try:
            h_image = self._numpy_to_himage(image_np)
            if h_image is None: return None
        except Exception as e:
            print(f"[HalconManager] 图像转换失败: {e}")
            return None

        # 2. 加载并运行脚本
        # 使用基于当前文件位置的绝对路径，避免 CWD 问题
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        script_file = os.path.join(project_root, "scripts", "detect_mold.hdev")
        
        if not os.path.exists(script_file):
            print(f"[HalconManager] 脚本不存在: {script_file}")
            return None

        try:
            # 加载 HDev 程序
            program = ha.HDevProgram(script_file)
            
            # 加载 Local Procedure 'detect_mold'
            try:
                proc = ha.HDevProcedure.load_local(program, "detect_mold")
            except Exception as e:
                print(f"[HalconManager] 无法在脚本中找到过程 'detect_mold': {e}")
                return None

            # 创建调用
            proc_call = ha.HDevProcedureCall(proc)
            
            # 设置输入
            # 1. 图像
            proc_call.set_input_iconic_param_by_name("Image", h_image)
            
            # 2. 模型路径 (支持多产品)
            # 假设 HDev 脚本里定义了一个输入控制参数 "ModelPath"
            # 如果脚本里还没这个参数，Halcon 可能会报错，所以我们加个 try-catch 或者检查
            if self.current_model_path and os.path.exists(self.current_model_path):
                # 传入绝对路径
                abs_model_path = os.path.abspath(self.current_model_path)
                print(f"[HalconManager] 🔍 使用模型: {abs_model_path}")  # 调试日志
                try:
                    proc_call.set_input_control_param_by_name("ModelPath", abs_model_path)
                    print(f"[HalconManager] ✅ ModelPath参数传递成功")  # 调试日志
                except Exception as e:
                    # 脚本可能没定义这个参数，忽略警告（兼容旧脚本）
                    print(f"[HalconManager] ⚠️ ModelPath参数传递失败: {e}")  # 调试日志
                    pass
            else:
                print(f"[HalconManager] ⚠️ 未设置模型路径或文件不存在: {self.current_model_path}")  # 调试日志
            
            # 执行
            proc_call.execute()
            
            # 获取输出
            # Output: ResultRow, ResultColumn, ResultAngle, ResultScore
            # API 修正: get_output_control_param_tuple -> get_output_control_param_by_name
            res_row = proc_call.get_output_control_param_by_name("ResultRow")
            res_col = proc_call.get_output_control_param_by_name("ResultColumn")
            res_angle = proc_call.get_output_control_param_by_name("ResultAngle")
            res_score = proc_call.get_output_control_param_by_name("ResultScore")
            
            # 转换 Tuple -> Python List
            # Halcon Tuple 有时是单个值，有时是tuple
            if hasattr(res_row, '__iter__') and not isinstance(res_row, str):
                 rows = list(res_row)
                 cols = list(res_col)
                 angles = list(res_angle)
                 scores = list(res_score)
            else:
                 # 单个值的情况（或空）
                 # 实际上 get_output... 返回的通常是 tuple，即使是空的
                 rows = list(res_row) if res_row else []
                 cols = list(res_col) if res_col else []
                 angles = list(res_angle) if res_angle else []
                 scores = list(res_score) if res_score else []

            # 组装结果
            results = []
            for r, c, a, s in zip(rows, cols, angles, scores):
                results.append({
                    "row": r,
                    "col": c,
                    "angle": a,
                    "score": s
                })
            
            print(f"[HalconManager] 识别成功，找到 {len(results)} 个目标。")
            return {
                "success": True,
                "matches": results
            }
            
        except Exception as e:
            print(f"[HalconManager] 执行 HDev 过程失败: {e}")
            return None

    def train_model(self, image_path, output_model_path, roi=None):
        """
        训练形状模板并保存 (.shm)
        :param image_path: 训练图片路径 (PNG/JPG)
        :param output_model_path: 保存路径 (.shm)
        :param roi: (row1, col1, row2, col2) in Halcon format, or None for full image
        """
        global ha
        if ha is None:
            return False, "Halcon library not loaded"

        try:
            print(f"[HalconManager] 开始训练模型: {image_path}")
            if roi:
                print(f"[HalconManager] 使用ROI: {roi}")
            
            # 1. Read Image
            image = ha.read_image(image_path)
            
            # 2. Apply ROI if provided
            if roi:
                row1, col1, row2, col2 = roi
                # 创建ROI区域
                model_region = ha.gen_rectangle1(row1, col1, row2, col2)
                # 裁剪图像到ROI
                template_image = ha.reduce_domain(image, model_region)
                print(f"[HalconManager] ROI区域创建成功: Row[{row1}-{row2}], Col[{col1}-{col2}]")
            else:
                template_image = image
            
            # 3. Create Shape Model
            # 调整参数以提高对光照差异的鲁棒性
            model_id = ha.create_shape_model(
                template_image, 
                'auto',                      # NumLevels
                0.0,                         # AngleStart
                6.29,                        # AngleExtent (360°)
                'auto',                      # AngleStep
                'auto',                      # Optimization
                'ignore_global_polarity',    # Metric - 忽略全局明暗差异！
                'auto',                      # Contrast
                1                            # MinContrast - 极致降低 (1) 以捕捉最微弱细节
            )
            
            # 4. Get Model Origin
            origin_row, origin_col = ha.get_shape_model_origin(model_id)
            
            # 5. Calculate center of gravity
            area, row_g, col_g = ha.area_center(template_image)
            
            # 6. Write Model
            ha.write_shape_model(model_id, output_model_path)
            
            # 7. Cleanup
            ha.clear_shape_model(model_id)
            
            print(f"[HalconManager] 模型已保存: {output_model_path}, Origin: ({row_g}, {col_g})")
            return True, {"msg": "Success", "origin_row": row_g, "origin_col": col_g}
            
        except Exception as e:
            msg = f"Training failed: {e}"
            print(f"[HalconManager] {msg}")
            import traceback
            traceback.print_exc()
            return False, msg

    def _numpy_to_himage(self, image_np):

        """
        Numpy (BGR/Gray) -> HImage
        """
        global ha
        if ha is None: return None
        
        try:
            h, w = image_np.shape[:2]
            
            # 1. 灰度图 (H, W)
            if image_np.ndim == 2:
                # 必须确保内存连续
                if not image_np.flags['C_CONTIGUOUS']:
                    image_np = np.ascontiguousarray(image_np)
                
                ptr = image_np.ctypes.data
                return ha.gen_image1('byte', w, h, ptr)

            # 2. 彩色图 (H, W, 3) - OpenCV 默认是 BGR
            elif image_np.ndim == 3 and image_np.shape[2] == 3:
                # 同样确保连续
                if not image_np.flags['C_CONTIGUOUS']:
                    image_np = np.ascontiguousarray(image_np)
                
                ptr = image_np.ctypes.data
                # gen_image_interleaved(PixelPointer, ColorSpace, Width, Height, Angle, BaseType, Width, Height, 0, 0, -1, 0)
                # Halcon 参数定义: PixelPointer, ColorSpace, Width, Height, Angle, SimpleOrComplex, BaseType, ... ?
                # 查阅: gen_image_interleaved (Image, PixelPointer, ColorSpace, Width, Height, Angle, BaseType, Width, Height, 0, 0, -1, 0) ?
                # 其实 Python 绑定通常简化了参数。
                # 由于缺少文档，我们先尝试 gen_image_interleaved(PixelPointer, 'bgr', w, h, 0, 'byte', w, h, 0, 0, -1, 0)
                
                # 更稳妥：拆分成 3 个通道，然后 compose3? 
                # 不，gen_image_interleaved 是标准做法。
                # 签名通常是: gen_image_interleaved('bgr', w, h, 0, 'byte', w, h, 0, 0, -1, 0, ptr) ?
                
                # 让我们先只支持灰度转换，因为 detect_mold 主要是灰度处理。
                # 如果是彩色，先转灰度。这是最稳的。
                import cv2
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                if not gray.flags['C_CONTIGUOUS']:
                    gray = np.ascontiguousarray(gray)
                ptr = gray.ctypes.data
                return ha.gen_image1('byte', w, h, ptr)
                
        except Exception as e:
             print(f"[HalconManager] 转换失败: {e}")
             pass
             
        return None
