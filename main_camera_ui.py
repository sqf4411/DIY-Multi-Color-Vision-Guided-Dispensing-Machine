import os
import sys
import shutil
import ctypes
from datetime import datetime
import json
import time
import cv2
import numpy as np
from core.machine_client import MachineClient

def add_mvs_dll_path():
    dll_dir = os.getenv("MVS_DLL_DIR")
    if dll_dir and os.path.isdir(dll_dir):
        if sys.platform == "win32":
            os.add_dll_directory(dll_dir)
        else:
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[INFO] MVS DLL directory added: {dll_dir}")
    else:
        print("[WARN] 环境变量 MVS_DLL_DIR 未设置或路径不存在，请检查。")

add_mvs_dll_path()

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QMessageBox, QLineEdit, QScrollArea,QDialog, QDialogButtonBox
)




from PyQt5.QtGui import QImage, QPixmap

# 你的检测函数
from core.mold_detector import detect_mold_and_cells, reload_golden_templates
from core.dry_run_dialog import DryRunDialog, DryRunControlDialog

# ---- 按你的实际路径修改 ----
sys.path.append(r"H:\MVS\Development\Samples\Python\MvImport")
from MvImport.MvCameraControl_class import *

# ================== 线程：相机取流 ==================
class CameraThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        # 获取 PayloadSize
        st_param = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            print("Get PayloadSize failed")
            return

        data_size = st_param.nCurValue
        data_buff = (ctypes.c_ubyte * data_size)()
        frame_info = MV_FRAME_OUT_INFO_EX()

        while self.running:
            ret = self.cam.MV_CC_GetOneFrameTimeout(
                data_buff, data_size, frame_info, 1000
            )
            if ret == 0:
                w, h = frame_info.nWidth, frame_info.nHeight
                img = np.frombuffer(data_buff, dtype=np.uint8)[: w * h]
                img = img.reshape((h, w))
                self.frame_signal.emit(img)

# ================== 线程：执行耗时的归位操作 ==================
class HomeWorker(QThread):
    """
    专门用于后台执行归位操作，防止卡死 UI 主线程
    """
    # 任务完成后，发送信号告诉主界面更新坐标 (x, y, z)
    finished_signal = pyqtSignal(float, float, float)

    def __init__(self, machine_client, axis):
        super().__init__()
        self.machine = machine_client
        self.axis = axis

    def run(self):
        # 这里是后台线程，怎么卡都不会影响界面
        if self.axis == 'x':
            self.machine.home_x()
        elif self.axis == 'y':
            self.machine.home_y()
        elif self.axis == 'z':
            self.machine.home_z()
        elif self.axis == 'all':
            self.machine.home_all()
        
        # 无论成功失败，都发信号
        # 实际位置需要再次查询，这里简单发个 0,0,0 或者让主界面自己去 poll
        # 为了方便，我们在 machine_client 里加了 _x, _y, _z 缓存，但最好还是读一下
        self.finished_signal.emit(self.machine._x, self.machine._y, self.machine._z)

# ================== 线程：异步移动 (解决 UI 卡顿) ==================
class MovementWorker(QThread):
    """
    专门用于执行相对移动 (Jog)，防止 HTTP 请求阻塞主线程
    """
    finished_signal = pyqtSignal(float, float, float)

    def __init__(self, machine, dx, dy, dz, f_xy, f_z):
        super().__init__()
        self.machine = machine
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.f_xy = f_xy
        self.f_z = f_z

    def run(self):
        # 执行移动
        self.machine.move_rel(self.dx, self.dy, self.dz, self.f_xy, self.f_z)
        # 移动完立即查询一次位置
        x, y, z = self.machine.get_position()
        self.finished_signal.emit(x, y, z)

# ================== 线程：后台轮询状态 (解决 UI 卡顿) ==================
class PollingWorker(QThread):
    """
    后台轮询机器位置和探针状态
    """
    pos_signal = pyqtSignal(float, float, float)
    probe_signal = pyqtSignal(str)

    def __init__(self, machine):
        super().__init__()
        self.machine = machine
        self.running = True

    def run(self):
        while self.running:
            if getattr(self.machine, "connected", False):
                try:
                    # 1. 查询坐标
                    x, y, z = self.machine.get_position()
                    self.pos_signal.emit(x, y, z)
                    
                    # 2. 查询探针
                    status = self.machine.query_probe_status()
                    self.probe_signal.emit(status)
                except Exception:
                    pass
            
            # 间隔 0.5 秒
            time.sleep(0.5)

    def stop(self):
        self.running = False
        self.wait()

# ================== 线程：自动对刀 ==================
class AutoToolSetWorker(QThread):
    finished_signal = pyqtSignal(bool, float, str) # success, new_z_base, msg

    def __init__(self, machine_client, safe_z, target_x, target_y):
        super().__init__()
        self.machine = machine_client
        self.safe_z = safe_z
        self.target_x = target_x
        self.target_y = target_y

    def run(self):
        try:
            # 1. 抬升到安全高度
            self.machine.move_z_absolute(self.safe_z, 600)
            time.sleep(0.5)

            # 2. 移动到对刀仪上方
            self.machine.move_xy_absolute(self.target_x, self.target_y, 3000)
            self.machine.wait_until_idle()
            time.sleep(0.5)

            # 3. 快速下降到 Z=125 (接近对刀仪，节省时间)
            # 用户提示对刀仪在 116 左右，我们从 125 开始探，留 9mm 余量防止针头变长
            self.machine.logger(f"🚀 快速下降到 Z=125.0...")
            self.machine.move_z_absolute(125.0, 600)
            self.machine.wait_until_idle()
            time.sleep(0.5)

            # 4. 第一阶段：粗探 (Z=125 -> 115, step=0.5)
            # 步长 0.5mm，下限 115.0mm (用户对刀仪在 116，行程 1mm，所以 115 是极限)
            self.machine.logger(f"🔍 第一阶段粗探 (Z=125->115, step=0.5)...")
            trigger_z = None
            current_z = 125.0
            found = False
            
            while current_z > 115.0:
                status = self.machine.query_probe_status()
                if status == 'triggered':
                    found = True
                    trigger_z = current_z
                    self.machine.logger(f"✅ 粗探触发于 Z={current_z:.2f}")
                    break
                
                current_z -= 0.5  # 步长 0.5mm
                self.machine.move_z_absolute(current_z, 600)
                self.machine.wait_until_idle()

            if not found:
                self.machine.logger(f"❌ 粗探失败：已达 Z=115.0mm 仍未触发！")
                # 失败后立即抬起 Z，防止误操作
                self.machine.move_z_absolute(self.safe_z, 600)
                self.finished_signal.emit(False, 0.0, "粗探未检测到信号 (已归位)")
                return

            # 5. 准备精探：抬起 3mm (优化：从 5mm 改为 3mm)
            self.machine.move_rel(0, 0, 3.0, feedrate_z=600)
            self.machine.wait_until_idle()
            time.sleep(0.5)

            # 6. 第二阶段：双次精探 (step=0.1)
            # 恢复双次探测取平均，保证精度
            results = []
            for i in range(2):
                self.machine.logger(f"🔍 精探第 {i+1} 次 (step=0.1)...")
                
                # 既然是手动精探，我们需要从当前位置慢慢往下挪
                # 当前位置应该是 粗探点 + 3mm (第一次) 或 上次精探点 + 2mm (第二次)
                start_z = self.machine.get_position()[2]
                limit_z = start_z - 10.0
                curr_z = start_z
                hit = False
                
                while curr_z > limit_z:
                    status = self.machine.query_probe_status()
                    if status == 'triggered':
                        hit = True
                        results.append(curr_z)
                        self.machine.logger(f"✅ 第 {i+1} 次结果: {curr_z:.3f}")
                        break
                    
                    curr_z -= 0.1
                    self.machine.move_z_absolute(curr_z, 300) # 慢速 F300
                    self.machine.wait_until_idle()
                
                if not hit:
                    self.finished_signal.emit(False, 0.0, f"精探第 {i+1} 次失败")
                    return

                # 抬起 2mm 准备下一次
                if i < 1:
                    self.machine.move_rel(0, 0, 2.0, feedrate_z=600)
                    self.machine.wait_until_idle()
                    time.sleep(0.2)

            # 7. 计算平均值
            avg_z = sum(results) / len(results)
            self.machine.logger(f"📊 双次精探平均值: {avg_z:.3f}")

            # 8. 抬起并完成
            self.machine.move_z_absolute(self.safe_z, 600)
            self.finished_signal.emit(True, avg_z, "对刀成功")

        except Exception as e:
            self.finished_signal.emit(False, 0.0, f"对刀过程出错: {e}")

# ================== 线程：后台发送 G-code ==================
# ================== 线程：逐行发送 G-code (支持暂停) ==================
class GCodeLineWorker(QThread):
    """
    逐行发送 G-code，支持暂停/继续/停止
    """
    progress_signal = pyqtSignal(int, int)  # 当前行, 总行数
    finished_signal = pyqtSignal(bool, str) # 成功/失败, 消息
    log_signal = pyqtSignal(str)            # 日志消息

    def __init__(self, machine_client, script_lines):
        super().__init__()
        self.machine = machine_client
        self.lines = script_lines
        self._is_paused = False
        self._is_stopped = False

    def run(self):
        total = len(self.lines)
        self.log_signal.emit(f"开始执行，共 {total} 行指令")
        
        for i, line in enumerate(self.lines):
            # 1. 检查停止
            if self._is_stopped:
                self.finished_signal.emit(False, "用户已停止")
                return

            # 2. 检查暂停
            while self._is_paused:
                if self._is_stopped:
                    self.finished_signal.emit(False, "用户已停止")
                    return
                self.msleep(100) # 等待 100ms

            # 3. 发送指令
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # 发送并等待 ok
            # 对于移动指令，Klipper 通常会立即返回 ok (放入缓冲区)，除非缓冲区满了
            # 对于 G4 (Dwell) 或 M400 (Finish Moves)，会阻塞直到完成
            # 我们设置一个较长的超时，防止长指令超时
            ok = self.machine._post_gcode(line, timeout=300)
            
            if not ok:
                self.finished_signal.emit(False, f"指令发送失败: {line}")
                return

            # 4. 发送进度
            self.progress_signal.emit(i + 1, total)
        
        self.finished_signal.emit(True, "执行完成")

    def pause(self):
        self._is_paused = True
        self.log_signal.emit("已暂停发送下一行指令")

    def resume(self):
        self._is_paused = False
        self.log_signal.emit("恢复发送指令")

    def stop(self):
        self._is_stopped = True
        self.log_signal.emit("正在停止...")
######## 滚轮
class ImageLabel(QLabel):
    """支持滚轮缩放 + 左键拖拽的 QLabel"""
    wheel_zoom = pyqtSignal(int)          # 滚轮：传 deltaY
    pan_delta = pyqtSignal(int, int)      # 拖拽：传 dx, dy
    clicked_pos = pyqtSignal(int, int)  # <--- ✅ 新增：点击位置信号

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._last_pos = None

    def wheelEvent(self, event):
        self.wheel_zoom.emit(event.angleDelta().y())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 优先处理点击信号 (传给主窗口判断是不是在选角点)
            self.clicked_pos.emit(event.x(), event.y())

            # 保持原有的拖拽逻辑
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            # 注意：向右拖动，内容向右移动 => 滚动条减少
            self.pan_delta.emit(delta.x(), delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._last_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# ====== ✅ 必须添加：定义设置文件名常量 ======
import json # 确保 json 库已经导入
import os   # 确保 os 库已经导入
# 这个文件将用于保存速度、加速度等应用参数
SETTINGS_FILE = "config/app_settings.json"
# ==========================================
# ================== 主窗口 ==================
class MainWindow(QWidget):
    # 定义信号，用于跨线程更新 UI
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIY 视觉点胶机控制台 (No30 还原版)")
        self.resize(1280, 800)

        # 连接日志信号
        self.log_signal.connect(self._log_impl)

        # 设置黑色主题样式
        self.setStyleSheet("""
        QWidget {
            background-color: #111827;
            color: #E5E7EB;
            font-family: "Segoe UI", "Microsoft YaHei";
            font-size: 14px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #374151;
            border-radius: 8px;
            margin-top: 10px;
            background-color: #1F2933;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px 0 6px;
        }

        /* 普通标签 */
        QLabel {
            color: #E5E7EB;
        }

        /* 文本框 / 数值框 / 下拉框 */
        QLineEdit,
        QDoubleSpinBox,
        QComboBox {
            background-color: #020617;
            border: 1px solid #4B5563;
            border-radius: 4px;
            padding: 2px 6px;
        }

        /* 日志框 */
        QTextEdit {
            background-color: #020617;
            border: 1px solid #4B5563;
            border-radius: 6px;
        }

        /* 按钮：蓝色主按钮风格 */
        QPushButton {
            min-height: 26px;
            padding: 4px 12px;
            border-radius: 4px;
            border: none;
            background-color: #2563EB;       /* 主色 */
            color: #F9FAFB;
        }
        QPushButton:hover {
            background-color: #1D4ED8;
        }
        QPushButton:pressed {
            background-color: #1E40AF;
        }
        QPushButton:disabled {
            background-color: #4B5563;
            color: #9CA3AF;
        }

        /* 滑块 */
        QSlider::groove:horizontal {
            border: 1px solid #4B5563;
            height: 4px;
            background: #374151;
            margin: 0 4px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #F97316;             /* 橙色小圆点 */
            border: 1px solid #F97316;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        """)

        self.cam = None
        self.config_path = "config/params.json"  # ✅ 新增：定义参数文件路径
        self.thread = None
        self.last_frame = None
        self.freeze_display = False  # 是否冻结画面
        self.live_detect = False     # 是否实时识别模具
        # 缩放相关
        self.zoom_scale = 1.0        # 在“自适应基准”上的倍数
        self.base_scale = None       # 适配窗口的基准比例（首帧或窗口变化时计算）

        # 新增：缩放比例
        self.zoom_scale = 1.0

        # ==== 相机标定 / 去畸变 ====
        self.K = None          # 内参矩阵
        self.D = None          # 畸变系数
        self.map1 = None       # remap 映射
        self.map2 = None
        self.calib_size = None # (w, h)
        self._calib_loaded = False
        
        self.cam2mach_matrix = None  # ✅ 新增：坐标变换矩阵

        # ====== 坐标标定相关 ======
        self.calib_step = 0  # 0=闲置, 10=微调视觉点, 1-4=记录机器点
        self.verify_mode = False  # ✅ 新增：默认为关闭状态
        self.verify_click_pt = None  # ✅ 新增：用于记录验证模式下点击的坐标

        # ====== ✅ 新增：针头标定状态 ======
        self.nozzle_calib_step = 0  # 0=闲置, 1=对准相机, 2=对准针头
        # ==================================

        # ====== ✅ 新增：用于在画面上绘制空跑路径 ======
        self.dry_run_paths_to_draw = []  # 存像素坐标路径
        # ===========================================

        self.calib_vision_pts = []  # 视觉坐标 (4,2)
        self.calib_machine_pts = []
        self.calib_sel_idx = 0  # 当前正在微调哪个角 (0-3)

        # 识别结果
        self.big_box = None
        self.cell_boxes = []
        self.cell_centers = []

        # 配置文件
        # self.config_path 已经在前面定义为 "params.json"
        self.saved_params = {}

        # 黄金模板
        self.slot_templates = None   # (4,4,2)
        self.latest_capture_path = None  # ✅ 新增：记录最近保存的图片路径

        # 参数范围
        self.exp_min = 100.0
        self.exp_max = 10000.0
        self.gain_min = 0.0
        self.gain_max = 10.0
        self.gamma_min = 0.5
        self.gamma_max = 2.0



        # ==== 运动 / 坐标相关 ====
        self.machine = MachineClient()
        self.machine.logger = self.log

        self.default_base_url = "http://192.168.60.26:7125"

        # 当前坐标（mm）——这里只作为初始显示，不去访问 Klipper
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        # 点动步长（mm）
        self.jog_step_mm = 1.0

        # ====== ✅ 新增：速度与加速度默认值 ======
        self.feedrate_xy = 3000.0  # 默认 XY 速度 (mm/min)
        self.feedrate_z = 500.0  # 默认 Z 速度 (mm/min)
        self.accel = 500.0  # 默认加速度 (mm/s^2)
        # ======================================

        # 针头相对「相机参考点」的 XY 偏移量（mm）
        self.nozzle_offset_x = 0.0
        self.nozzle_offset_y = 0.0

        # 相机像素 -> 机器 XY 的映射矩阵
        self.cam2mach_matrix = None

        # ===== 坐标标定相关 =====
        # 是否在画面中心显示十字线（标定时用）
        self.show_crosshair = False
        # 标定时记录的图像坐标 & 机器坐标
        self._calib_img_pts = []
        self._calib_mach_pts = []
        # 为了在画面上画出将要标定的腔体中心点和编号
        self.calib_points_for_ui = None

        # ====== ✅ 新增：普通模式下的十字中心开关 ======
        self.show_center_crosshair = False
        # ==========================================

        # ===== Z 高度参数 =====
        # 空跑/点胶时的安全高度与工作高度（mm）
        # 用户反馈：Z轴最大200+（上），0是下。
        self.safe_z = 20.0    # 抬刀 / 快移高度 (Legacy)
        self.safe_z_cavity = 20.0   # 腔内安全高度 (小抬刀)
        self.safe_z_product = 50.0  # 产品安全高度 (大抬刀)
        self.work_z = 1.0           # 点胶 / 下刀高度 (接近0)
        
        # 对刀基准 (上次对刀时的 Z 坐标)
        self.z_base = 0.0
        
        # 对刀仪坐标 (XY)
        self.tool_setter_x = None
        self.tool_setter_y = None
        
        # 对刀模式：False=正常, True=手动找对刀点
        self.manual_tool_set_mode = False

        # 实时坐标轮询 (使用后台线程，防止卡顿)
        self.poll_worker = PollingWorker(self.machine)
        self.poll_worker.pos_signal.connect(self.update_xyz_display)
        self.poll_worker.probe_signal.connect(self.update_probe_status)

        # ==========================================
        # 在所有 UI 元素初始化完成后，调用加载设置
        self.init_ui()
        self.load_app_settings()
        self.load_saved_params()
        self.load_golden_templates()
        self.init_calibration()

    # ================== 日志相关 ==================
    def _log_impl(self, msg):
        """实际更新 UI 的槽函数，运行在主线程"""
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_box.append(f"{timestamp}{msg}")
        # 自动滚动到底部
        cursor = self.log_box.textCursor()
        cursor.movePosition(cursor.End)
        self.log_box.setTextCursor(cursor)

    def log(self, msg, *args):
        """
        统一日志入口。
        无论谁调用，都发信号给主线程去更新 UI。
        支持多参数，类似 print
        """
        if args:
            msg = f"{msg} " + " ".join(str(a) for a in args)
        self.log_signal.emit(str(msg))

    # ================== UI ==================
    def init_ui(self):
        # ---- 左侧大画面 ----
        self.label = ImageLabel("无图像")
        self.label.setMinimumSize(640, 480)  # 加这一行保底
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: #000000;
            color: #9CA3AF;
            border-radius: 8px;
            border: 1px solid #374151;
        """)

        # 用 QScrollArea 包起来，放大后可以拖动查看
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(False)

        # 居中显示
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # 初始隐藏滚动条（缩放>1 时再打开）
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.scroll_area)

        # 信号：滚轮缩放 + 拖拽平移
        self.label.wheel_zoom.connect(self.on_wheel_zoom)
        self.label.pan_delta.connect(self.on_pan_delta)

        # ✅ 新增：连接点击信号
        self.label.clicked_pos.connect(self.on_label_clicked)

        # ====== ✅ 新增：显示中心按钮 (放在画面下方) ======
        self.btn_show_crosshair = QPushButton("显示中心")
        self.btn_show_crosshair.setCheckable(True) # 可以做成 toggle 样式
        self.btn_show_crosshair.clicked.connect(self.on_toggle_crosshair)
        
        left_layout.addWidget(self.btn_show_crosshair)
        # ==============================================

        # ---- 相机信息 ----
        self.lbl_cam_res = QLabel("-")
        self.lbl_cam_fps = QLabel("-")
        self.lbl_cam_fmt = QLabel("-")

        info_group = QGroupBox("相机信息")
        info_form = QFormLayout()
        info_form.addRow("分辨率：", self.lbl_cam_res)
        info_form.addRow("帧率：", self.lbl_cam_fps)
        info_form.addRow("像素格式：", self.lbl_cam_fmt)
        info_group.setLayout(info_form)

        # ---- Klipper 连接 ----
        conn_group = QGroupBox("Klipper 连接")
        conn_layout = QHBoxLayout()

        # 1. IP 输入框 (标签+框)
        lbl_ip = QLabel("IP:")
        self.edit_ip = QLineEdit("192.168.60.26")
        self.edit_ip.setFixedWidth(110)  # 给IP留够宽度
        self.edit_ip.setAlignment(Qt.AlignCenter)

        # 2. Port 输入框 (标签+框)
        lbl_port = QLabel("Port:")
        self.edit_port = QLineEdit("7125")
        self.edit_port.setFixedWidth(50)  # 端口号短一点
        self.edit_port.setAlignment(Qt.AlignCenter)

        # 3. 连接按钮 (初始灰色)
        self.btn_connect_klipper = QPushButton("连接")
        self.btn_connect_klipper.setCursor(Qt.PointingHandCursor)
        self.btn_connect_klipper.setMinimumHeight(28)
        self.btn_connect_klipper.setFixedWidth(80)  # ✅ 强制固定宽度
        self.btn_connect_klipper.setStyleSheet("""
                    QPushButton {
                        background-color: #6B7280;  /* 灰色 */
                        color: white;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #4B5563;
                    }
                """)

        # 4. 急停按钮 (红色 STOP)
        self.btn_estop = QPushButton("STOP")
        self.btn_estop.setCursor(Qt.PointingHandCursor)
        self.btn_estop.setMinimumHeight(28)
        self.btn_estop.setFixedWidth(80)  # ✅ 与连接按钮宽度一致
        self.btn_estop.setStyleSheet("""
                    QPushButton {
                        background-color: #DC2626;  /* 红色 */
                        color: white;
                        font-weight: bold;
                        border: 2px solid #991B1B;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background-color: #EF4444;
                    }
                    QPushButton:pressed {
                        background-color: #7F1D1D;
                    }
                """)

        # ====== 布局排列 (左侧输入框 -> 弹簧 -> 右侧按钮) ======
        conn_layout.addWidget(lbl_ip)
        conn_layout.addWidget(self.edit_ip)

        conn_layout.addSpacing(10)  # IP 和 Port 之间稍微隔开一点

        conn_layout.addWidget(lbl_port)
        conn_layout.addWidget(self.edit_port)

        conn_layout.addStretch()  # ✅ 关键修改：弹簧放在这里，把后面的东西顶到最右边

        # 5. 相机控制按钮 (移到这里)
        self.btn_start = QPushButton("启动相机")
        self.btn_stop = QPushButton("停止相机")
        self.btn_stop.setEnabled(False)

        # 样式微调，让它们看起来协调
        for btn in [self.btn_start, self.btn_stop]:
            btn.setMinimumHeight(28)
            btn.setFixedWidth(80)

        conn_layout.addWidget(self.btn_start)
        conn_layout.addSpacing(5)
        conn_layout.addWidget(self.btn_stop)
        conn_layout.addSpacing(10) # 与连接按钮隔开

        conn_layout.addWidget(self.btn_connect_klipper)
        conn_layout.addSpacing(5)  # 两个按钮之间留一点小缝隙，不要贴太死
        conn_layout.addWidget(self.btn_estop)

        conn_group.setLayout(conn_layout)





        # ---- 控制按钮 ----

        self.btn_save = QPushButton("保存图片")
        self.btn_detect = QPushButton("识别模具")
        self.btn_golden = QPushButton("黄金示教")
        self.btn_resume = QPushButton("恢复预览")
        self.btn_path = QPushButton("编辑点胶路径")

        # 点胶任务相关按钮
        self.btn_calibrate = QPushButton("标定坐标系")
        # ====== ✅ 新增：验证精度按钮 ======
        self.btn_verify = QPushButton("验证精度")
        self.btn_verify.setToolTip("点击开启/关闭【点击定位】功能")
        # ================================
        self.btn_virtual = QPushButton("虚拟跑")
        self.btn_dry_run = QPushButton("空跑")
        self.btn_real_run = QPushButton("开始点胶")
        self.btn_test_valve = QPushButton("测试出胶")


        self.btn_save.setEnabled(False)

        # 第一排：相机 & 示教
        btn_layout = QHBoxLayout()

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_golden)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_path)

        # 第二排：点胶任务
        task_layout = QHBoxLayout()
        task_layout.addWidget(self.btn_calibrate)
        task_layout.addWidget(self.btn_verify)  # ✅ 验证精度按钮
        task_layout.addWidget(self.btn_virtual)
        task_layout.addWidget(self.btn_dry_run)
        task_layout.addWidget(self.btn_real_run)
        task_layout.addWidget(self.btn_test_valve)

        # ---- 曝光 ----
        self.lbl_exp = QLabel("曝光(us)")
        self.sld_exp = QSlider(Qt.Horizontal)
        self.sld_exp.setRange(0, 100)
        self.spn_exp = QDoubleSpinBox()
        self.spn_exp.setDecimals(0)
        self.spn_exp.setSuffix(" us")

        row_exp = QHBoxLayout()
        row_exp.addWidget(self.sld_exp)
        row_exp.addWidget(self.spn_exp)

        # ---- 增益 ----
        self.lbl_gain = QLabel("增益(dB)")
        self.sld_gain = QSlider(Qt.Horizontal)
        self.sld_gain.setRange(0, 100)
        self.spn_gain = QDoubleSpinBox()
        self.spn_gain.setDecimals(2)
        self.spn_gain.setSingleStep(0.1)
        self.spn_gain.setSuffix(" dB")

        row_gain = QHBoxLayout()
        row_gain.addWidget(self.sld_gain)
        row_gain.addWidget(self.spn_gain)

        # ---- Gamma ----
        self.lbl_gamma = QLabel("Gamma")
        self.sld_gamma = QSlider(Qt.Horizontal)
        self.sld_gamma.setRange(0, 100)
        self.spn_gamma = QDoubleSpinBox()
        self.spn_gamma.setDecimals(2)
        self.spn_gamma.setSingleStep(0.1)

        row_gamma = QHBoxLayout()
        row_gamma.addWidget(self.sld_gamma)
        row_gamma.addWidget(self.spn_gamma)

        # ====== ✅ 第一步：加回翻转勾选框 ======
        from PyQt5.QtWidgets import QCheckBox

        row_flip = QHBoxLayout()
        self.chk_flip_x = QCheckBox("水平翻转 (Flip X)")
        self.chk_flip_y = QCheckBox("垂直翻转 (Flip Y)")

        # 连接信号：一点击就刷新画面
        self.chk_flip_x.stateChanged.connect(self.render_current_frame)
        self.chk_flip_y.stateChanged.connect(self.render_current_frame)

        row_flip.addWidget(self.chk_flip_x)
        row_flip.addWidget(self.chk_flip_y)
        # ====================================


        # ---- 右边参数组 ----
        param_group = QGroupBox("相机参数")
        form = QFormLayout()
        form.addRow(self.lbl_exp, row_exp)
        form.addRow(self.lbl_gain, row_gain)
        form.addRow(self.lbl_gamma, row_gamma)
        param_group.setLayout(form)


        # ---- XYZ 运动控制 ----
        motion_group = QGroupBox("XYZ 运动控制")
        motion_layout = QVBoxLayout()

        # 当前坐标显示（2 位小数）
        coord_layout = QHBoxLayout()
        self.lbl_pos_x = QLabel("X: 0.00")
        self.lbl_pos_y = QLabel("Y: 0.00")
        self.lbl_pos_z = QLabel("Z: 0.00")
        coord_layout.addWidget(self.lbl_pos_x)
        coord_layout.addWidget(self.lbl_pos_y)
        coord_layout.addWidget(self.lbl_pos_z)
        motion_layout.addLayout(coord_layout)

        # ===== Z 高度参数 (安全高度 / 点胶高度) =====
        zparam_layout = QHBoxLayout()

        # 1. 腔内安全高度
        lbl_safe_cavity = QLabel("腔内安全Z")
        self.spn_safe_z_cavity = QDoubleSpinBox()
        self.spn_safe_z_cavity.setRange(0.0, 300.0)
        self.spn_safe_z_cavity.setDecimals(2)
        self.spn_safe_z_cavity.setSingleStep(0.1)
        self.spn_safe_z_cavity.setValue(self.safe_z_cavity)

        # 2. 产品安全高度
        lbl_safe_product = QLabel("产品安全Z")
        self.spn_safe_z_product = QDoubleSpinBox()
        self.spn_safe_z_product.setRange(0.0, 300.0)
        self.spn_safe_z_product.setDecimals(2)
        self.spn_safe_z_product.setSingleStep(0.1)
        self.spn_safe_z_product.setValue(self.safe_z_product)

        # 3. 点胶高度
        lbl_work = QLabel("点胶Z")
        self.spn_work_z = QDoubleSpinBox()
        self.spn_work_z.setRange(0.0, 300.0)
        self.spn_work_z.setDecimals(2)
        self.spn_work_z.setSingleStep(0.1)
        self.spn_work_z.setValue(self.work_z)

        zparam_layout.addWidget(lbl_safe_cavity)
        zparam_layout.addWidget(self.spn_safe_z_cavity)
        zparam_layout.addWidget(lbl_safe_product)
        zparam_layout.addWidget(self.spn_safe_z_product)
        zparam_layout.addWidget(lbl_work)
        zparam_layout.addWidget(self.spn_work_z)

        motion_layout.addLayout(zparam_layout)
        
        # ===== Z 高度 / 对刀 =====
        zbase_group = QGroupBox("Z 高度 / 对刀")
        zbase_layout = QVBoxLayout()
        
        # 1. Z 基准显示
        self.lbl_z_base = QLabel(f"当前 Z 基准: {self.z_base:.2f} mm")
        self.lbl_z_base.setStyleSheet("font-weight: bold; font-size: 14px; color: #FCD34D;")
        
        # 2. 对刀仪状态灯 (圆形 Label)
        status_layout = QHBoxLayout()
        lbl_status_text = QLabel("对刀仪状态:")
        self.lbl_probe_status = QLabel("●")
        self.lbl_probe_status.setStyleSheet("color: #9CA3AF; font-size: 20px;") # 默认灰色
        self.lbl_probe_status.setToolTip("灰色=未知, 绿色=未触发, 红色=已触发")
        status_layout.addWidget(lbl_status_text)
        status_layout.addWidget(self.lbl_probe_status)
        status_layout.addStretch()

        # 3. 按钮
        self.btn_auto_tool_set = QPushButton("一键对刀")
        self.btn_auto_tool_set.setMinimumHeight(30)
        self.btn_auto_tool_set.setStyleSheet("background-color: #2563EB; font-weight: bold;")
        self.btn_auto_tool_set.clicked.connect(self.on_btn_auto_tool_set)
        
        zbase_layout.addWidget(self.lbl_z_base)
        zbase_layout.addLayout(status_layout)
        zbase_layout.addWidget(self.btn_auto_tool_set)
        zbase_group.setLayout(zbase_layout)
        
        motion_layout.addWidget(zbase_group)

        # 步长设置 + 预设按钮
        step_layout = QVBoxLayout()

        step_top = QHBoxLayout()
        lbl_step = QLabel("步长 (mm)")
        self.spn_step = QDoubleSpinBox()
        self.spn_step.setRange(0.01, 100.0)
        self.spn_step.setDecimals(2)        # ★ 只保留 2 位小数
        self.spn_step.setSingleStep(0.01)
        self.spn_step.setValue(self.jog_step_mm)
        step_top.addWidget(lbl_step)
        step_top.addWidget(self.spn_step)
        step_layout.addLayout(step_top)

        # 常用步长按钮
        preset_layout = QHBoxLayout()
        for val in [0.01, 0.1, 1, 5, 10, 25, 50, 100]:
            # 修改：使用简单的 str(val) 或者 f"{val:g}" 来自动处理
            text = f"{val:g}"
            btn = QPushButton(text)
            btn.setCheckable(False)
            btn.clicked.connect(lambda _, v=val: self.spn_step.setValue(v))
            preset_layout.addWidget(btn)
        step_layout.addLayout(preset_layout)

        motion_layout.addLayout(step_layout)


     # 现在 XYZ 控制那块里你应该有这样的布局（精简版）：
        grid = QGridLayout()

        self.btn_y_plus = QPushButton("↑")
        self.btn_x_minus = QPushButton("←")
        self.btn_x_plus = QPushButton("→")
        self.btn_y_minus = QPushButton("↓")

        grid.addWidget(self.btn_y_plus, 0, 1)
        grid.addWidget(self.btn_x_minus, 1, 0)
        grid.addWidget(self.btn_x_plus, 1, 2)
        grid.addWidget(self.btn_y_minus, 2, 1)

        self.btn_z_plus = QPushButton("Z+")
        self.btn_z_minus = QPushButton("Z-")
        grid.addWidget(self.btn_z_plus, 0, 3)
        grid.addWidget(self.btn_z_minus, 2, 3)

        self.btn_home_all = QPushButton("全部归位")
        self.btn_home_x = QPushButton("归位 X")
        self.btn_home_y = QPushButton("归位 Y")
        self.btn_home_z = QPushButton("归位 Z")
        grid.addWidget(self.btn_home_all, 0, 4)
        grid.addWidget(self.btn_home_x,   1, 4)
        grid.addWidget(self.btn_home_y,   2, 4)
        grid.addWidget(self.btn_home_z, 1, 3)

        motion_layout.addLayout(grid)

        # ====== ✅ 新增：速度与加速度调节滑块 ======
        speed_group = QGroupBox("速度 & 加速度调节")
        speed_layout = QVBoxLayout()

        # 1. XY 速度
        row_sxy = QHBoxLayout()
        self.lbl_spd_xy = QLabel(f"XY速度: {int(self.feedrate_xy)}")
        self.sld_spd_xy = QSlider(Qt.Horizontal)
        self.sld_spd_xy.setRange(100, 10000)  # 范围 100 ~ 10000
        self.sld_spd_xy.setValue(int(self.feedrate_xy))
        self.sld_spd_xy.valueChanged.connect(self.on_speed_changed)
        row_sxy.addWidget(self.lbl_spd_xy)
        row_sxy.addWidget(self.sld_spd_xy)

        # 2. Z 速度
        row_sz = QHBoxLayout()
        self.lbl_spd_z = QLabel(f"Z 速度: {int(self.feedrate_z)}")
        self.sld_spd_z = QSlider(Qt.Horizontal)
        self.sld_spd_z.setRange(100, 2000)  # 范围 100 ~ 2000 (Z轴慢点)
        self.sld_spd_z.setValue(int(self.feedrate_z))
        self.sld_spd_z.valueChanged.connect(self.on_speed_changed)
        row_sz.addWidget(self.lbl_spd_z)
        row_sz.addWidget(self.sld_spd_z)

        # 3. 加速度 (Accel)
        row_acc = QHBoxLayout()
        self.lbl_acc = QLabel(f"加速度: {int(self.accel)}")
        self.sld_acc = QSlider(Qt.Horizontal)
        self.sld_acc.setRange(100, 3000)  # 范围 100 ~ 3000
        self.sld_acc.setValue(int(self.accel))
        # 加速度需要松开鼠标才发送指令，防止拖动时发送太多指令卡死 Klipper
        self.sld_acc.sliderReleased.connect(self.on_accel_changed)
        # 同时也连接 valueChanged 只为了更新显示文字
        self.sld_acc.valueChanged.connect(lambda v: self.lbl_acc.setText(f"加速度: {v}"))

        row_acc.addWidget(self.lbl_acc)
        row_acc.addWidget(self.sld_acc)

        speed_layout.addLayout(row_sxy)
        speed_layout.addLayout(row_sz)
        speed_layout.addLayout(row_acc)
        speed_group.setLayout(speed_layout)

        motion_layout.addWidget(speed_group)
        # =========================================

        motion_group.setLayout(motion_layout)
        self.motion_group = motion_group


        # ---- 针头偏移显示 & 标定 ----
        offset_group = QGroupBox("针头偏移 (相机 → 针头)")
        offset_layout = QVBoxLayout()
        self.lbl_offset_x = QLabel("ΔX: 0.000 mm")
        self.lbl_offset_y = QLabel("ΔY: 0.000 mm")
        self.btn_calib_nozzle = QPushButton("标定针头偏移")

        offset_layout.addWidget(self.lbl_offset_x)
        offset_layout.addWidget(self.lbl_offset_y)
        offset_layout.addWidget(self.btn_calib_nozzle)
        offset_group.setLayout(offset_layout)
        self.offset_group = offset_group


        # ---- 日志窗口 ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)

        # ---- 右侧整体布局 ----
        right_layout = QVBoxLayout()
        right_layout.addWidget(info_group)
        right_layout.addWidget(conn_group)          # ✅ 新增：Klipper 连接
        right_layout.addWidget(self.motion_group)   # XYZ 控制
        right_layout.addWidget(self.offset_group)   # 针头偏移
        right_layout.addLayout(btn_layout)
        right_layout.addLayout(task_layout)         # 点胶任务按钮排
        right_layout.addWidget(param_group)
        right_layout.addWidget(self.log_box)
        right_layout.addStretch(1)

        # ---- 总布局 ----
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=4) #左侧视窗大小占比
        main_layout.addLayout(right_layout, stretch=2)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        self.setLayout(main_layout)

        # ================== 信号连接 ==================
        self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop.clicked.connect(self.stop_cam)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_connect_klipper.clicked.connect(self.on_connect_klipper)

        self.btn_detect.clicked.connect(self.detect_mold)
        self.btn_golden.clicked.connect(self.on_golden_teach)
        self.btn_resume.clicked.connect(self.resume_preview)
        self.btn_path.clicked.connect(self.on_edit_path)

        # 标定精度确认
        self.btn_verify.clicked.connect(self.on_toggle_verify_mode)

        # 针头偏移标定
        self.btn_calib_nozzle.clicked.connect(self.on_calibrate_nozzle_offset)


        # 点胶任务按钮
        self.btn_calibrate.clicked.connect(self.on_calibrate_coords)
        self.btn_virtual.clicked.connect(self.on_virtual_run)
        self.btn_dry_run.clicked.connect(self.on_dry_run)
        self.btn_real_run.clicked.connect(self.on_real_run)
        self.btn_test_valve.clicked.connect(self.on_test_valve)

        # 相机参数联动
        self.sld_exp.valueChanged.connect(self.on_exp_slider_changed)
        self.sld_gain.valueChanged.connect(self.on_gain_slider_changed)
        self.sld_gamma.valueChanged.connect(self.on_gamma_slider_changed)

        self.spn_exp.valueChanged.connect(self.on_exp_spin_changed)
        self.spn_gain.valueChanged.connect(self.on_gain_spin_changed)
        self.spn_gamma.valueChanged.connect(self.on_gamma_spin_changed)

        # XYZ 点动按钮
        self.btn_x_minus.clicked.connect(lambda: self.on_jog('x', -1))
        self.btn_x_plus.clicked.connect(lambda: self.on_jog('x', +1))
        self.btn_y_minus.clicked.connect(lambda: self.on_jog('y', -1))
        self.btn_y_plus.clicked.connect(lambda: self.on_jog('y', +1))
        self.btn_z_minus.clicked.connect(lambda: self.on_jog('z', -1))
        self.btn_z_plus.clicked.connect(lambda: self.on_jog('z', +1))

        # 归位按钮
        self.btn_home_all.clicked.connect(lambda: self.on_home_axis('all'))
        self.btn_home_x.clicked.connect(lambda: self.on_home_axis('x'))
        self.btn_home_y.clicked.connect(lambda: self.on_home_axis('y'))
        self.btn_home_z.clicked.connect(lambda: self.on_home_axis('z'))

        # 初始化一次坐标显示 & 针头偏移显示
        self.update_xyz_display()
        self.update_nozzle_offset_display()

        self.update_xyz_display(self.pos_x, self.pos_y, self.pos_z)

        self.spn_safe_z_cavity.valueChanged.connect(self.on_z_params_changed)
        self.spn_safe_z_product.valueChanged.connect(self.on_z_params_changed)
        self.spn_work_z.valueChanged.connect(self.on_z_params_changed)


    def on_connect_klipper(self):
        """
        连接 Klipper：
        1. 拼接 IP 和 Port
        2. 连接成功后，立即强制同步坐标！
        """
        ip = self.edit_ip.text().strip()
        port = self.edit_port.text().strip()

        if not ip or not port:
            QMessageBox.warning(self, "错误", "IP 或 端口不能为空")
            return

        full_url = f"http://{ip}:{port}"

        self.machine.configure(full_url)
        ok = self.machine.connect()

        if ok:
            # 1. 样式变绿
            self.btn_connect_klipper.setText("已连接")
            self.btn_connect_klipper.setStyleSheet("""
                QPushButton {
                    background-color: #10B981;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)

            # ====== ✅ 关键修复：连接后强制同步坐标 ======
            self.machine._sync_toolhead_status()
            mx, my, mz = self.machine.get_position()
            self.update_xyz_display(mx, my, mz)
            self.log(f"🔄 同步坐标成功: X={mx:.2f}, Y={my:.2f}, Z={mz:.2f}")
            self.log(f"✅ 已连接到 Klipper: {full_url}")
            # 开启实时坐标轮询
            if not self.poll_worker.isRunning():
                self.poll_worker.start()
            # ===========================================
        else:
            self.btn_connect_klipper.setText("重试")
            self.btn_connect_klipper.setStyleSheet("""
                QPushButton {
                    background-color: #6B7280;
                    color: white;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            self.log(f"⚠ 连接 Klipper 失败: {full_url}")
            QMessageBox.critical(self, "连接失败", f"无法连接到 {full_url}\n请检查网络或 IP 设置")

    def update_probe_status(self, status):
        """更新探针状态显示 (由 PollingWorker 触发)"""
        if status == 'triggered':
            self.lbl_probe_status.setStyleSheet("color: #EF4444; font-size: 20px;") # 红
            self.lbl_probe_status.setText("● (触发)")
        elif status == 'open':
            self.lbl_probe_status.setStyleSheet("color: #10B981; font-size: 20px;") # 绿
            self.lbl_probe_status.setText("● (未触发)")
        else:
            self.lbl_probe_status.setStyleSheet("color: #9CA3AF; font-size: 20px;") # 灰
            self.lbl_probe_status.setText("● (未知)")


    def on_emergency_stop_clicked(self):
        """点击急停按钮：立即发送 M112"""
        # 急停不需要弹窗确认，必须分秒必争，点下即停
        self.machine.emergency_stop()

        self.log("🚨🚨🚨 已发送急停指令 (M112) 🚨🚨🚨")
        self.log("⚠ 机器已停机。如需继续操作，请在 Fluidd 中执行【Firmware Restart】，然后重新连接。")

        # 更新状态显示
        self.lbl_klipper_status.setText("已急停")
        QMessageBox.critical(self, "急停触发",
                             "🚨 急停指令已发送！\n机器已停止所有动作。\n\n请排除故障后重启 Klipper 固件。")

    # ================== 运动 / 坐标相关 ==================
    # ================== 运动 / 坐标相关 ==================
    def on_jog(self, axis: str, direction: int):
        """
        [异步版] 使用 MovementWorker 执行移动，防止卡顿 UI
        """
        step = float(self.spn_step.value())
        dx = dy = dz = 0.0

        if axis == 'x':
            dx = direction * step
        elif axis == 'y':
            dy = direction * step
        elif axis == 'z':
            dz = direction * step

        # 创建并启动线程
        self.jog_worker = MovementWorker(self.machine, dx, dy, dz, self.feedrate_xy, self.feedrate_z)
        self.jog_worker.finished_signal.connect(self.on_jog_finished)
        self.jog_worker.start()

    def on_jog_finished(self, x, y, z):
        """移动结束后的回调"""
        self.update_xyz_display(x, y, z)
        self.log(f"✅ 移动完成: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

    # ====== ✅ 新增：速度/加速度 响应函数 (请复制这一整块) ======
    def on_speed_changed(self):
        """更新 XY 和 Z 的速度变量"""
        # 从滑块读取数值
        self.feedrate_xy = float(self.sld_spd_xy.value())
        self.feedrate_z = float(self.sld_spd_z.value())

        # 更新标签显示的文字
        self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")
        self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

    def on_accel_changed(self):
        """更新加速度，并直接发送 SET_VELOCITY_LIMIT 指令给 Klipper"""
        val = self.sld_acc.value()
        self.accel = float(val)
        self.lbl_acc.setText(f"加速度: {val}")

        # 发送 Klipper 指令 (只有连接后才发)
        if self.machine.connected:
            # 限制加速度和减速度
            script = f"SET_VELOCITY_LIMIT ACCEL={val} ACCEL_TO_DECEL={val / 2}"
            self.machine._post_gcode(script)
            self.log(f"⚡ 加速度已调整为: {val}")
    # =========================================================

    def move_machine_rel(self, dx: float, dy: float, dz: float):
        """
        相对运动（单位 mm）
        -> 委托给 MachineClient，再从 MachineClient 读回真实坐标。
        """
        if dx == 0 and dy == 0 and dz == 0:
            return

        self.machine.move_rel(dx, dy, dz)
        self.pos_x, self.pos_y, self.pos_z = self.machine.get_position()

        self.update_xyz_display()
        self.log(
            "JOG:",
            f"dX={dx:.2f}, dY={dy:.2f}, dZ={dz:.2f} -> "
            f"X={self.pos_x:.2f}, Y={self.pos_y:.2f}, Z={self.pos_z:.2f}"
        )

    def update_xyz_display(self, x=None, y=None, z=None):
        """
        [修正版] 实事求是显示：
        机器是多少，界面就显示多少。去掉之前的负号欺骗。
        """
        if x is not None: self.pos_x = float(x)
        if y is not None: self.pos_y = float(y)
        if z is not None: self.pos_z = float(z)

        # ====== 👇 修改这里：直接显示，不要加负号 ======
        display_x = self.pos_x
        display_y = self.pos_y
        display_z = self.pos_z

        self.lbl_pos_x.setText(f"X: {display_x:.2f}")
        self.lbl_pos_y.setText(f"Y: {display_y:.2f}")
        self.lbl_pos_z.setText(f"Z: {display_z:.2f}")
        # ====== ✅ 新增：Z轴基准日志显示 ======
        if hasattr(self, "lbl_z_ref"):
            self.lbl_z_ref.setText(f"当前 Z 基准: {self.pos_z:.2f} mm")
    def on_home_axis(self, axis: str):
        """
        单轴 / 全部归位：X / Y / Z / all
        (多线程修复版：不再卡顿相机画面)
        """
        self.log(f"⏳ 正在后台执行归位 ({axis.upper()})... 请稍候")

        # 1. 创建线程任务
        self.home_worker = HomeWorker(self.machine, axis)

        # 2. 连接信号：当线程结束时，执行 self.on_home_finished
        self.home_worker.finished_signal.connect(self.on_home_finished)

        # 3. 启动线程
        self.home_worker.start()

        # 此时主线程立刻获得自由，相机画面继续流畅刷新！

    def on_home_finished(self, x, y, z):
        """归位线程结束后的回调函数"""
        # 更新界面坐标
        self.update_xyz_display(x, y, z)
        self.log(f"✅ 归位完成: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

        # 不要在这里销毁线程对象，否则会报 "QThread: Destroyed while thread is still running"
        # self.home_worker = None

    # ================== 相机相关 ==================




    # ---- 一键对刀（占位） ----
    def on_tool_probe_clicked(self):
        """
        一键对刀入口（当前是占位实现，后续对接 Klipper 对刀宏）。
        """
        QMessageBox.information(
            self,
            "一键对刀（占位）",
            "对刀仪一键对刀功能尚未与 Klipper 宏联动，\n"
            "当前版本仅作为 UI 按钮和日志占位。"
        )
        self.log("TODO", "一键对刀逻辑将在后续版本中与 Klipper 对刀宏对接。")

    # ================== 针头偏移相关 ==================
    def update_nozzle_offset_display(self):
        """根据当前 self.nozzle_offset_x/y 刷新界面显示。"""
        if hasattr(self, "lbl_offset_x"):
            self.lbl_offset_x.setText(f"ΔX: {self.nozzle_offset_x:.3f} mm")
        if hasattr(self, "lbl_offset_y"):
            self.lbl_offset_y.setText(f"ΔY: {self.nozzle_offset_y:.3f} mm")

    def on_calibrate_nozzle_offset(self):
        """
        标定针头偏移：相机对点 (Step 1) -> 针头对点 (Step 2) -> 计算
        """
        if self.nozzle_calib_step == 0:
            # 启动流程
            self.nozzle_calib_step = 1
            self.show_crosshair = True  # 显示十字准星
            self.log("▶ 启动针头偏移标定 (步骤 1/2)")
            self.btn_calib_nozzle.setText("记录相机中心位置")
            self.btn_calib_nozzle.setStyleSheet("background-color: #F59E0B;")  # 橙色

            QMessageBox.information(
                self,
                "针头标定 - 步骤 1",
                "请将相机中心（红十字）对准一个清晰的物理标记点（例如模具的尖角），然后点击按钮记录坐标。"
            )

        elif self.nozzle_calib_step == 1:
            # 记录相机位置
            self.cam_x, self.cam_y, self.cam_z = self.machine.get_position()  # 记录Z轴，防止后续出错
            self.nozzle_calib_step = 2
            self.log(f"📍 记录相机中心 ({self.cam_x:.3f}, {self.cam_y:.3f})")
            self.btn_calib_nozzle.setText("记录针头中心位置")

            QMessageBox.information(
                self,
                "针头标定 - 步骤 2",
                "请手动控制机器移动，使针头尖端对准刚才的那个物理标记点，然后点击按钮记录坐标。"
            )

        elif self.nozzle_calib_step == 2:
            # 记录针头位置并计算偏移
            self.nozzle_x, self.nozzle_y, _ = self.machine.get_position()

            # 计算偏移量： ΔX = 针头位置 - 相机位置
            delta_x = self.nozzle_x - self.cam_x
            delta_y = self.nozzle_y - self.cam_y

            self.nozzle_offset_x = delta_x
            self.nozzle_offset_y = delta_y

            # 刷新显示
            self.update_nozzle_offset_display()

            self.log(f"✅ 针头偏移标定完成！")
            self.log(f"   ΔX: {delta_x:.3f} mm")
            self.log(f"   ΔY: {delta_y:.3f} mm")

            self.save_app_settings()  # ✅ 保存参数
            QMessageBox.information(self, "完成", "针头偏移量已计算并保存！")

            # 流程结束，恢复状态
            self.nozzle_calib_step = 0
            self.show_crosshair = False
            self.btn_calib_nozzle.setText("标定针头偏移")
            self.btn_calib_nozzle.setStyleSheet("")

    def on_wheel_zoom(self, delta_y: int):
        """
        鼠标滚轮缩放：向上放大，向下缩小
        """
        if self.last_frame is None:
            return

        # 每次滚轮调整 10%
        if delta_y > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1

        # 如果缩回 1 倍以下，我们就锁在 1 倍，并重新按窗口适配
        if self.zoom_scale <= 1.0:
            self.zoom_scale = 1.0
            self.base_scale = None  # 下次渲染按窗口重新计算“fit”比例

        # 根据倍数切换滚动条显示
        if self.zoom_scale > 1.0:
            # 放大时允许出现滚动条
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            # 1 倍时不显示滚动条，画面居中
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # 顺便把滚动条位置调回“中间”
            hbar = self.scroll_area.horizontalScrollBar()
            vbar = self.scroll_area.verticalScrollBar()
            hbar.setValue((hbar.minimum() + hbar.maximum()) // 2)
            vbar.setValue((vbar.minimum() + vbar.maximum()) // 2)

        self.render_current_frame()

    def on_pan_delta(self, dx: int, dy: int):
        """
        鼠标左键拖拽 -> 平移视图
        dx, dy 是鼠标移动量（像素）
        """
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()

        # 鼠标向右拖动，希望看到右边内容 => 滚动条减少
        hbar.setValue(hbar.value() - dx)
        vbar.setValue(vbar.value() - dy)
    def start_cam(self):

        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

        if deviceList.nDeviceNum == 0:
            self.log("No camera found")
            return

        dev_info = ctypes.cast(
            deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)
        ).contents

        self.cam = MvCamera()
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_Mono8)

        try:
            self.cam.MV_CC_SetBoolValue("GammaEnable", True)
        except:
            pass

        self.read_param_ranges()
        self.read_cam_info()
        
        # 1. 先尝试应用一次参数 (暂时屏蔽，防止报错 0x80000003)
        # self.apply_saved_params()

        # 2. 启动取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.log(f"❌ 相机启动取流失败，错误码: {ret}")
            return

        self.sync_from_camera_to_ui()

        self.thread = CameraThread(self.cam)
        self.thread.frame_signal.connect(self.update_image)
        self.thread.start()
        
        # 3. 【关键】延时 1 秒再次应用参数
        # 某些相机在 StartGrabbing 后会重置参数，或者需要一点时间才能接收参数
        QTimer.singleShot(1000, self.apply_saved_params)

        self.btn_start.setEnabled(False)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(True)

    def stop_cam(self):
        if not self.cam:
            return

        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.cam = None

        self.label.setText("已停止")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.lbl_cam_res.setText("-")
        self.lbl_cam_fps.setText("-")
        self.lbl_cam_fmt.setText("-")

    def read_param_ranges(self):
        # 曝光
        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("ExposureTime", v)
            self.exp_min = max(self.exp_min, v.fMin)
            self.exp_max = min(self.exp_max, v.fMax)
        except:
            pass

        # 增益
        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gain", v)
            self.gain_min = max(self.gain_min, v.fMin)
            self.gain_max = min(self.gain_max, v.fMax)
        except:
            pass

        # Gamma
        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gamma", v)
            self.gamma_min = max(self.gamma_min, v.fMin)
            self.gamma_max = min(self.gamma_max, v.fMax)
        except:
            pass

        self.spn_exp.setRange(self.exp_min, self.exp_max)
        self.spn_gain.setRange(self.gain_min, self.gain_max)
        self.spn_gamma.setRange(self.gamma_min, self.gamma_max)

    def read_cam_info(self):
        if not self.cam:
            return

        try:
            w_val = MVCC_INTVALUE()
            h_val = MVCC_INTVALUE()
            self.cam.MV_CC_GetIntValue("Width", w_val)
            self.cam.MV_CC_GetIntValue("Height", h_val)
            self.lbl_cam_res.setText(f"{w_val.nCurValue} x {h_val.nCurValue}")
        except:
            self.lbl_cam_res.setText("-")

        try:
            fps_val = MVCC_FLOATVALUE()
            ret = self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", fps_val)
            if ret != 0:
                ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", fps_val)
            if ret == 0:
                self.lbl_cam_fps.setText(f"{fps_val.fCurValue:.2f} fps")
            else:
                self.lbl_cam_fps.setText("-")
        except:
            self.lbl_cam_fps.setText("-")

        try:
            enum_val = MVCC_ENUMVALUE()
            ret = self.cam.MV_CC_GetEnumValue("PixelFormat", enum_val)
            if ret == 0:
                if enum_val.nCurValue == PixelType_Gvsp_Mono8:
                    self.lbl_cam_fmt.setText("Mono8")
                else:
                    self.lbl_cam_fmt.setText(f"0x{enum_val.nCurValue:X}")
            else:
                self.lbl_cam_fmt.setText("-")
        except:
            self.lbl_cam_fmt.setText("-")

    def load_saved_params(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.saved_params = json.load(f)
                
                # ✅ 加载后立即更新 UI 显示
                if "ExposureTime" in self.saved_params:
                    val = float(self.saved_params["ExposureTime"])
                    if hasattr(self, 'spn_exp'):
                        self.spn_exp.setValue(val)
                        # 滑块也同步一下 (注意范围映射)
                        # 这里简单处理，假设 spinbox change 会联动 slider
                
                if "Gain" in self.saved_params:
                    val = float(self.saved_params["Gain"])
                    if hasattr(self, 'spn_gain'):
                        self.spn_gain.setValue(val)
                
                if "Gamma" in self.saved_params:
                    val = float(self.saved_params["Gamma"])
                    if hasattr(self, 'spn_gamma'):
                        self.spn_gamma.setValue(val)
                        
                self.log(f"✅ 已加载相机参数: {self.config_path}")
            else:
                self.saved_params = {}
                self.log("ℹ️ 未找到相机参数文件，使用默认值。")
        except Exception as e:
            self.log(f"读取参数文件失败：{e}")
            self.saved_params = {}

    def init_calibration(self):
        """从 calib 目录加载相机标定参数（camera_matrix.npy / dist_coeffs.npy）"""
        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "config", "calib")
            cam_path = os.path.join(calib_dir, "camera_matrix.npy")
            dist_path = os.path.join(calib_dir, "dist_coeffs.npy")

            if os.path.exists(cam_path) and os.path.exists(dist_path):
                self.K = np.load(cam_path)
                self.D = np.load(dist_path)
                self._calib_loaded = True
                msg = "✅ 已加载标定参数 camera_matrix.npy / dist_coeffs.npy"
            else:
                self._calib_loaded = False
                msg = "⚠ config/calib 目录下未找到 camera_matrix.npy 或 dist_coeffs.npy，将使用原始图像"

        except Exception as e:
            self._calib_loaded = False
            msg = f"⚠ 加载标定参数失败：{e}"

        if hasattr(self, "log_box"):
            self.log(msg)
        else:
            print(msg)

        # ====== ✅ 新增：加载坐标标定矩阵 (cam2mach_matrix.npy) ======
        try:
            if os.path.exists("config/cam2mach_matrix.npy"):
                self.cam2mach_matrix = np.load("config/cam2mach_matrix.npy")
                self.log("✅ 已加载坐标标定矩阵 cam2mach_matrix.npy")
            else:
                self.cam2mach_matrix = None
                self.log("ℹ️ 未找到坐标标定矩阵，请先执行【标定坐标系】")
        except Exception as e:
            self.cam2mach_matrix = None
            self.log(f"❌ 加载坐标标定矩阵失败: {e}")

    # ===== 黄金模板 =====
    def load_golden_templates(self):
        try:
            if os.path.exists("config/slot_templates.npy"):
                self.slot_templates = np.load("config/slot_templates.npy")
                self.log(f"✅ 黄金模板 config/slot_templates.npy 加载成功，形状：{self.slot_templates.shape}")
            else:
                self.log("⚠ 未找到 config/slot_templates.npy，暂时无法做黄金对位")
        except Exception as e:
            self.log("加载黄金模板失败：", e)
            self.slot_templates = None

    def apply_saved_params(self):
        if not self.cam or not self.saved_params:
            return

        if "ExposureTime" in self.saved_params:
            self.set_exposure(float(self.saved_params["ExposureTime"]))
        if "Gain" in self.saved_params:
            self.set_gain(float(self.saved_params["Gain"]))
        if "Gamma" in self.saved_params:
            self.set_gamma(float(self.saved_params["Gamma"]))

    def save_params(self):
        try:
            data = {
                "ExposureTime": float(self.spn_exp.value()),
                "Gain": float(self.spn_gain.value()),
                "Gamma": float(self.spn_gamma.value())
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log("保存参数失败：", e)

    def load_app_settings(self):
        """加载应用设置：速度和加速度"""
        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

                # 1. 恢复 XY 速度
                if 'feedrate_xy' in settings:
                    val = float(settings['feedrate_xy'])
                    self.sld_spd_xy.setValue(int(val))  # 更新滑块
                    self.feedrate_xy = val  # 更新内部变量
                    self.lbl_spd_xy.setText(f"XY速度: {int(val)}")  # 更新标签

                # 2. 恢复 Z 速度
                if 'feedrate_z' in settings:
                    val = float(settings['feedrate_z'])
                    self.sld_spd_z.setValue(int(val))
                    self.feedrate_z = val
                    self.lbl_spd_z.setText(f"Z 速度: {int(val)}")

                # 3. 恢复 加速度
                if 'accel' in settings:
                    val = float(settings['accel'])
                    self.sld_acc.setValue(int(val))
                    self.accel = val
                    self.lbl_acc.setText(f"加速度: {int(val)}")

                # 4. 恢复安全高度 Z (旧版兼容 + 新版)
                if 'safe_z_cavity' in settings:
                    val = float(settings['safe_z_cavity'])
                    self.safe_z_cavity = val
                    if hasattr(self, 'spn_safe_z_cavity'):
                        self.spn_safe_z_cavity.setValue(val)
                
                if 'safe_z_product' in settings:
                    val = float(settings['safe_z_product'])
                    self.safe_z_product = val
                    if hasattr(self, 'spn_safe_z_product'):
                        self.spn_safe_z_product.setValue(val)
                
                # 兼容旧配置：如果只有 safe_z，则赋给 safe_z_cavity
                if 'safe_z' in settings and 'safe_z_cavity' not in settings:
                    val = float(settings['safe_z'])
                    self.safe_z_cavity = val
                    if hasattr(self, 'spn_safe_z_cavity'):
                        self.spn_safe_z_cavity.setValue(val)

                # 5. 恢复点胶高度 Z
                if 'work_z' in settings:
                    val = float(settings['work_z'])
                    self.work_z = val
                    if hasattr(self, 'spn_work_z'):
                        self.spn_work_z.setValue(val)

                # 6. 恢复对刀基准
                if 'z_base' in settings:
                    self.z_base = float(settings['z_base'])
                    if hasattr(self, 'lbl_z_base'):
                        self.lbl_z_base.setText(f"当前 Z 基准: {self.z_base:.2f} mm")

                # 7. 恢复对刀仪坐标
                if 'tool_setter_x' in settings and 'tool_setter_y' in settings:
                    tx = settings['tool_setter_x']
                    ty = settings['tool_setter_y']
                    if tx is not None and ty is not None:
                        self.tool_setter_x = float(tx)
                        self.tool_setter_y = float(ty)

                # 8. 恢复针头偏移
                if 'nozzle_offset_x' in settings and 'nozzle_offset_y' in settings:
                    nx = settings['nozzle_offset_x']
                    ny = settings['nozzle_offset_y']
                    if nx is not None and ny is not None:
                        self.nozzle_offset_x = float(nx)
                        self.nozzle_offset_y = float(ny)
                        self.update_nozzle_offset_display()

            # 可以在日志中记录成功加载
            self.log("✅ 成功加载历史速度/加速度参数。")

        except Exception as e:
            self.log(f"❌ 加载应用参数失败: {e}")

    def save_app_settings(self):
        """保存应用设置：速度和加速度"""
        settings = {
            'feedrate_xy': self.feedrate_xy,
            'feedrate_z': self.feedrate_z,
            'accel': self.accel,

            'safe_z_cavity': self.safe_z_cavity,
            'safe_z_product': self.safe_z_product,
            'work_z': self.work_z,
            'z_base': self.z_base,
            'tool_setter_x': self.tool_setter_x,
            'tool_setter_y': self.tool_setter_y,
            'nozzle_offset_x': self.nozzle_offset_x,
            'nozzle_offset_y': self.nozzle_offset_y,
        }

        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"❌ 保存应用参数失败: {e}")



    def update_image(self, img: np.ndarray):
        self.last_frame = img.copy()
        if self.freeze_display:
            return
        self.render_current_frame()

    def render_current_frame(self):
        """硬编码翻转版 + 验证显示增强 + 修复标定点不显示BUG"""
        if self.last_frame is None:
            return

        # 1. 拷贝 & 翻转 (保持 -1)
        img = self.last_frame.copy()
        img = cv2.flip(img, -1)

        # 2. 转成 BGR
        if len(img.shape) == 2:
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = img

        h, w = vis_img.shape[:2]
        cx, cy = w // 2, h // 2

        # ====== 🎨 A. 验证模式显示逻辑 ======
        if self.verify_mode:
            # 画中心十字
            cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)
            cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)
            cv2.circle(vis_img, (cx, cy), 15, (0, 0, 255), 2)

            # 画点击的目标点
            if self.verify_click_pt is not None:
                vx, vy = int(self.verify_click_pt[0]), int(self.verify_click_pt[1])
                cv2.drawMarker(vis_img, (vx, vy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(vis_img, (vx, vy), 5, (0, 255, 0), -1)
                cv2.putText(vis_img, "Target", (vx + 10, vy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis_img, "VERIFY MODE: Click any point to move center here", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ====== 🛠️ B. 标定阶段 1 (视觉微调模式) - 【找回丢失的代码！】 ======
        elif self.calib_step == 10:
            if len(self.calib_vision_pts) == 4:
                for i, pt in enumerate(self.calib_vision_pts):
                    px, py = int(pt[0]), int(pt[1])
                    if i == self.calib_sel_idx:
                        # 选中态：红色实心 + 黄色大圈 + 辅助十字
                        cv2.circle(vis_img, (px, py), 6, (0, 0, 255), -1)
                        cv2.circle(vis_img, (px, py), 20, (0, 255, 255), 2)
                        cv2.line(vis_img, (px - 30, py), (px + 30, py), (0, 255, 255), 1)
                        cv2.line(vis_img, (px, py - 30), (px, py + 30), (0, 255, 255), 1)
                        cv2.putText(vis_img, f"#{i + 1} Moving", (px + 20, py - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        # 普通态：绿色空心
                        cv2.circle(vis_img, (px, py), 6, (0, 255, 0), 2)
                        cv2.putText(vis_img, f"#{i + 1}", (px + 10, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                msg = "ADJUST: Press [1-4] to Switch | [WASD] to Move"
                cv2.putText(vis_img, msg, (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ====== 🎨 E. 空跑路径显示 (新增 + 相机中心十字 = 点胶针) ======
        # 优先处理普通模式下的十字中心显示 (用户请求)
        if self.show_center_crosshair:
             # 画中心十字 (全屏长线)
            cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)
            cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)
            # 画个小圈
            cv2.circle(vis_img, (cx, cy), 20, (0, 0, 255), 2)
            
            # 显示文字提示
            cv2.putText(vis_img, "CENTER CROSSHAIR", (20, h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if len(self.dry_run_paths_to_draw) > 0:
            # 1️⃣ 空跑时先画“相机中心 = 点胶针”十字线
            cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)   # 竖线
            cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)   # 横线
            cv2.circle(vis_img, (cx, cy), 15, (0, 0, 255), 2)     # 中心小圈

            # 2️⃣ 再遍历每一条空跑路径画出来
            # paths_to_draw 现在是 List[List[path]] (产品 -> 路径片段)
            # 或者为了兼容旧代码，我们先把它展平再画
            flat_paths = []
            for prod_paths in self.dry_run_paths_to_draw:
                if isinstance(prod_paths, list):
                    flat_paths.extend(prod_paths)
                else:
                    # 兼容旧格式 (如果 mold_detector 没更新)
                    flat_paths.append(prod_paths)

            for path in flat_paths:
                # path 是 (N, 2) 的数组，需要转成 int32
                pts = path.reshape((-1, 1, 2)).astype(np.int32)
                
                if len(path) == 1:
                    # Draw Dot (Magenta)
                    cx, cy = int(path[0][0]), int(path[0][1])
                    cv2.circle(vis_img, (cx, cy), 8, (255, 0, 255), -1)
                else:
                    cv2.polylines(vis_img, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)

                    # 标记起点
                    if len(path) > 0:
                        sx, sy = int(path[0][0]), int(path[0][1])
                        cv2.circle(vis_img, (sx, sy), 5, (0, 0, 255), -1)

            cv2.putText(vis_img, "DRY RUNNING...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)





        # ====== C. 标定阶段 2 (机器对位模式) ======
        elif 1 <= self.calib_step <= 4:
            if self.show_crosshair:
                cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)
                cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)

            if len(self.calib_vision_pts) == 4:
                target = self.calib_vision_pts[self.calib_step - 1]
                tx, ty = int(target[0]), int(target[1])
                cv2.arrowedLine(vis_img, (cx, cy), (tx, ty), (0, 255, 0), 3)
                cv2.circle(vis_img, (tx, ty), 10, (0, 255, 255), 2)

        # ====== D. 正常检测模式 (优化：跳帧检测) ======
        elif self.live_detect:
            # 初始化计数器
            if not hasattr(self, '_detect_frame_count'):
                self._detect_frame_count = 0
                self._last_detected_img = None
            
            self._detect_frame_count += 1
            
            # 每 3 帧检测一次 (降低 66% 负荷)
            if self._detect_frame_count % 3 == 0:
                from core.mold_detector import detect_mold_and_cells
                detected_img, _, _, _ = detect_mold_and_cells(img)
                self._last_detected_img = detected_img
                vis_img = detected_img
            else:
                # 中间帧：如果有缓存就用缓存，没有就用原图
                if self._last_detected_img is not None:
                    vis_img = self._last_detected_img
                else:
                    vis_img = img
        else:
            # 如果关闭了检测，重置计数器
            if hasattr(self, '_detect_frame_count'):
                self._detect_frame_count = 0
                self._last_detected_img = None

        # -----------------------------------------------------------
        # 3. 显示
        # -----------------------------------------------------------
        try:
            rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            qpix = QPixmap.fromImage(qimg)

            view_w = self.scroll_area.viewport().width()
            view_h = self.scroll_area.viewport().height()
            if view_w <= 10: view_w = 640
            if view_h <= 10: view_h = 480

            temp_pix = qpix.scaled(view_w, view_h, Qt.KeepAspectRatio, Qt.FastTransformation)
            base_w = temp_pix.width()
            base_h = temp_pix.height()

            final_w = int(base_w * self.zoom_scale)
            final_h = int(base_h * self.zoom_scale)

            final_pix = qpix.scaled(final_w, final_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            if self.label.size() != final_pix.size():
                self.label.resize(final_pix.size())

            self.label.setPixmap(final_pix)

        except Exception as e:
            self.log("[render_current_frame] 渲染出错：", e)

    def save_image(self):
        if self.last_frame is None:
            self.log("❌ 没有图像可以保存")
            return

        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
            self.log("⏸ 暂停取流，用于安全保存图片")

        frame_to_save = self.last_frame.copy()

        os.makedirs("captures", exist_ok=True)
        default_name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.png")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            os.path.join("captures", default_name),
            "Image Files (*.png *.jpg *.bmp)"
        )

        if save_path:
            ok = cv2.imwrite(save_path, frame_to_save)
            if ok:
                self.log(f"📸 已保存图片：{save_path}")
                self.latest_capture_path = save_path  # ✅ 记录路径
            else:
                self.log("❌ 保存图片失败（cv2.imwrite 返回 False）")
        else:
            self.log("⚠ 已取消保存图片")

        if self.cam:
            self.thread = CameraThread(self.cam)
            self.thread.frame_signal.connect(self.update_image)
            self.thread.start()
            self.log("▶ 相机取流已恢复")

    # ================== 识别模具 ==================
    def detect_mold(self):
        if self.last_frame is None:
            self.log("当前无图像，先启动相机")
            return

        self.live_detect = not self.live_detect

        if self.live_detect:
            self.btn_detect.setText("停止识别")
            self.log("✅ 已开启实时识别（每帧都会画大框+小框）")
        else:
            self.btn_detect.setText("识别模具")
            self.log("⏹ 已关闭实时识别")

        # 切换状态后，用当前帧重新渲染一次
        self.render_current_frame()

    # ================== 黄金示教 ==========================
    def on_golden_teach(self):
        """点击‘黄金示教’，使用本地 mold2.png 文件进行示教"""
        self.freeze_display = True

        # ====== ✅ 3按钮选择逻辑 ======
        candidate_path = self.latest_capture_path
        if not candidate_path:
            if os.path.exists("captures"):
                files = [os.path.join("captures", f) for f in os.listdir("captures") if f.lower().endswith(('.png', '.jpg', '.bmp'))]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(x))
                    candidate_path = files[-1]

        if candidate_path and os.path.exists(candidate_path):
            filename = os.path.basename(candidate_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(candidate_path)).strftime('%H:%M:%S')

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("选择模板源")
            msg_box.setText(f"检测到最新图片：{filename} ({mtime})\n请选择要使用的模板：")
            
            btn_use_latest = msg_box.addButton("使用最新图片", QMessageBox.AcceptRole)
            btn_use_mold2 = msg_box.addButton("使用现有 mold2", QMessageBox.ActionRole)
            btn_cancel = msg_box.addButton("取消", QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            if msg_box.clickedButton() == btn_use_latest:
                try:
                    # 1. 备份旧的
                    if os.path.exists("config/mold2.png"):
                        shutil.copy("config/mold2.png", "config/mold2.bak")
                        self.log("ℹ️ 已备份原 config/mold2.png 为 config/mold2.bak")

                    # 2. 读取最新图片并旋转 180 度
                    img_latest = cv2.imread(candidate_path)
                    if img_latest is not None:
                        # 旋转 180 度 (flip -1)
                        img_rotated = cv2.flip(img_latest, -1)
                        cv2.imwrite("config/mold2.png", img_rotated)
                        self.log(f"✅ 已旋转180度并更新 config/mold2.png 为: {filename}")
                    else:
                        self.log("❌ 读取最新图片失败")
                        self.freeze_display = False
                        return
                except Exception as e:
                    self.log(f"❌ 处理图片失败: {e}")
                    self.freeze_display = False
                    return
            elif msg_box.clickedButton() == btn_use_mold2:
                self.log("ℹ️ 继续使用现有 config/mold2.png")
            else:
                self.log("🚫 操作已取消")
                self.freeze_display = False
                return
        # ============================

        if not os.path.exists("config/mold2.png"):
            self.log("❌ 未找到 config/mold2.png，请先准备好这张图片")
            self.freeze_display = False
            return

        from core.golden_teach import run_golden_teach
        ok = run_golden_teach()

        self.freeze_display = False

        if ok:
            self.log("✅ 黄金示教完成，重新加载模板")
            reload_golden_templates()
        else:
            self.log("⚠ 黄金示教取消或失败")

    # ================== 十字中心开关 ==================
    def on_toggle_crosshair(self):
        """切换显示中心十字"""
        self.show_center_crosshair = not self.show_center_crosshair
        
        if self.show_center_crosshair:
            self.btn_show_crosshair.setText("关闭中心")
            self.log("✅ 已开启中心十字显示")
        else:
            self.btn_show_crosshair.setText("显示中心")
            self.log("⏹ 已关闭中心十字显示")
            
        self.render_current_frame()

    # ================== 点胶路径编辑 ==================
    def on_edit_path(self):
        """点击按钮：在黄金图片中编辑某个小框的点胶路径"""
        self.log("🧵 打开点胶路径示教窗口（默认小框 index=0）")

        # ====== ✅ 3按钮选择逻辑 ======
        candidate_path = self.latest_capture_path
        if not candidate_path:
            if os.path.exists("captures"):
                files = [os.path.join("captures", f) for f in os.listdir("captures") if f.lower().endswith(('.png', '.jpg', '.bmp'))]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(x))
                    candidate_path = files[-1]

        if candidate_path and os.path.exists(candidate_path):
            filename = os.path.basename(candidate_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(candidate_path)).strftime('%H:%M:%S')

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("选择模板源")
            msg_box.setText(f"检测到最新图片：{filename} ({mtime})\n请选择要使用的模板：")
            
            btn_use_latest = msg_box.addButton("使用最新图片", QMessageBox.AcceptRole)
            btn_use_mold2 = msg_box.addButton("使用现有 mold2", QMessageBox.ActionRole)
            btn_cancel = msg_box.addButton("取消", QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            if msg_box.clickedButton() == btn_use_latest:
                try:
                    # 1. 备份旧的
                    if os.path.exists("config/mold2.png"):
                        shutil.copy("config/mold2.png", "config/mold2.bak")
                        self.log("ℹ️ 已备份原 config/mold2.png 为 config/mold2.bak")

                    # 2. 读取最新图片并旋转 180 度
                    img_latest = cv2.imread(candidate_path)
                    if img_latest is not None:
                        # 旋转 180 度 (flip -1)
                        img_rotated = cv2.flip(img_latest, -1)
                        cv2.imwrite("config/mold2.png", img_rotated)
                        self.log(f"✅ 已旋转180度并更新 config/mold2.png 为: {filename}")
                    else:
                        self.log("❌ 读取最新图片失败")
                        return
                except Exception as e:
                    self.log(f"❌ 处理图片失败: {e}")
                    return
            elif msg_box.clickedButton() == btn_use_mold2:
                self.log("ℹ️ 继续使用现有 config/mold2.png")
            else:
                self.log("🚫 操作已取消")
                return
        # ============================
        try:
            from core.path_teach import run_path_teach
            ok = run_path_teach(cell_index=0, img_path="config/mold2.png")
            if ok:
                self.log("✅ 点胶路径已保存到 disp_poly_cell0.npy / disp_dots_cell0.npy，已重新加载")
                from core.mold_detector import reload_golden_templates
                reload_golden_templates()
            else:
                self.log("⚠ 点胶路径编辑已取消或未保存")
        except Exception as e:
            self.log(f"❌ 打开路径示教失败：{e}")

    # ================== 点胶任务按钮（占位） ==================
        # 替换原有的 on_calibrate_coords
    def on_calibrate_coords(self):
        """标定流程主控：自动识别 -> 进入微调 -> 4步记录 -> 计算"""

        # --- 阶段 1: 启动 & 自动识别 ---
        if self.calib_step == 0:
            if self.last_frame is None:
                self.log("❌ 请先启动相机")
                return

            # ====== 🛠️ 关键修改：识别前先翻转图像 ======
            # 必须与 render_current_frame 里的翻转逻辑完全一致！
            # 之前我们用了 cv2.flip(img, -1)，这里也要用 -1
            img_to_detect = self.last_frame.copy()
            img_to_detect = cv2.flip(img_to_detect, -1)
            # =========================================

            # 1. 使用翻转后的图去识别
            from core.mold_detector import detect_mold_and_cells
            _, big_box, _, _ = detect_mold_and_cells(img_to_detect)

            if big_box is not None:
                self.calib_vision_pts = big_box.astype(np.float32)
                # 排序: 左上, 右上, 右下, 左下
                from core.mold_detector import _order_points
                self.calib_vision_pts = _order_points(self.calib_vision_pts)
                self.log("✅ 自动识别到大框，进入微调模式。")
            elif os.path.exists("golden_corners.npy"):
                # 如果没有识别到，加载旧数据。注意：旧数据如果是翻转前保存的，可能也会不对。
                # 建议第一次都重新自动识别一遍。
                self.calib_vision_pts = np.load("golden_corners.npy")
                self.log("⚠ 自动识别失败，加载上次保存的角点。")
            else:
                QMessageBox.warning(self, "失败", "无法识别模具且无历史数据。\n请确保模具在画面中，或先进行示教。")
                return

            # ... (后续弹出微调对话框的代码保持不变) ...
            # 进入微调状态 (状态码设为 10)
            self.calib_step = 10
            self.calib_sel_idx = 0
            self.btn_calibrate.setText("确认视觉角点 (下一步)")
            self.btn_calibrate.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")

            reply = QMessageBox.information(
                self,
                "微调模式",
                "【步骤 1/2：微调视觉角点】\n\n"
                "1. 滚动滚轮放大画面。\n"
                "2. 点击画面中的红圈选中角点，使用【WASD】键微调位置。\n"
                "3. 确保 4 个角点都精准对齐后，再次点击按钮进入下一步。",
                QMessageBox.Ok | QMessageBox.Cancel
            )

            if reply == QMessageBox.Cancel:
                self.calib_step = 0
                self.btn_calibrate.setText("标定坐标系")
                self.btn_calibrate.setStyleSheet("")
                self.log("🚫 标定操作已取消")
                self.render_current_frame()
                return
            # =========== 修改结束 ===========

        # --- 阶段 2: 确认微调，进入机器对位 ---
        elif self.calib_step == 10:
            # 保存微调好的点，下次可以用
            np.save("golden_corners.npy", self.calib_vision_pts)

            # 切换到机器对位流程 (状态 1)
            self.calib_step = 1
            self.calib_machine_pts = []
            self.show_crosshair = True

            self.btn_calibrate.setText("记录第 1 点 (左上)")
            self.btn_calibrate.setStyleSheet("background-color: #F59E0B; color: black; font-weight: bold;")  # 橙色

            self.log("✅ 视觉角点已确认。进入机器对位流程。")
            self.log("👉 请将相机中心（十字准星）对准模具【左上角】，然后点击按钮。")

        # --- 阶段 3: 记录机器坐标 (1-4) ---
        elif 1 <= self.calib_step <= 4:
            mx, my, mz = self.machine.get_position()
            self.calib_machine_pts.append([mx, my])
            self.log(f"📍 记录机器点 {self.calib_step}: ({mx:.2f}, {my:.2f})")

            self.calib_step += 1

            if self.calib_step <= 4:
                names = ["左上", "右上", "右下", "左下"]
                next_name = names[self.calib_step - 1]
                self.btn_calibrate.setText(f"记录第 {self.calib_step} 点 ({next_name})")
                self.log(f"👉 下一步：对准【{next_name}】")
            else:
                self.finish_calibration()

    # 精确确认
    def on_toggle_verify_mode(self):
        """切换验证模式的开关"""
        if self.cam2mach_matrix is None:
            QMessageBox.warning(self, "无法验证", "请先完成【标定坐标系】！")
            return

        self.verify_mode = not self.verify_mode

        if self.verify_mode:
            self.btn_verify.setText("停止验证")
            self.btn_verify.setStyleSheet("background-color: #10B981; color: white;")  # 变绿
            QMessageBox.information(self, "验证模式开启",
                                    "点击画面任意点，机器将尝试将其移至中心。\n\n目标：绿色标记应与红色十字重合。")
        else:
            self.btn_verify.setText("验证精度")
            self.btn_verify.setStyleSheet("")  # 恢复默认
            self.verify_click_pt = None  # ✅ 清空点击记录
            self.log("验证模式已关闭。")

        self.render_current_frame()  # ✅ 立即刷新画面
    # --- 新增：处理点击选点 ---
    def on_label_clicked(self, widget_x, widget_y):
        """
        [修复版] 相对移动逻辑：
        无论机器在哪，计算点击点与屏幕中心的【相对距离】，控制机器去追。
        """
        if self.last_frame is None: return

        # 1. 坐标换算
        h_img, w_img = self.last_frame.shape[:2]
        h_widget = self.label.height()
        w_widget = self.label.width()
        if h_widget == 0 or w_widget == 0: return

        click_img_x = widget_x * (w_img / w_widget)
        click_img_y = widget_y * (h_img / h_widget)

        # ====== 情况 A：标定微调 (保持不变) ======
        if self.calib_step == 10:
            min_dist = float('inf')
            best_idx = -1
            for i, pt in enumerate(self.calib_vision_pts):
                px, py = pt[0], pt[1]
                dist = ((px - click_img_x)**2 + (py - click_img_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if best_idx != -1:
                self.calib_sel_idx = best_idx
                self.render_current_frame()
            return

        # ====== 情况 B：验证模式 (核心修改) ======
        if self.verify_mode and self.cam2mach_matrix is not None:
            # 1. 记录点击点 (画图用)
            self.verify_click_pt = (click_img_x, click_img_y)
            self.render_current_frame()

            # 2. 计算【相对像素偏移】 (点击点 - 屏幕中心)
            center_x, center_y = w_img / 2, h_img / 2
            dx_pix = click_img_x - center_x
            dy_pix = click_img_y - center_y

            # 3. 利用矩阵的线性部分 (2x2) 计算【相对毫米偏移】
            # 矩阵 M = [[m00, m01, tx], [m10, m11, ty]]
            # dx_mm = m00*dx_pix + m01*dy_pix
            # dy_mm = m10*dx_pix + m11*dy_pix
            m = self.cam2mach_matrix
            dx_mm = m[0, 0] * dx_pix + m[0, 1] * dy_pix
            dy_mm = m[1, 0] * dx_pix + m[1, 1] * dy_pix

            self.log(f"🎯 [验证] 像素偏差({dx_pix:.1f}, {dy_pix:.1f}) -> 机器需移动({dx_mm:.2f}, {dy_mm:.2f})")

            # 4. 发送相对移动指令 (G91)
            # 速度稍慢一点方便观察，例如 3000
            speed = 3000
            script = f"G91\nG0 X{dx_mm:.3f} Y{dy_mm:.3f} F{speed}\nG90" # 动完切回绝对模式
            self.machine._post_gcode(script)

    def keyPressEvent(self, event):
        """键盘事件：WASD 微调，1234 切换选中点"""
        if self.calib_step == 10:
            key = event.key()
            step = 1.0  # 微调步长 1px

            # 切换选中点
            if key == Qt.Key_1:
                self.calib_sel_idx = 0
            elif key == Qt.Key_2:
                self.calib_sel_idx = 1
            elif key == Qt.Key_3:
                self.calib_sel_idx = 2
            elif key == Qt.Key_4:
                self.calib_sel_idx = 3

            # 微调坐标
            elif key == Qt.Key_W:
                self.calib_vision_pts[self.calib_sel_idx][1] -= step
            elif key == Qt.Key_S:
                self.calib_vision_pts[self.calib_sel_idx][1] += step
            elif key == Qt.Key_A:
                self.calib_vision_pts[self.calib_sel_idx][0] -= step
            elif key == Qt.Key_D:
                self.calib_vision_pts[self.calib_sel_idx][0] += step

            # 触发重绘
            self.render_current_frame()
            # 必须调用 accept，否则可能传给其他控件
            event.accept()
        else:
            super().keyPressEvent(event)

    def finish_calibration(self):
        """计算最终矩阵"""
        try:
            pts_img = np.array(self.calib_vision_pts, dtype=np.float32)
            pts_mach = np.array(self.calib_machine_pts, dtype=np.float32)
            M, _ = cv2.estimateAffine2D(pts_img, pts_mach)

            if M is None: raise Exception("点共线或数据异常")

            self.cam2mach_matrix = M
            np.save("cam2mach_matrix.npy", M)
            self.log("✅ 标定完成！矩阵已保存。")
            QMessageBox.information(self, "成功", "坐标系标定完成！")
        except Exception as e:
            self.log(f"❌ 计算失败: {e}")

        self.calib_step = 0
        self.show_crosshair = False
        self.btn_calibrate.setText("标定坐标系")
        self.btn_calibrate.setStyleSheet("")

    def on_virtual_run(self):
        """
        按钮：相机虚拟跑
        思路：在当前画面上，把4个产品的点胶路径按顺序“画一遍”，不动机器，只用于视觉检查。
        当前先占位。
        """
        if self.last_frame is None:
            self.log("❌ 当前没有画面，先启动相机或载入示例图再虚拟跑。")
            return
        self.log("👁️ 虚拟跑：后续会在画面上动态显示点胶路径（当前为占位）。")

    def on_dry_run(self):
        """
        按钮：空跑 (流程升级版)
        1. 识别并计算路径。
        2. 屏幕绘制绿线（视觉检查）。
        3. 弹出参数设置窗口 (DryRunDialog)。
        4. 用户确认后，后台发送指令。
        """
        # 1. 基础检查
        if self.cam2mach_matrix is None:
            QMessageBox.warning(self, "未标定", "请先完成【标定坐标系】！")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "无图像", "请先启动相机！")
            return

        # 暂停实时识别，防止干扰
        if self.live_detect:
            self.detect_mold()

        try:
            # --- A. 图像处理 & 识别 ---
            from core.mold_detector import get_calculated_paths
            img_dry = self.last_frame.copy()
            img_dry = cv2.flip(img_dry, -1)  # 翻转

            paths_px = get_calculated_paths(img_dry)
            if not paths_px:
                QMessageBox.warning(self, "未识别", "未识别到模具路径！")
                return

            # --- B. 视觉反馈：立即画出绿线 ---
            self.dry_run_paths_to_draw = paths_px
            self.show_crosshair = True
            self.render_current_frame()

            # --- C. 弹出参数对话框 ---
            h_img, w_img = img_dry.shape[:2]
            center_x, center_y = w_img / 2, h_img / 2
            
            # 准备初始参数
            initial_params = {
                'safe_z_cavity': float(self.safe_z_cavity),
                'safe_z_product': float(self.safe_z_product),
                'work_z': float(self.work_z),
                'speed_xy': int(self.feedrate_xy),
                'speed_z': int(self.feedrate_z)
            }
            
            # 获取当前机器坐标作为原点
            origin_pos = self.machine.get_position()
            nozzle_offset = (self.nozzle_offset_x, self.nozzle_offset_y)

            dialog = DryRunDialog(
                self, 
                paths_px, 
                self.cam2mach_matrix, 
                origin_pos, 
                nozzle_offset, 
                initial_params,
                (center_x, center_y)
            )

            if dialog.exec_() == QDialog.Accepted:
                # --- D. 执行 ---
                full_script = dialog.get_gcode()
                self.log("🚀 开始执行空跑 (逐行模式)...")
                
                # 1. 拆分指令
                lines = full_script.split('\n')
                
                # 2. 创建逐行 Worker
                self.dry_run_worker = GCodeLineWorker(self.machine, lines)
                self.dry_run_worker.finished_signal.connect(self.on_dry_run_finished)
                self.dry_run_worker.log_signal.connect(self.log) # 连接日志
                self.dry_run_worker.start()
                
                # 可选：更新界面上的默认值，方便下次使用
                self.spn_safe_z_cavity.setValue(dialog.safe_z_cavity)
                self.spn_safe_z_product.setValue(dialog.safe_z_product)
                self.spn_work_z.setValue(dialog.work_z)
                self.feedrate_xy = dialog.speed_xy
                self.feedrate_z = dialog.speed_z

                # 3. 弹出控制面板 (传入 worker)
                self.dry_run_ctrl_dlg = DryRunControlDialog(self, self.dry_run_worker)
                self.dry_run_ctrl_dlg.exec_()
            else:
                self.log("🚫 用户取消空跑")
                self.dry_run_paths_to_draw = []  # 清除绿线
                self.render_current_frame()

        except Exception as e:
            self.log(f"❌ 空跑准备出错: {e}")
            import traceback
            traceback.print_exc()

    def on_dry_run_finished(self, success, msg):
        """空跑线程结束后的回调"""
        # 关闭控制面板
        if hasattr(self, 'dry_run_ctrl_dlg') and self.dry_run_ctrl_dlg.isVisible():
            self.dry_run_ctrl_dlg.accept()

        # 清除屏幕上的绿线
        self.dry_run_paths_to_draw = []
        self.render_current_frame()

        if success:
            QMessageBox.information(self, "完成", "✅ 空跑任务执行完毕！")
            self.log("✅ 空跑结束。")
        else:
            QMessageBox.warning(self, "失败", f"❌ 执行失败: {msg}")
            self.log("❌ 空跑异常中止。")
            return


    def on_real_run(self):
        """
        按钮：开始点胶
        """
        if self.cam2mach_matrix is None:
            self.log("⚠ 还没有做坐标标定（cam2mach_matrix 为 None），不能开始真实点胶。")
            return
        self.log("💧 开始点胶：将来会真正控制 XYZ 和点胶阀（当前为占位）。")

    def on_test_valve(self):
        """
        按钮：测试出胶
        将来会通过 MachineClient 触发继电器 / MOS，控制电磁阀 ON/OFF。
        """
        self.log("🔧 测试出胶：后续会接继电器/MOS 控制信号（当前为占位，只打印日志）。")

    # ================== 相机参数联动 ==================
    def sync_from_camera_to_ui(self):
        if not self.cam:
            return

        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("ExposureTime", v)
            val = float(v.fCurValue)
            self.spn_exp.blockSignals(True)
            self.spn_exp.setValue(val)
            self.spn_exp.blockSignals(False)

            pos = self.value_to_slider(val, self.exp_min, self.exp_max)
            self.sld_exp.blockSignals(True)
            self.sld_exp.setValue(pos)
            self.sld_exp.blockSignals(False)
        except:
            pass

        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gain", v)
            val = float(v.fCurValue)
            self.spn_gain.blockSignals(True)
            self.spn_gain.setValue(val)
            self.spn_gain.blockSignals(False)

            pos = self.value_to_slider(val, self.gain_min, self.gain_max)
            self.sld_gain.blockSignals(True)
            self.sld_gain.setValue(pos)
            self.sld_gain.blockSignals(False)
        except:
            pass

        try:
            v = MVCC_FLOATVALUE()
            self.cam.MV_CC_GetFloatValue("Gamma", v)
            val = float(v.fCurValue)
            self.spn_gamma.blockSignals(True)
            self.spn_gamma.setValue(val)
            self.spn_gamma.blockSignals(False)

            pos = self.value_to_slider(val, self.gamma_min, self.gamma_max)
            self.sld_gamma.blockSignals(True)
            self.sld_gamma.setValue(pos)
            self.sld_gamma.blockSignals(False)
        except:
            pass

    @staticmethod
    def slider_to_value(pos, vmin, vmax):
        return vmin + (vmax - vmin) * (pos / 100.0)

    @staticmethod
    def value_to_slider(val, vmin, vmax):
        if vmax <= vmin:
            return 0
        return int(100 * (val - vmin) / (vmax - vmin))

    def on_exp_slider_changed(self, pos):
        if not self.cam:
            return
        val = self.slider_to_value(pos, self.exp_min, self.exp_max)
        self.set_exposure(val, update_slider=False)

    def on_gain_slider_changed(self, pos):
        if not self.cam:
            return
        val = self.slider_to_value(pos, self.gain_min, self.gain_max)
        self.set_gain(val, update_slider=False)

    def on_gamma_slider_changed(self, pos):
        if not self.cam:
            return
        val = self.slider_to_value(pos, self.gamma_min, self.gamma_max)
        self.set_gamma(val, update_slider=False)

    def on_exp_spin_changed(self, val):
        if not self.cam:
            return
        self.set_exposure(float(val), update_spin=False)

    def on_gain_spin_changed(self, val):
        if not self.cam:
            return
        self.set_gain(float(val), update_spin=False)

    def on_gamma_spin_changed(self, val):
        if not self.cam:
            return
        self.set_gamma(float(val), update_spin=False)

    def set_exposure(self, val, update_slider=True, update_spin=True):
        val = max(self.exp_min, min(self.exp_max, val))
        self.cam.MV_CC_SetFloatValue("ExposureTime", float(val))

        if update_spin:
            self.spn_exp.blockSignals(True)
            self.spn_exp.setValue(val)
            self.spn_exp.blockSignals(False)

        if update_slider:
            pos = self.value_to_slider(val, self.exp_min, self.exp_max)
            self.sld_exp.blockSignals(True)
            self.sld_exp.setValue(pos)
            self.sld_exp.blockSignals(False)

        self.save_params()

    def set_gain(self, val, update_slider=True, update_spin=True):
        val = max(self.gain_min, min(self.gain_max, val))
        self.cam.MV_CC_SetFloatValue("Gain", float(val))

        if update_spin:
            self.spn_gain.blockSignals(True)
            self.spn_gain.setValue(val)
            self.spn_gain.blockSignals(False)

        if update_slider:
            pos = self.value_to_slider(val, self.gain_min, self.gain_max)
            self.sld_gain.blockSignals(True)
            self.sld_gain.setValue(pos)
            self.sld_gain.blockSignals(False)

        self.save_params()

    def set_gamma(self, val, update_slider=True, update_spin=True):
        val = max(self.gamma_min, min(self.gamma_max, val))
        try:
            self.cam.MV_CC_SetFloatValue("Gamma", float(val))
        except:
            pass

        if update_spin:
            self.spn_gamma.blockSignals(True)
            self.spn_gamma.setValue(val)
            self.spn_gamma.blockSignals(False)

        if update_slider:
            pos = self.value_to_slider(val, self.gamma_min, self.gamma_max)
            self.sld_gamma.blockSignals(True)
            self.sld_gamma.setValue(pos)
            self.sld_gamma.blockSignals(False)

        self.save_params()

    # ================== 其它 ==================
    def resume_preview(self):
        self.freeze_display = False

    def on_speed_changed(self):
        """更新 XY 和 Z 的速度变量"""
        self.feedrate_xy = float(self.sld_spd_xy.value())
        self.feedrate_z = float(self.sld_spd_z.value())

        # 更新标签文字
        self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")
        self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

        # 【新增】调节后立即保存
        self.save_app_settings()

    def on_accel_changed(self):
        """更新加速度，并直接发送 SET_VELOCITY_LIMIT 指令给 Klipper"""
        val = self.sld_acc.value()
        self.accel = float(val)
        self.lbl_acc.setText(f"加速度: {val}")

        # 发送 Klipper 指令
        if self.machine.connected:
            # 限制加速度和减速度
            script = f"SET_VELOCITY_LIMIT ACCEL={val} ACCEL_TO_DECEL={val / 2}"
            self.machine._post_gcode(script)
            self.log(f"⚡ 加速度已调整为: {val}")

        # 【新增】调节后立即保存
        self.save_app_settings()

    def on_btn_auto_tool_set(self):
        """一键对刀逻辑 (含手动找点模式)"""
        
        # --- 情况 A: 正在手动找点模式，用户点击按钮表示“确认位置” ---
        if self.manual_tool_set_mode:
            # 获取当前坐标
            mx, my, mz = self.machine.get_position()
            self.tool_setter_x = mx
            self.tool_setter_y = my
            
            # 保存设置
            self.save_app_settings()
            
            # 退出手动模式
            self.manual_tool_set_mode = False
            self.btn_auto_tool_set.setText("一键对刀")
            self.btn_auto_tool_set.setStyleSheet("background-color: #2563EB; font-weight: bold;")
            
            QMessageBox.information(self, "位置已保存", 
                                    f"对刀仪坐标已保存为:\nX: {mx:.2f}\nY: {my:.2f}\n\n现在可以执行自动对刀了。")
            return

        # --- 情况 B: 尚未设置对刀仪坐标 ---
        if self.tool_setter_x is None or self.tool_setter_y is None:
            reply = QMessageBox.question(
                self, "未设置对刀仪",
                "尚未记录对刀仪的 XY 坐标。\n\n"
                "是否进入【手动找点模式】？\n"
                "进入后，请移动针头到对刀仪正上方，然后再次点击按钮确认。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.manual_tool_set_mode = True
                self.btn_auto_tool_set.setText("确认对刀仪位置 (点击保存)")
                self.btn_auto_tool_set.setStyleSheet("background-color: #F59E0B; color: black; font-weight: bold;") # 橙色
                self.log("👉 进入手动找点模式：请移动到对刀仪上方，并观察状态灯。")
            return

        # --- 情况 C: 已有坐标，执行自动对刀 ---
        # 修改：允许用户选择“重新标定位置”
        box = QMessageBox(self)
        box.setWindowTitle("确认对刀")
        box.setText(f"已保存对刀仪坐标 (X{self.tool_setter_x:.1f}, Y{self.tool_setter_y:.1f})。\n\n"
                    "请选择操作：")
        btn_auto = box.addButton("开始自动对刀", QMessageBox.YesRole)
        btn_manual = box.addButton("重新标定位置", QMessageBox.NoRole)
        btn_cancel = box.addButton("取消", QMessageBox.RejectRole)
        
        box.exec_()
        
        if box.clickedButton() == btn_manual:
            # 进入手动模式
            self.manual_tool_set_mode = True
            self.btn_auto_tool_set.setText("确认对刀仪位置 (点击保存)")
            self.btn_auto_tool_set.setStyleSheet("background-color: #F59E0B; color: black; font-weight: bold;") 
            self.log("👉 进入手动找点模式：请移动到对刀仪上方，并观察状态灯。")
            return
            
        elif box.clickedButton() == btn_auto:
            # 继续执行自动对刀
            pass
        else:
            # 取消
            return

        self.log("🚀 开始自动对刀...")
        self.btn_auto_tool_set.setEnabled(False)
        
        # 启动后台线程
        self.tool_worker = AutoToolSetWorker(self.machine, self.safe_z_product, self.tool_setter_x, self.tool_setter_y)
        self.tool_worker.finished_signal.connect(self.on_auto_tool_set_finished)
        self.tool_worker.start()

    def on_auto_tool_set_finished(self, success, trigger_z, msg):
        self.btn_auto_tool_set.setEnabled(True)
        self.log(msg)
        
        if success:
            # 计算偏差并更新
            if self.z_base == 0.0:
                # 第一次对刀，直接记录
                self.z_base = trigger_z
                self.log(f"✅ 首次对刀，基准 Z 已设为: {self.z_base:.3f}")
            else:
                # 计算差值
                diff = trigger_z - self.z_base
                self.log(f"ℹ️ 旧基准: {self.z_base:.3f}, 新基准: {trigger_z:.3f}, 偏差: {diff:.3f}")
                
                # 更新 Z 参数
                self.safe_z_cavity += diff
                self.safe_z_product += diff
                self.work_z += diff
                
                # 更新 UI
                self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
                self.spn_safe_z_product.setValue(self.safe_z_product)
                self.spn_work_z.setValue(self.work_z)
                
                self.z_base = trigger_z
                self.log(f"✅ 所有 Z 高度已自动修正 (Offset: {diff:.3f})")

            # 更新 Label
            self.lbl_z_base.setText(f"当前 Z 基准: {self.z_base:.2f} mm")
            
            # 保存设置
            self.save_app_settings()

    # ================== 其它 ==================
    def closeEvent(self, e):
        # 停止轮询线程
        if hasattr(self, 'poll_worker'):
            self.poll_worker.stop()

        self.stop_cam()
        # 【新增】关闭前保存所有参数
        self.save_app_settings()
        e.accept()

    def on_z_params_changed(self):
        """安全高度 / 点胶高度 改变时更新变量并保存"""
        self.safe_z_cavity = float(self.spn_safe_z_cavity.value())
        self.safe_z_product = float(self.spn_safe_z_product.value())
        self.work_z = float(self.spn_work_z.value())
        self.save_app_settings()

if __name__ == "__main__":
    print("🚀 App Starting...")
    try:
        app = QApplication(sys.argv)
        w = MainWindow()
        w.resize(1600, 950)
        w.show()
        print("✅ App Window Shown. Entering Event Loop...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
