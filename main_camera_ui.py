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

# ====== Attempt to silence OpenCV internal logs via API ======
# try:
#     cv2.utils.logging.setLogLevel(0) # 0 = Silent
# except AttributeError:
#     pass
# =========================================================

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
    QMessageBox, QLineEdit, QScrollArea,QDialog, QDialogButtonBox,
    QCheckBox, QTabWidget, QSplitter
)




from PyQt5.QtGui import QImage, QPixmap

# 你的检测函数
from core.mold_detector import detect_mold_and_cells, reload_golden_templates
from core.dry_run_dialog import DryRunDialog, DryRunControlDialog
from core.batch_run_dialog import BatchRunDialog

from core.machine_client import MachineClient
from core.config_manager import ConfigManager
from core.camera_manager import CameraManager
from core.halcon_manager import HalconManager
from core.visualizer_3d import Visualizer3D

# MvImport dependency removed from UI (managed by CameraManager)
# from MvImport.MvCameraControl_class import *

# ================== 线程：相机取流 (已移除，使用 CameraManager) ==================
# class CameraThread(QThread): ... removed ...

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

# ================== 线程：USB相机取流 (MJPEG/HTTP) ==================
class USBCameraWorker(QThread):
    frame_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(bool) # True=Connected, False=Disconnected

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True
        self.cap = None

    def run(self):
        print(f"[USBCamera] Connecting to {self.url}...")
        self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print(f"[USBCamera] Failed to open URL: {self.url}")
            self.status_signal.emit(False)
            return
        
        self.status_signal.emit(True)
        print(f"[USBCamera] Connected.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Stream ended or error
                print("[USBCamera] Stream disconnected.")
                self.status_signal.emit(False)
                time.sleep(1) # Retry delay
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url) # Reconnect attempt
                continue

            # Process frame
            if frame is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    # QImage.Format_RGB888
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.frame_signal.emit(qimg)
                    
                    # Debug: print dot every 10 frames to avoid spam but confirm liveness
                    # self.frame_count = getattr(self, 'frame_count', 0) + 1
                    # if self.frame_count % 10 == 0:
                    #     sys.stdout.write('.')
                    #     sys.stdout.flush()
                except Exception as e:
                     print(f"[USBCamera] Error processing frame: {e}")
            
            # Limit FPS
            time.sleep(0.01)

        if self.cap:
            self.cap.release()
        print("[USBCamera] Stopped.")

    def stop(self):
        self.running = False
        self.wait()

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
        self.setWindowTitle("DIY多色视觉点胶机控制台")
        self.resize(1280, 800)
        
        # ====== ✅ Initialize Managers ======
        self.config = ConfigManager()
        self.camera_manager = CameraManager()
        self.halcon_manager = HalconManager()
        self.camera_manager.get_frame_signal().connect(self.update_image) if self.camera_manager.get_frame_signal() else None
        # Load Config
        # self.config = ConfigManager() # This line is duplicated and should be removed or moved.
        # self.load_app_settings()  # Moved after UI init

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
        # 新增：缩放比例
        self.zoom_scale = 1.0
        
        self.detect_skip_count = 0        # 新增：跳帧计数器

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
        self.calib_step_size = 1.0  # ✅ 新增：标定微调步长
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
        self.nozzle_offset_z = 0.0

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
        self.safe_z = None          # 抬刀 / 快移高度 (Legacy)
        self.safe_z_cavity = None   # 腔内安全高度 (小抬刀)
        self.safe_z_product = None  # 产品安全高度 (大抬刀)
        self.work_z = None          # 点胶 / 下刀高度 (接近0)
        
        # ====== ✅ 新增：相机空跑专用高度参数 ======
        self.cam_safe_z_cavity = None
        self.cam_safe_z_product = None
        self.cam_work_z = None # 相机通常比针头高，避免撞到
        # =======================================
        
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

        # USB Camera Worker
        self.usb_cam_worker = None

        # ==========================================
        # 在所有 UI 元素初始化完成后，调用加载设置
        self.init_ui()
        self.load_app_settings()
        self.load_saved_params()
        self.load_golden_templates()
        self.init_calibration()
        self.init_done = True

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
        print(f"[LOG] {msg}") # Enable console logging
        self.log_signal.emit(str(msg))

    # ================== Tool Set Group (New) ==================
    def init_tool_set_group(self):
        # ===== Z 高度 / 对刀 (独立组) =====
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
        
        return zbase_group

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
        # ---- Klipper 连接 ----
        conn_group = QGroupBox("Klipper 连接")
        #改为垂直布局，容纳两行
        conn_final_layout = QVBoxLayout() 

        # === 第一行：IP, Port, Buttons ===
        row1_layout = QHBoxLayout()
        
        # 1. IP
        lbl_ip = QLabel("IP:")
        self.edit_ip = QLineEdit("192.168.60.26")
        self.edit_ip.setFixedWidth(110)
        self.edit_ip.setAlignment(Qt.AlignCenter)

        # 2. Port
        lbl_port = QLabel("Port:")
        self.edit_port = QLineEdit("7125")
        self.edit_port.setFixedWidth(50)
        self.edit_port.setAlignment(Qt.AlignCenter)
        
        # 3. 按钮定义
        self.btn_connect_klipper = QPushButton("连接")
        self.btn_connect_klipper.setCursor(Qt.PointingHandCursor)
        self.btn_connect_klipper.setMinimumHeight(28)
        self.btn_connect_klipper.setFixedWidth(80)  
        self.btn_connect_klipper.setStyleSheet("""
                    QPushButton {
                        background-color: #6B7280;
                        color: white;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #4B5563;
                    }
                """)

        self.btn_estop = QPushButton("STOP")
        self.btn_estop.setCursor(Qt.PointingHandCursor)
        self.btn_estop.setMinimumHeight(28)
        self.btn_estop.setFixedWidth(80) 
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
        
        self.btn_start = QPushButton("启动相机")
        self.btn_stop = QPushButton("停止相机")
        self.btn_stop.setEnabled(False)
        for btn in [self.btn_start, self.btn_stop]:
            btn.setMinimumHeight(28)
            btn.setFixedWidth(80)

        # ====== Row 1 排列 ======
        row1_layout.addWidget(lbl_ip)
        row1_layout.addWidget(self.edit_ip)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(lbl_port)
        row1_layout.addWidget(self.edit_port)
        row1_layout.addStretch() 
        
        row1_layout.addWidget(self.btn_start)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_stop)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(self.btn_connect_klipper)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_estop)

        # === 第二行：USB URL ===
        row2_layout = QHBoxLayout()
        lbl_usb = QLabel("USB Stream:")
        self.edit_usb_url = QLineEdit("http://192.168.60.26/webcam/?action=stream") 
        self.edit_usb_url.setPlaceholderText("http://<ip>:8080/stream or /webcam/?action=stream")
        # self.edit_usb_url.setFixedWidth(250) # 让它自适应宽度

        row2_layout.addWidget(lbl_usb)
        row2_layout.addWidget(self.edit_usb_url)
        
        conn_final_layout.addLayout(row1_layout)
        conn_final_layout.addLayout(row2_layout)

        conn_group.setLayout(conn_final_layout)





        # ---- 控制按钮 ----

        self.btn_save = QPushButton("保存图片")
        self.btn_detect = QPushButton("识别模具")
        self.btn_golden = QPushButton("黄金示教")
        self.btn_resume = QPushButton("恢复预览")
        self.btn_path = QPushButton("编辑点胶路径")

        # 点胶任务相关按钮
        self.btn_calibrate = QPushButton("标定坐标系")
        self.btn_calibrate.setFixedWidth(120) # 固定宽度防止文字变化导致界面抖动
        # ====== ✅ 新增：验证精度按钮 ======
        self.btn_verify = QPushButton("验证精度")
        self.btn_verify.setToolTip("点击开启/关闭【点击定位】功能")
        # ================================
        self.btn_virtual = QPushButton("相机空跑")
        self.btn_dry_run = QPushButton("针头空跑")
        self.btn_real_run = QPushButton("开始点胶")
        self.btn_batch_run = QPushButton("批量运行")
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
        task_layout.addWidget(self.btn_batch_run)
        task_layout.addWidget(self.btn_test_valve)

        # ================== 2. 相机参数组 (重构) ==================
        param_group = QGroupBox("相机参数")
        param_layout = QGridLayout()

        # 曝光 (Linear Scale 0-10ms, Step 0.1)
        param_layout.addWidget(QLabel("曝光 (ms):"), 0, 0)
        self.sld_exp = QSlider(Qt.Horizontal)
        self.sld_exp.setRange(0, 100) # 0.0 - 10.0 ms
        self.sld_exp.valueChanged.connect(self.on_exp_slider_changed)
        param_layout.addWidget(self.sld_exp, 0, 1)

        self.spn_exp = QDoubleSpinBox()
        self.spn_exp.setRange(0.0, 10.0)
        self.spn_exp.setDecimals(1)
        self.spn_exp.setSuffix(" ms")
        self.spn_exp.valueChanged.connect(self.on_exp_spin_changed)
        param_layout.addWidget(self.spn_exp, 0, 2)

        # 增益 (0-10 dB, Step 0.1)
        param_layout.addWidget(QLabel("增益 (dB):"), 1, 0)
        self.sld_gain = QSlider(Qt.Horizontal)
        self.sld_gain.setRange(0, 100) # 0.0 - 10.0 dB
        self.sld_gain.valueChanged.connect(self.on_gain_slider_changed)
        param_layout.addWidget(self.sld_gain, 1, 1)

        self.spn_gain = QDoubleSpinBox()
        self.spn_gain.setRange(0.0, 10.0)
        self.spn_gain.setDecimals(1)
        self.spn_gain.setSuffix(" dB")
        self.spn_gain.valueChanged.connect(self.on_gain_spin_changed)
        param_layout.addWidget(self.spn_gain, 1, 2)

        # Gamma (0.1-4.0, Step 0.1) -- Reference Implementation
        param_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.sld_gamma = QSlider(Qt.Horizontal)
        self.sld_gamma.setRange(1, 40) # 0.1 - 4.0
        self.sld_gamma.valueChanged.connect(self.on_gamma_slider_changed)
        param_layout.addWidget(self.sld_gamma, 2, 1)

        self.spn_gamma = QDoubleSpinBox()
        self.spn_gamma.setRange(0.1, 4.0)
        self.spn_gamma.setSingleStep(0.1)
        self.spn_gamma.valueChanged.connect(self.on_gamma_spin_changed)
        param_layout.addWidget(self.spn_gamma, 2, 2)

        param_group.setLayout(param_layout)
        self.param_group = param_group

        # ================== 3. 视觉后端组 (Halcon) ==================
        vision_group = QGroupBox("视觉后端")
        vision_layout = QVBoxLayout()
        
        self.chk_use_halcon = QCheckBox("启用 Halcon 识别")
        self.chk_use_halcon.clicked.connect(self.on_halcon_toggled)
        
        self.btn_load_halcon = QPushButton("加载 Halcon 脚本")
        self.btn_load_halcon.setEnabled(False)
        self.btn_load_halcon.clicked.connect(self.on_load_halcon_script)
        
        vision_layout.addWidget(self.chk_use_halcon)
        vision_layout.addWidget(self.btn_load_halcon)
        vision_group.setLayout(vision_layout)
        self.vision_group = vision_group


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
        self.spn_safe_z_cavity.setValue(self.safe_z_cavity if self.safe_z_cavity is not None else 0.0)

        # 2. 产品安全高度
        lbl_safe_product = QLabel("产品安全Z")
        self.spn_safe_z_product = QDoubleSpinBox()
        self.spn_safe_z_product.setRange(0.0, 300.0)
        self.spn_safe_z_product.setDecimals(2)
        self.spn_safe_z_product.setSingleStep(0.1)
        self.spn_safe_z_product.setValue(self.safe_z_product if self.safe_z_product is not None else 0.0)

        # 3. 点胶高度
        lbl_work = QLabel("点胶Z")
        self.spn_work_z = QDoubleSpinBox()
        self.spn_work_z.setRange(0.0, 300.0)
        self.spn_work_z.setDecimals(2)
        self.spn_work_z.setSingleStep(0.1)
        self.spn_work_z.setValue(self.work_z if self.work_z is not None else 0.0)

        zparam_layout.addWidget(lbl_safe_cavity)
        zparam_layout.addWidget(self.spn_safe_z_cavity)
        zparam_layout.addWidget(lbl_safe_product)
        zparam_layout.addWidget(self.spn_safe_z_product)
        zparam_layout.addWidget(lbl_work)
        zparam_layout.addWidget(self.spn_work_z)

        motion_layout.addLayout(zparam_layout)
        

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
        self.btn_calib_nozzle.clicked.connect(self.on_calibrate_nozzle_offset)

        offset_layout.addWidget(self.lbl_offset_x)
        offset_layout.addWidget(self.lbl_offset_y)
        offset_layout.addWidget(self.btn_calib_nozzle)
        offset_group.setLayout(offset_layout)
        self.offset_group = offset_group
        
        # New: 初始化独立的对刀组
        self.tool_set_group = self.init_tool_set_group()


        # ---- 日志窗口 ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)

        # ================== 右侧：单栏垂直布局 (宽版) ==================
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(info_group)
        right_layout.addWidget(conn_group)
        right_layout.addWidget(self.motion_group)
        right_layout.addWidget(self.tool_set_group) # [NEW] 放在这里 (XYZ 下面)
        
        right_layout.addWidget(self.offset_group)
        right_layout.addLayout(btn_layout)   # 相机/示教按钮
        right_layout.addLayout(task_layout)  # 点胶任务按钮
        right_layout.addWidget(self.param_group)
        right_layout.addWidget(self.vision_group)
        
        # 日志窗口
        right_layout.addWidget(self.log_box)
        
        # 弹簧 (确保上方紧凑)
        right_layout.addStretch()

        # ---- 右侧容器 ----
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        # right_widget.setFixedWidth(500) # 移除固定宽度，交给 Splitter 管理

        # ================== 双栏布局核心 (可拖拽分割) ==================
        # [左栏: 相机(上) + 3D(下)] - [右栏: 固定控制面板]

        # 1. 3D 视图 (缩放版 - 左下)
        self.vis3d = Visualizer3D()
        self.vis3d.setMinimumHeight(200) 
        vis_group = QGroupBox("3D 针头姿态")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.vis3d)
        vis_group.setLayout(vis_layout)

        # 2. USB 相机视图 (新增 - 右下)
        usb_group = QGroupBox("USB 针头实景")
        usb_layout = QVBoxLayout()
        
        # Controls (Flip)
        usb_ctrl_layout = QHBoxLayout()
        usb_ctrl_layout.addStretch()
        self.chk_usb_flip_h = QCheckBox("水平翻转")
        self.chk_usb_flip_h.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        self.chk_usb_flip_h.setChecked(True) # 默认开启
        
        self.chk_usb_flip = QCheckBox("垂直翻转")
        self.chk_usb_flip.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        self.chk_usb_flip.setChecked(True) # 默认开启
        
        usb_ctrl_layout.addWidget(self.chk_usb_flip_h)
        usb_ctrl_layout.addWidget(self.chk_usb_flip)
        
        self.label_usb_cam = ImageLabel("Waiting for Stream...")
        self.label_usb_cam.setMinimumSize(320, 240)
        self.label_usb_cam.setStyleSheet("""
            background-color: #000000;
            color: #6B7280;
            border-radius: 4px;
        """)
        self.label_usb_cam.setAlignment(Qt.AlignCenter)
        
        usb_layout.addLayout(usb_ctrl_layout)
        usb_layout.addWidget(self.label_usb_cam)
        usb_group.setLayout(usb_layout)

        # 3. 底部水平分割器
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(vis_group)
        bottom_splitter.addWidget(usb_group)
        
        # 强制 50:50 (假设宽度足够，各给 600px 期望值)
        bottom_splitter.setSizes([400, 400]) 
        bottom_splitter.setCollapsible(0, False)
        bottom_splitter.setCollapsible(1, False)

        # setStretchFactor 只是辅助，setSizes 更直接
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        # 2. 左侧总布局 (垂直: 相机 + (3D|USB))
        # 必须包装成 Widget 才能放入 Splitter
        left_widget = QWidget()
        left_final_layout = QVBoxLayout()
        
        camera_widget = QWidget()
        camera_widget.setLayout(left_layout)
        
        left_final_layout.addWidget(camera_widget, stretch=2) # 相机占 2/3
        left_final_layout.addWidget(bottom_splitter, stretch=1) # 底部占 1/3
        
        left_widget.setLayout(left_final_layout)

        # 3. 主布局 (使用 Splitter)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # 设置初始比例 (左侧自适应, 右侧 600px)
        splitter.setSizes([1280-600, 600]) 
        splitter.setCollapsible(0, False) 
        
        # 让左侧(0)自动拉伸，右侧(1)尽量保持宽度
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        
        # Fix for QWidget inheritance
        self.setLayout(main_layout)
        
        # 设定窗口初始大小
        self.resize(1280, 800)

        # 初始化一次坐标显示 & 针头偏移显示
        self.update_xyz_display()
        self.update_nozzle_offset_display()

    # ================== Slider/SpinBox Synchronization ==================
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
        self.btn_batch_run.clicked.connect(self.on_batch_run)
        self.btn_test_valve.clicked.connect(self.on_test_valve)

        # 相机参数联动
        # Slider signals removed as sliders were removed.
        # SpinBox signals are already connected in definition to direct camera manager methods.


        # XYZ 点动按钮
        # XYZ 点动按钮
        self.btn_x_minus.clicked.connect(lambda: self.on_jog('x', -1))
        self.btn_x_plus.clicked.connect(lambda: self.on_jog('x', +1))
        self.btn_y_minus.clicked.connect(lambda: self.on_jog('y', -1))
        self.btn_y_plus.clicked.connect(lambda: self.on_jog('y', +1))
        self.btn_z_minus.clicked.connect(lambda: self.on_jog('z', -1))
        self.btn_z_plus.clicked.connect(lambda: self.on_jog('z', +1))

        # --- Vision & Task Buttons ---
        # 恢复用户之前的逻辑，不重复连接
        # self.btn_golden.clicked.connect(self.on_golden_teach)
        # self.btn_path.clicked.connect(self.on_edit_path)
        # self.btn_detect.clicked.connect(self.detect_mold)
        # self.btn_resume.clicked.connect(self.resume_preview)
        
        # self.btn_calibrate.clicked.connect(self.on_calibrate_coords)
        # self.btn_verify.clicked.connect(self.on_toggle_verify_mode)
        
        # self.btn_virtual.clicked.connect(self.on_virtual_run)
        # self.btn_dry_run.clicked.connect(self.on_dry_run)
        # self.btn_real_run.clicked.connect(self.on_real_run)
        # self.btn_batch_run.clicked.connect(self.on_batch_run)
        # self.btn_test_valve.clicked.connect(self.on_test_valve)
        # self.btn_save.clicked.connect(self.save_image)
        
        # 归位按钮
        self.btn_home_all.clicked.connect(lambda: self.on_home_axis('all'))
        self.btn_home_x.clicked.connect(lambda: self.on_home_axis('x'))
        self.btn_home_y.clicked.connect(lambda: self.on_home_axis('y'))
        self.btn_home_z.clicked.connect(lambda: self.on_home_axis('z'))

        # 初始化一次坐标显示 & 针头偏移显示
        self.update_xyz_display()
        self.update_nozzle_offset_display()

    # ================== Slider/SpinBox Synchronization ==================
    # ================== Slider/SpinBox Synchronization (Simple Linear like Gamma) ==================
    def on_exp_slider_changed(self, val):
        # Slider: 0 - 100 (0.1ms per step)
        # Range: 0.0 - 10.0 ms
        
        float_val = val / 10.0 # Same as Gamma
        
        self.spn_exp.blockSignals(True)
        self.spn_exp.setValue(float_val)
        self.spn_exp.blockSignals(False)
        
        # Camera takes microseconds
        self.camera_manager.set_exposure(float_val * 1000.0)

    def on_exp_spin_changed(self, val):
        # val is ms (0-10.0)
        
        self.camera_manager.set_exposure(val * 1000.0)
        
        # Slider: 0 - 100
        slider_val = int(val * 10.0) # Same as Gamma
        
        self.sld_exp.blockSignals(True)
        self.sld_exp.setValue(slider_val)
        self.sld_exp.blockSignals(False)

    def on_gain_slider_changed(self, val):
        # Slider: 0 - 100
        # Range: 0.0 - 10.0 dB
        
        float_val = val / 10.0 # Same as Gamma
        
        self.spn_gain.blockSignals(True)
        self.spn_gain.setValue(float_val)
        self.spn_gain.blockSignals(False)
        
        self.camera_manager.set_gain(float_val)

    def on_gain_spin_changed(self, val):
        self.camera_manager.set_gain(val)
        
        slider_val = int(val * 10.0) # Same as Gamma
        
        self.sld_gain.blockSignals(True)
        self.sld_gain.setValue(slider_val)
        self.sld_gain.blockSignals(False)

    def on_gamma_slider_changed(self, val):
        # Slider: 1 - 40 (0.1 per step)
        # Range: 0.1 - 4.0
        
        gamma_val = val / 10.0
        
        self.spn_gamma.blockSignals(True)
        self.spn_gamma.setValue(gamma_val)
        self.spn_gamma.blockSignals(False)
        
        self.camera_manager.set_gamma(gamma_val)

    def on_gamma_spin_changed(self, val):
        self.camera_manager.set_gamma(val)
        
        slider_val = int(val * 10.0)
        
        self.sld_gamma.blockSignals(True)
        self.sld_gamma.setValue(slider_val)
        self.sld_gamma.blockSignals(False)

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
        更新界面上的坐标显示，并同步更新 3D 视图。
        """
        if x is not None: self.pos_x = float(x)
        if y is not None: self.pos_y = float(y)
        if z is not None: self.pos_z = float(z)

        # 界面显示
        self.lbl_pos_x.setText(f"X: {self.pos_x:.2f}")
        self.lbl_pos_y.setText(f"Y: {self.pos_y:.2f}")
        self.lbl_pos_z.setText(f"Z: {self.pos_z:.2f}")
        
        # Z轴基准日志显示 (保留原有逻辑)
        if hasattr(self, "lbl_z_ref"):
             self.lbl_z_ref.setText(f"当前 Z 基准: {self.pos_z:.2f} mm")

        # 更新 3D 视图 (针头位置 = 机器坐标 + 针头偏移)
        if hasattr(self, 'vis3d'):
             real_x = self.pos_x + self.nozzle_offset_x
             real_y = self.pos_y + self.nozzle_offset_y
             real_z = self.pos_z + self.nozzle_offset_z
             self.vis3d.update_position(real_x, real_y, real_z)
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
    # ---- 一键对刀（占位） ----
    def on_tool_probe_clicked(self):
        """
        一键对刀入口（当前是占位实现，后续对接 Klipper 对刀宏）。
        """
        QMessageBox.information(
            self,
            "一键对刀（占位）",
            "对刀仪一键对刀功能尚未与 Klipper 宏联动，\\n"
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
        标定针头偏移：使用非模态对话框，允许用户同时操作 XYZ 移动。
        """
        # 如果对话框已经存在且可见，就置顶
        if hasattr(self, 'nozzle_calib_dlg') and self.nozzle_calib_dlg.isVisible():
            self.nozzle_calib_dlg.raise_()
            self.nozzle_calib_dlg.activateWindow()
            return

        # 1. 启动标定
        self.log("▶ 启动针头偏移标定...")
        self.show_crosshair = True # 显示十字
        
        # 2. 创建并显示对话框
        from core.nozzle_calibration_dialog import NozzleCalibrationDialog
        self.nozzle_calib_dlg = NozzleCalibrationDialog(self, self.machine)
        self.nozzle_calib_dlg.calibration_finished.connect(self.on_nozzle_calib_finished)
        self.nozzle_calib_dlg.finished.connect(self.on_nozzle_calib_closed) # 无论如何关闭都触发
        self.nozzle_calib_dlg.show()
        
    def on_nozzle_calib_finished(self, dx, dy):
        """标定完成的回调"""
        self.nozzle_offset_x = dx
        self.nozzle_offset_y = dy
        self.update_nozzle_offset_display()
        self.save_app_settings()
        
        self.log(f"✅ 针头偏移标定完成并保存！")
        self.log(f"   ΔX: {dx:.3f} mm")
        self.log(f"   ΔY: {dy:.3f} mm")
        
    def on_nozzle_calib_closed(self, result):
        """对话框关闭时（无论是保存还是取消），关闭十字准星"""
        self.show_crosshair = False
        self.log("⏹ 针头标定结束。")

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
        if not self.camera_manager.open_camera():
            self.log("❌ No camera found or failed to open")
            return

        # Read ranges from manager but CLAMP them to reasonable UI limits
        # 用户反馈范围太大，这里强制限制 UI 上的调节范围
        cam_min_exp, cam_max_exp = self.camera_manager.exp_range
        cam_min_gain, cam_max_gain = self.camera_manager.gain_range
        # cam_min_gamma, cam_max_gamma = self.camera_manager.gamma_range

        # 曝光: 限制最大 10 ms (UI Range 0-10.0)
        self.exp_min = 0.1
        self.exp_max = 10.0
        
        # 增益: 限制最大 10dB (UI Range 0-10.0)
        self.gain_min = cam_min_gain
        self.gain_max = min(cam_max_gain, 10.0)
        
        # Gamma: 保持 0.1 - 4.0
        self.gamma_min = 0.1
        self.gamma_max = 4.0

        # Update SpinBox ranges
        # CRITICAL: Force SpinBox range to ensure it can accept the full slider range
        self.spn_exp.blockSignals(True)
        self.spn_exp.setRange(0.0, 10.0) # Matches Slider 0-100 / 10
        self.spn_exp.blockSignals(False)
        
        self.spn_gain.blockSignals(True)
        self.spn_gain.setRange(0.0, 10.0) # Matches Slider 0-100 / 10
        self.spn_gain.blockSignals(False)
        
        # Ensure sliders are in sync with current parameter values
        # Force initial sync from current params
        if hasattr(self, 'sync_from_camera_to_ui'):
             self.sync_from_camera_to_ui()

        self.log(f"🔧 相机范围重置: Exp[0-10ms], Gain[0-10dB] (Linear)")

        # Update Info
        info = self.camera_manager.get_cam_info()
        self.lbl_cam_res.setText(f"{info['width']} x {info['height']}")
        self.lbl_cam_fps.setText(f"{info['fps']:.2f} fps")
        self.lbl_cam_fmt.setText(info['pixel_format_str'])
        
        # 1. Start Grabbing
        if not self.camera_manager.start_grabbing():
             self.log("❌ Failed to start grabbing")
             return

        self.sync_from_camera_to_ui()

        # Connect signal (Critical: Connect AFTER start_grabbing where worker is created)
        sig = self.camera_manager.get_frame_signal()
        if sig:
            try: sig.disconnect() 
            except: pass
            sig.connect(self.update_image)
        
        # 2. Apply saved params
        QTimer.singleShot(1000, self.apply_saved_params)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(True)

        # Start USB Camera as well if URL is present (Added)
        # Check if widget exists (init_ui happens later?) No, init_ui is done.
        if hasattr(self, 'edit_usb_url'):
            usb_url = self.edit_usb_url.text().strip()
            if usb_url and (not self.usb_cam_worker or not self.usb_cam_worker.isRunning()):
                 self.usb_cam_worker = USBCameraWorker(usb_url)
                 self.usb_cam_worker.frame_signal.connect(self.update_usb_image)
                 self.usb_cam_worker.status_signal.connect(self.on_usb_status)
                 self.usb_cam_worker.start()


    def stop_cam(self):
        self.camera_manager.close_camera()

        self.label.setText("已停止")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.lbl_cam_res.setText("-")
        self.lbl_cam_fps.setText("-")
        self.lbl_cam_fmt.setText("-")
        
        # Stop USB Camera
        if self.usb_cam_worker:
            self.usb_cam_worker.stop()
            self.usb_cam_worker = None
            if hasattr(self, 'label_usb_cam'):
                self.label_usb_cam.setText("Stopped")


    # read_param_ranges and read_cam_info removed (logic moved to CameraManager)

    def load_saved_params(self):
        # Load from ConfigManager
        params = self.config.cam_params
        
        if "ExposureTime" in params:
            val = float(params["ExposureTime"])
            # Load stored us, display as ms
            ms_val = val / 1000.0
            if hasattr(self, 'spn_exp'):
                self.spn_exp.setValue(ms_val)
        
        if "Gain" in params:
            val = float(params["Gain"])
            if hasattr(self, 'spn_gain'):
                self.spn_gain.setValue(val)
        
        if "Gamma" in params:
            val = float(params["Gamma"])
            if hasattr(self, 'spn_gamma'):
                self.spn_gamma.setValue(val)
                
        self.log(f"✅ 已加载相机参数")

    def save_params(self):
        if self.camera_manager.is_open:
            current = self.camera_manager.get_params()
            self.config.save_cam_params(current)
            self.log(f"✅ 已保存相机参数")

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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            matrix_path = os.path.join(base_dir, "config", "cam2mach_matrix.npy")
            
            self.log(f"DEBUG: Attempting to load matrix from: {matrix_path}")

            if os.path.exists(matrix_path):
                mtime = os.path.getmtime(matrix_path)
                time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                self.log(f"DEBUG: File found. Size={os.path.getsize(matrix_path)} bytes, Modified={time_str}")
                
                self.cam2mach_matrix = np.load(matrix_path)
                self.log("✅ 已加载坐标标定矩阵 cam2mach_matrix.npy")
                if self.cam2mach_matrix is not None:
                     self.log(f"DEBUG: Matrix[0,0]={self.cam2mach_matrix[0,0]:.4f}, Tx={self.cam2mach_matrix[0,2]:.4f}")
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
                self.slot_templates = None
                
            # Load Golden Path for visualization
            if os.path.exists("config/disp_poly_cell0.npy"):
                # Path is usually (N, 2)
                self.golden_path_cache = np.load("config/disp_poly_cell0.npy")
                self.log(f"✅ 加载点胶路径缓存: {len(self.golden_path_cache)} 点")
            else:
                self.golden_path_cache = None
                
        except Exception as e:
            self.log("加载黄金模板失败：", e)
            self.slot_templates = None
            self.golden_path_cache = None

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
        # Variables (with defaults from config)
        self.feedrate_xy = self.config.get("feedrate_xy", 3000.0)
        self.feedrate_z = self.config.get("feedrate_z", 500.0)
        self.accel = self.config.get("accel", 500.0)
        self.safe_z_cavity = self.config.get("safe_z_cavity", 5.0)
        self.safe_z_product = self.config.get("safe_z_product", 20.0)
        self.work_z = self.config.get("work_z", 0.0)
        self.jog_step = self.config.get("jog_step", 1.0)
        self.z_base = self.config.get("z_base", 0.0)
        self.tool_setter_x = self.config.get("tool_setter_x", None)
        self.tool_setter_y = self.config.get("tool_setter_y", None)
        self.nozzle_offset_x = self.config.get("nozzle_offset_x", 0.0)
        self.nozzle_offset_y = self.config.get("nozzle_offset_y", 0.0)
        
        # Camera Height Params
        self.cam_safe_z_cavity = self.config.get("cam_safe_z_cavity", self.safe_z_cavity)
        self.cam_safe_z_product = self.config.get("cam_safe_z_product", self.safe_z_product)
        self.cam_work_z = self.config.get("cam_work_z", self.work_z)

        # Update UI
        self.sld_spd_xy.setValue(int(self.feedrate_xy))
        self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")

        self.sld_spd_z.setValue(int(self.feedrate_z))
        self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

        self.sld_acc.setValue(int(self.accel))
        self.lbl_acc.setText(f"加速度: {int(self.accel)}")

        self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
        self.spn_safe_z_product.setValue(self.safe_z_product)
        self.spn_work_z.setValue(self.work_z)
        
        if hasattr(self, 'lbl_z_base'):
             self.lbl_z_base.setText(f"当前 Z 基准: {self.z_base:.2f} mm")

        # Step Size Buttons
        # Find button matching jog_step and verify logic (omitted for brevity, keeping old behavior logic implicitly)
        # Actually checking buttons logic might be needed if I rely on UI state for step.
        # Assuming UI step buttons logic handles itself on click.

    def save_app_settings(self):
        self.config.set("feedrate_xy", self.feedrate_xy)
        self.config.set("feedrate_z", self.feedrate_z)
        self.config.set("accel", self.accel)
        self.config.set("safe_z_cavity", self.safe_z_cavity)
        self.config.set("safe_z_product", self.safe_z_product)
        self.config.set("work_z", self.work_z)
        self.config.set("jog_step", self.jog_step)
        self.config.set("z_base", self.z_base)
        self.config.set("tool_setter_x", self.tool_setter_x)
        self.config.set("tool_setter_y", self.tool_setter_y)
        self.config.set("nozzle_offset_x", self.nozzle_offset_x)
        self.config.set("nozzle_offset_y", self.nozzle_offset_y)
        
        self.config.set("cam_safe_z_cavity", self.cam_safe_z_cavity)
        self.config.set("cam_safe_z_product", self.cam_safe_z_product)
        self.config.set("cam_work_z", self.cam_work_z)
        
        # Save USB URL
        if hasattr(self, 'edit_usb_url'):
            self.config.set("usb_cam_url", self.edit_usb_url.text().strip())
        
        self.config.save_app_settings()



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

    # ====== USB Camera Slots ======
    def update_usb_image(self, qimg):
        if hasattr(self, 'label_usb_cam'):
             # Check Flip
             h_flip = False
             v_flip = False
             
             if hasattr(self, 'chk_usb_flip_h') and self.chk_usb_flip_h.isChecked():
                 h_flip = True
             if hasattr(self, 'chk_usb_flip') and self.chk_usb_flip.isChecked():
                 v_flip = True
                 
             if h_flip or v_flip:
                 qimg = qimg.mirrored(h_flip, v_flip)

             # 简单的缩放以适应 label
             # scaled_img = qimg.scaled(self.label_usb_cam.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.label_usb_cam.setPixmap(QPixmap.fromImage(qimg))
    
    def on_usb_status(self, connected):
        if not connected:
            if hasattr(self, 'label_usb_cam'):
                self.label_usb_cam.setText("Disconnected / Retry...")
                self.label_usb_cam.setStyleSheet("color: red; background-color: black;")
        else:
             if hasattr(self, 'label_usb_cam'):
                self.label_usb_cam.setStyleSheet("color: white; background-color: black;")

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

        # ====== 🛠️ B. 标定阶段 1 (手动8点标定模式) ======
        elif self.calib_step in [100, 101]:
            # 画出所有已选的点
            for i, pt in enumerate(self.calib_vision_pts):
                px, py = int(pt[0]), int(pt[1])
                
                # 最后一个点如果是正在微调中 (Step 101)，显示为红色
                is_active = (self.calib_step == 101 and i == len(self.calib_vision_pts) - 1)
                
                if is_active:
                    color = (0, 0, 255) # Red for active
                    cv2.circle(vis_img, (px, py), 6, color, -1)
                    cv2.circle(vis_img, (px, py), 20, (0, 255, 255), 2)
                    cv2.line(vis_img, (px - 30, py), (px + 30, py), (0, 255, 255), 1)
                    cv2.line(vis_img, (px, py - 30), (px, py + 30), (0, 255, 255), 1)
                    cv2.putText(vis_img, f"#{i + 1} Adjusting", (px + 20, py - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    color = (0, 255, 0) # Green for confirmed
                    cv2.circle(vis_img, (px, py), 6, color, -1)
                    cv2.putText(vis_img, f"#{i + 1}", (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if self.calib_step == 100:
                msg = f"Click to select Point #{len(self.calib_vision_pts)+1} (Z to Undo)"
                cv2.putText(vis_img, msg, (30, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                msg = "WASD to Move | Q for Step | SPACE to Confirm | Z to Undo"
                cv2.putText(vis_img, msg, (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ====== 🛠️ C. 标定阶段 2 (机器对位模式) - 视觉引导 ======
        elif self.calib_step >= 200:
             # 计算当前正在对第几个点
            idx = self.calib_step - 200
            if idx < len(self.calib_vision_pts):
                target_pt = self.calib_vision_pts[idx]
                tx, ty = int(target_pt[0]), int(target_pt[1])
                
                # 0. 必须显示中心十字 (红色)，作为对准基准
                cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)
                cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)
                cv2.circle(vis_img, (cx, cy), 15, (0, 0, 255), 2)
                
                # 1. 画目标点 (橙色)
                cv2.circle(vis_img, (tx, ty), 8, (0, 165, 255), -1) 
                cv2.putText(vis_img, f"TARGET #{idx + 1}", (tx + 15, ty - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)

                # 2. 画箭头：从【屏幕中心】指向【目标点】 (Center -> Target)
                # 含义：请移动机器，去追这个点 / 或者仅仅指示这个点在哪
                # 用户要求：箭头指向目标点
                cv2.arrowedLine(vis_img, (cx, cy), (tx, ty), (0, 255, 255), 3, tipLength=0.1)
                
                dist_px = ((tx - cx)**2 + (ty - cy)**2)**0.5
                cv2.putText(vis_img, f"Dist: {dist_px:.1f}px", (cx + 20, cy + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ====== 🛠️ D. 标定阶段 1 (视觉微调模式) - 【找回丢失的代码！】 ======
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

        # ====== F. 针头标定模式 (新增) ======
        # 注意：现在使用非模态对话框，不再依赖 self.nozzle_calib_step
        # 只要 self.show_crosshair 为 True 且不是在其他特定模式下，就会显示十字
        # 但为了避免混淆，我们可以检查一下是否正在标定
        elif hasattr(self, 'nozzle_calib_dlg') and self.nozzle_calib_dlg.isVisible():
             if self.show_crosshair:
                # 画中心十字 (红色)
                cv2.line(vis_img, (cx, 0), (cx, h), (0, 0, 255), 1)
                cv2.line(vis_img, (0, cy), (w, cy), (0, 0, 255), 1)
                cv2.circle(vis_img, (cx, cy), 20, (0, 0, 255), 2)
                cv2.putText(vis_img, "NOZZLE CALIBRATION", (20, h - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
            # 每 3 帧检测一次 (降低 66% 负荷)
            if self._detect_frame_count % 3 == 0:
                # Ensure grayscale for vision processing
                if len(img.shape) == 3:
                     gray_for_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                     gray_for_proc = img
                     
                _, detected_vis_img = self.get_vision_result(gray_for_proc)
                
                self._last_detected_img = detected_vis_img
                vis_img = detected_vis_img
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

            # Overlay Calibration Info (using QPainter on QPixmap)
            if self.calib_step == 10:
                from PyQt5.QtGui import QPainter, QPen, QColor, QFont
                painter = QPainter(final_pix)
                painter.setPen(QPen(QColor(0, 255, 255), 2))
                painter.setFont(QFont("Arial", 14, QFont.Bold))
                painter.drawText(20, 40, f"Step: {self.calib_step_size} px (Press Q)")
                painter.end()

            self.label.setPixmap(final_pix)

        except Exception as e:
            self.log("[render_current_frame] 渲染出错：", e)

    def save_image(self):
        if self.last_frame is None:
            self.log("❌ 没有图像可以保存")
            return

        # Pause grabbing
        self.camera_manager.stop_grabbing()
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

        # Resume grabbing
        if self.camera_manager.is_open:
            self.camera_manager.start_grabbing()
            # Reconnect signal
            sig = self.camera_manager.get_frame_signal()
            if sig:
                try: sig.disconnect()
                except: pass
                sig.connect(self.update_image)
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
        
        # 1. Halcon 模式下不需要黄金示教
        if self.chk_use_halcon.isChecked():
            QMessageBox.information(self, "提示", "Halcon 模式不需要手动进行黄金示教。\n它会自动使用最新的保存图片作为基准。")
            return

        self.freeze_display = True
        
        # Define absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config")
        mold2_path = os.path.join(config_dir, "mold2.png")
        mold2_bak_path = os.path.join(config_dir, "mold2.bak")

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
                    if os.path.exists(mold2_path):
                        shutil.copy(mold2_path, mold2_bak_path)
                        self.log("ℹ️ 已备份原 config/mold2.png 为 config/mold2.bak")

                    # 2. 读取最新图片并旋转 180 度
                    img_latest = cv2.imread(candidate_path)
                    if img_latest is not None:
                        # 旋转 180 度 (flip -1)
                        img_rotated = cv2.flip(img_latest, -1)
                        cv2.imwrite(mold2_path, img_rotated)
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

        # Use absolute path check
        if not os.path.exists(mold2_path):
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

        from core.path_teach import run_path_teach
        
        # 暂时默认编辑第 0 个格子
        cell_idx = 0
        self.log(f"🧵 启动路径编辑器 (Cell {cell_idx})...")
        
        # 运行编辑器 (会阻塞直到窗口关闭)
        ok = run_path_teach(cell_index=cell_idx)
        
        if ok:
            self.log(f"✅ 路径示教完成 (Cell {cell_idx})")
            # 关键修复：编辑完成后，必须强制刷新 mold_detector 的缓存
            from core.mold_detector import reload_golden_templates
            reload_golden_templates()
            self.log("🔄 已刷新模具匹配模板")
            
            # Halcon Cache Invalidation
            if hasattr(self, 'halcon_path_template'):
                self.halcon_path_template = None
                self.log("🔄 已清除 Halcon 路径缓存，下次识别将重新计算")
        else:
            self.log("⚠ 路径示教取消或未保存")
    # ================== 点胶任务按钮（占位） ==================
        # 替换原有的 on_calibrate_coords
    def on_calibrate_coords(self):
        """标定流程主控：手动 8 点视觉取点 -> 8 点机器对位 -> 计算"""

        # --- 阶段 1: 启动 - 进入手动视觉取点模式 (8点) ---
        if self.calib_step == 0:
            if self.last_frame is None:
                self.log("❌ 请先启动相机")
                return

            self.calib_vision_pts = []  # 清空视觉点
            self.calib_machine_pts = [] # 清空机器点
            self.calib_step = 100       # 状态 100: 等待点击第 N 个视觉点
            
            self.btn_calibrate.setText("请点击画面选取第 1 个点")
            self.btn_calibrate.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;") 
            self.btn_calibrate.setEnabled(False) 

            msg = (
                "【手动8点标定模式】\n\n"
                "1. 请在左侧画面中点击模具的**8个特征点**（尽量覆盖各区域）。\n"
                "2. 操作说明：\n"
                "   - 点击画面 -> 放置红点\n"
                "   - WASD 键 -> 微调位置 (Q键改步长)\n"
                "   - 空格/回车 -> 确认该点\n"
                "   - **Z 键 -> 撤销/删除上一个点**\n\n"
                "👉 现在，请点击第 1 个点。"
            )
            self.log(msg)
            QMessageBox.information(self, "标定指南 (8点 + 撤销)", msg)
            self.render_current_frame()

        # --- 阶段 1.5: 视觉点确认 (状态 101 -> 100) ---
        elif self.calib_step == 101:
            # 用户点击了“确认第 N 点”按钮
            
            n_confirmed = len(self.calib_vision_pts)
            self.log(f"✅ 第 {n_confirmed} 个视觉点已确认。")
            
            if n_confirmed < 8:
                self.calib_step = 100 # 回到等待点击状态
                self.btn_calibrate.setText(f"请点击画面选取第 {n_confirmed + 1} 个点")
                self.btn_calibrate.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")
                self.btn_calibrate.setEnabled(False) 
                self.log(f"👉 请继续点击第 {n_confirmed + 1} 个点。 (按 Z 可撤销)")
            else:
                # 8点选完，进入机器对位
                self.calib_step = 200 # 机器对位状态起始
                self.show_crosshair = True
                
                self.btn_calibrate.setText("记录第 1 点 (需移动机器对准)")
                self.btn_calibrate.setStyleSheet("background-color: #F59E0B; color: black; font-weight: bold;") # 橙色
                self.btn_calibrate.setEnabled(True)
                
                self.log("✅ 8个视觉点采集完毕！进入机器对位阶段。")
                self.log("👉 请根据屏幕上的【箭头指示】，移动机器使相机中心对准【第 1 个视觉点】。")

        # --- 阶段 2: 机器对位 (状态 200+) ---
        elif self.calib_step >= 200:
            # 记录当前机器坐标
            mx, my, mz = self.machine.get_position()
            self.calib_machine_pts.append([mx, my])
            
            # 200 -> idx 1
            idx = self.calib_step - 200 + 1 
            self.log(f"📍 记录机器点 {idx}: ({mx:.2f}, {my:.2f})")
            
            if len(self.calib_machine_pts) < 8:
                self.calib_step += 1
                idx_next = idx + 1
                self.btn_calibrate.setText(f"记录第 {idx_next} 点 (需移动机器对准)")
                self.log(f"👉 请根据箭头指示，移动机器对准第 {idx_next} 个点。")
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

        # ====== 情况 A：手动6点标定模式 (新逻辑 100/101) ======
        if self.calib_step == 100:
            # 用户点击了屏幕，添加/更新当前点
            self.calib_vision_pts.append([click_img_x, click_img_y])
            self.calib_step = 101 # 进入微调确认阶段
            
            n = len(self.calib_vision_pts)
            self.btn_calibrate.setText(f"确认第 {n} 个点")
            self.btn_calibrate.setStyleSheet("background-color: #10B981; color: white; font-weight: bold;") # 绿色
            self.btn_calibrate.setEnabled(True)
            
            self.log(f"📍 已选中第 {n} 个点。请使用 WASD 微调，然后点击按钮确认。")
            self.render_current_frame()
            return

        elif self.calib_step == 101:
            # 如果是 101 (微调中)，点击屏幕可以“瞬间移动”当前正在调的那个点
            if self.calib_vision_pts:
                self.calib_vision_pts[-1] = [click_img_x, click_img_y]
                self.render_current_frame()
            return

        # ====== 情况 B：标定微调 (旧逻辑，保留兼容) ======
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
            # dx_mm = m00*dx_pix + m01*dy_pix=
            # dy_mm = m10*dx_pix + m11*dy_pix
            m = self.cam2mach_matrix
            dx_mm = m[0, 0] * dx_pix + m[0, 1] * dy_pix
            dy_mm = m[1, 0] * dx_pix + m[1, 1] * dy_pix

            self.log(f"🎯 [验证] 像素偏差({dx_pix:.1f}, {dy_pix:.1f}) -> 机器需移动({dx_mm:.2f}, {dy_mm:.2f})")

            # 4. 安全检查：计算目标绝对坐标，防止超程
            cur_x, cur_y, _ = self.machine.get_position()
            target_x = cur_x + dx_mm
            target_y = cur_y + dy_mm

            # 假设软限位是 0 ~ 300 (根据实际机器调整)
            LIMIT_MIN = 0.0
            LIMIT_MAX = 300.0

            if not (LIMIT_MIN <= target_x <= LIMIT_MAX and LIMIT_MIN <= target_y <= LIMIT_MAX):
                self.log(f"⛔ 目标超程！({target_x:.2f}, {target_y:.2f}) 超出范围 [{LIMIT_MIN}, {LIMIT_MAX}]，已取消移动。")
                return

            # 5. 发送相对移动指令 (G91)
            # 速度稍慢一点方便观察，例如 3000
            speed = 3000
            script = f"G91\nG0 X{dx_mm:.3f} Y{dy_mm:.3f} F{speed}\nG90" # 动完切回绝对模式
            self.machine._post_gcode(script)

    def keyPressEvent(self, event):
        """键盘事件：WASD 微调，1234 切换选中点，Q 切换步长"""
        
        # ====== 新增: 手动8点标定微调 (Step 100/101) ======
        if self.calib_step == 100:
            if event.key() == Qt.Key_Z:
                if len(self.calib_vision_pts) > 0:
                    self.calib_vision_pts.pop()
                    n = len(self.calib_vision_pts)
                    self.btn_calibrate.setText(f"请点击画面选取第 {n + 1} 个点")
                    self.btn_calibrate.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")
                    self.btn_calibrate.setEnabled(False)
                    self.log(f"⏪ 已撤销上一个点。当前已选: {n} 个")
                    self.render_current_frame()
                else:
                    self.log("⚠ 没有可以撤销的点")
            return

        if self.calib_step == 101:
            key = event.key()
            step = self.calib_step_size
            
            # Z: 撤销当前正在选的点 -> 回到 Step 100
            if key == Qt.Key_Z:
                if self.calib_vision_pts:
                    self.calib_vision_pts.pop()
                    n = len(self.calib_vision_pts)
                    self.calib_step = 100
                    self.btn_calibrate.setText(f"请点击画面选取第 {n + 1} 个点")
                    self.btn_calibrate.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")
                    self.btn_calibrate.setEnabled(False)
                    self.log(f"⏪ 已取消当前点。请重新点击。")
                    self.render_current_frame()
                return

            # Q: 切换步长
            if key == Qt.Key_Q:
                if self.calib_step_size == 1.0: self.calib_step_size = 5.0
                elif self.calib_step_size == 5.0: self.calib_step_size = 10.0
                elif self.calib_step_size == 10.0: self.calib_step_size = 0.1
                elif self.calib_step_size == 0.1: self.calib_step_size = 0.5
                else: self.calib_step_size = 1.0
                self.log(f"🔧 微调步长: {self.calib_step_size} px")
                
            # WASD: 微调当前点 (列表最后一个)
            elif key == Qt.Key_W: self.calib_vision_pts[-1][1] -= step
            elif key == Qt.Key_S: self.calib_vision_pts[-1][1] += step
            elif key == Qt.Key_A: self.calib_vision_pts[-1][0] -= step
            elif key == Qt.Key_D: self.calib_vision_pts[-1][0] += step
            
            # Space/Enter: 确认当前点 (相当于点击按钮)
            elif key == Qt.Key_Space or key == Qt.Key_Return:
                self.on_calibrate_coords() # 触发下一步逻辑
                return

            if key in [Qt.Key_Q, Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D]:
                self.render_current_frame()
            event.accept()
            return

        if self.calib_step == 10:
            key = event.key()
            step = self.calib_step_size  # Use dynamic step size

            # 切换步长
            if key == Qt.Key_Q:
                # Cycle: 1.0 -> 5.0 -> 10.0 -> 0.1 -> 0.5 -> 1.0
                if self.calib_step_size == 1.0: self.calib_step_size = 5.0
                elif self.calib_step_size == 5.0: self.calib_step_size = 10.0
                elif self.calib_step_size == 10.0: self.calib_step_size = 0.1
                elif self.calib_step_size == 0.1: self.calib_step_size = 0.5
                else: self.calib_step_size = 1.0
                
                self.log(f"🔧 标定微调步长: {self.calib_step_size} px")
                self.render_current_frame()

            # 切换选中点
            elif key == Qt.Key_1:
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
            if key in [Qt.Key_Q, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                       Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D]:
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
        
            # Consistent Absolute Path Saving
            base_dir = os.path.dirname(os.path.abspath(__file__))
            matrix_path = os.path.join(base_dir, "config", "cam2mach_matrix.npy")
            self.log(f"DEBUG: Saving matrix to: {matrix_path}")
            np.save(matrix_path, M)
        
            self.log("✅ 标定完成！矩阵已保存。")
            self.log(f"DEBUG: Saved Matrix[0,0]={M[0,0]:.4f}, Tx={M[0,2]:.4f}")
            QMessageBox.information(self, "成功", "坐标系标定完成！")
        except Exception as e:
            self.log(f"❌ 计算失败: {e}")

        self.calib_step = 0
        self.show_crosshair = False
        self.btn_calibrate.setText("标定坐标系")
        self.btn_calibrate.setStyleSheet("")

    def on_virtual_run(self):
        """
        按钮：相机空跑 (原虚拟跑)
        使用相机中心进行路径演示，不加偏移，使用【相机专用】高度参数。
        """
        # 准备相机高度参数
        height_params = {
            'safe_z_cavity': self.cam_safe_z_cavity,
            'safe_z_product': self.cam_safe_z_product,
            'work_z': self.cam_work_z,
            'speed_xy': int(self.feedrate_xy),
            'speed_z': int(self.feedrate_z)
        }
        
        def save_cam_params(dialog):
            self.cam_safe_z_cavity = dialog.safe_z_cavity
            self.cam_safe_z_product = dialog.safe_z_product
            self.cam_work_z = dialog.work_z
            # 速度参数是共享的，也顺便更新一下
            self.feedrate_xy = dialog.speed_xy
            self.feedrate_z = dialog.speed_z
            self.save_app_settings()

        self._prepare_dry_run(
            use_nozzle_offset=False, 
            title="相机空跑 (Camera Dry Run)",
            initial_params=height_params,
            save_callback=save_cam_params
        )

    def on_dry_run(self):
        """
        按钮：针头空跑
        使用针头进行路径演示，加上偏移，使用【主界面】高度参数。
        """
        # 0. 检查参数是否设置
        if self.safe_z_cavity is None or self.safe_z_product is None or self.work_z is None:
             QMessageBox.warning(self, "参数未设置", "检测到针头高度参数未设置！\n\n请先在批量运行或设置中配置高度参数。")
             return

        # 准备针头高度参数 (直接取主界面值)
        height_params = {
            'safe_z_cavity': float(self.safe_z_cavity),
            'safe_z_product': float(self.safe_z_product),
            'work_z': float(self.work_z),
            'speed_xy': int(self.feedrate_xy),
            'speed_z': int(self.feedrate_z)
        }

        def save_nozzle_params(dialog):
            # 更新主界面控件，控件会自动更新变量
            self.spn_safe_z_cavity.setValue(dialog.safe_z_cavity)
            self.spn_safe_z_product.setValue(dialog.safe_z_product)
            self.spn_work_z.setValue(dialog.work_z)
            self.feedrate_xy = dialog.speed_xy
            self.feedrate_z = dialog.speed_z
            self.save_app_settings()

        self._prepare_dry_run(
            use_nozzle_offset=True, 
            title="针头空跑 (Nozzle Dry Run)",
            initial_params=height_params,
            save_callback=save_nozzle_params
        )

    def _prepare_dry_run(self, use_nozzle_offset=False, title="空跑", initial_params=None, save_callback=None):
        """
        统一的空跑准备逻辑
        :param use_nozzle_offset: 是否应用针头偏移
        :param initial_params: 初始高度和速度参数 (dict)
        :param save_callback: 保存参数的回调函数 (func(dialog))
        """
        # 1. 基础检查
        if self.cam2mach_matrix is None:
            QMessageBox.warning(self, "未标定", "请先完成【标定坐标系】！")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "无图像", "请先启动相机！")
            return

        # 暂停实时识别
        if self.live_detect:
            self.detect_mold()

        try:
            # --- A. 图像处理 & 识别 ---
            # --- A. 图像处理 & 识别 ---
            img_dry = self.last_frame.copy()
            img_dry = cv2.flip(img_dry, -1)  # 翻转
            
            if len(img_dry.shape) == 3:
                 gray_dry = cv2.cvtColor(img_dry, cv2.COLOR_BGR2GRAY)
            else:
                 gray_dry = img_dry

            paths_px, _ = self.get_vision_result(gray_dry)
            if not paths_px:
                QMessageBox.warning(self, "未识别", "未识别到模具路径！")
                return

            # --- B. 视觉反馈 ---
            self.dry_run_paths_to_draw = paths_px
            self.show_crosshair = True
            self.render_current_frame()

            # --- C. 弹出参数对话框 ---
            h_img, w_img = img_dry.shape[:2]
            center_x, center_y = w_img / 2, h_img / 2
            
            # 如果没有传入参数，就用默认的 (防呆)
            if initial_params is None:
                initial_params = {
                    'safe_z_cavity': float(self.safe_z_cavity),
                    'safe_z_product': float(self.safe_z_product),
                    'work_z': float(self.work_z),
                    'speed_xy': int(self.feedrate_xy),
                    'speed_z': int(self.feedrate_z)
                }
            
            origin_pos = self.machine.get_position()
            
            # 根据模式决定偏移量
            if use_nozzle_offset:
                nozzle_offset = (self.nozzle_offset_x, self.nozzle_offset_y)
            else:
                nozzle_offset = (0.0, 0.0)

            dialog = DryRunDialog(
                self, 
                paths_px, 
                self.cam2mach_matrix, 
                origin_pos, 
                nozzle_offset, 
                initial_params,
                (center_x, center_y),
                is_real_run=False, # 空跑都不出胶
                save_callback=save_callback  # ✅ 传入回调，供“仅保存”按钮使用
            )
            
            dialog.setWindowTitle(f"{title} - 参数确认")

            if dialog.exec_() == QDialog.Accepted:
                # --- D. 执行 ---
                full_script = dialog.get_gcode()
                self.log(f"🚀 开始执行【{title}】...")
                
                lines = full_script.split('\n')
                self.dry_run_worker = GCodeLineWorker(self.machine, lines)
                self.dry_run_worker.finished_signal.connect(self.on_dry_run_finished)
                self.dry_run_worker.log_signal.connect(self.log)
                self.dry_run_worker.start()
                
                # 回调保存参数
                if save_callback:
                    save_callback(dialog)

                # 弹出控制面板
                self.dry_run_ctrl_dlg = DryRunControlDialog(self, self.dry_run_worker)
                self.dry_run_ctrl_dlg.setWindowTitle(f"{title} - 运行中")
                self.dry_run_ctrl_dlg.exec_()
            else:
                self.log(f"🚫 用户取消{title}")
                self.dry_run_paths_to_draw = []
                self.render_current_frame()

        except Exception as e:
            self.log(f"❌ {title}准备出错: {e}")
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
        按钮：开始点胶 (真实模式)
        逻辑与空跑类似，但会生成 VALVE_ON / VALVE_OFF 指令。
        """
        # 1. 基础检查
        if self.cam2mach_matrix is None:
            QMessageBox.warning(self, "未标定", "请先完成【标定坐标系】！")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "无图像", "请先启动相机！")
            return

        # 暂停实时识别
        if self.live_detect:
            self.detect_mold()

        try:
            # --- A. 图像处理 & 识别 ---
            # USE Unified Vision Interface
            img_dry = self.last_frame.copy()
            img_dry = cv2.flip(img_dry, -1)  # 翻转
            
            if len(img_dry.shape) == 3:
                 gray_dry = cv2.cvtColor(img_dry, cv2.COLOR_BGR2GRAY)
            else:
                 gray_dry = img_dry

            paths_px, _ = self.get_vision_result(gray_dry)
            if not paths_px:
                QMessageBox.warning(self, "未识别", "未识别到模具路径！")
                return

            # --- B. 视觉反馈 ---
            self.dry_run_paths_to_draw = paths_px
            self.show_crosshair = True
            self.render_current_frame()

            # --- C. 弹出参数对话框 (is_real_run=True) ---
            h_img, w_img = img_dry.shape[:2]
            center_x, center_y = w_img / 2, h_img / 2
            
            initial_params = {
                'safe_z_cavity': float(self.safe_z_cavity),
                'safe_z_product': float(self.safe_z_product),
                'work_z': float(self.work_z),
                'speed_xy': int(self.feedrate_xy),
                'speed_z': int(self.feedrate_z)
            }
            
            origin_pos = self.machine.get_position()
            nozzle_offset = (self.nozzle_offset_x, self.nozzle_offset_y)

            # 关键：传入 is_real_run=True
            dialog = DryRunDialog(
                self, 
                paths_px, 
                self.cam2mach_matrix, 
                origin_pos, 
                nozzle_offset, 
                initial_params,
                (center_x, center_y),
                is_real_run=True,
                save_callback=self.on_dry_run.save_nozzle_params # 复用针头参数保存逻辑
            )

            if dialog.exec_() == QDialog.Accepted:
                # --- D. 执行 ---
                full_script = dialog.get_gcode()
                self.log("🚀 开始执行【真实点胶】...")
                
                lines = full_script.split('\n')
                self.dry_run_worker = GCodeLineWorker(self.machine, lines)
                self.dry_run_worker.finished_signal.connect(self.on_dry_run_finished)
                self.dry_run_worker.log_signal.connect(self.log)
                self.dry_run_worker.start()
                
                # 保存参数
                self.spn_safe_z_cavity.setValue(dialog.safe_z_cavity)
                self.spn_safe_z_product.setValue(dialog.safe_z_product)
                self.spn_work_z.setValue(dialog.work_z)
                self.feedrate_xy = dialog.speed_xy
                self.feedrate_z = dialog.speed_z

                # 弹出控制面板
                self.dry_run_ctrl_dlg = DryRunControlDialog(self, self.dry_run_worker)
                self.dry_run_ctrl_dlg.setWindowTitle("点胶控制 (运行中)")
                self.dry_run_ctrl_dlg.exec_()
            else:
                self.log("🚫 用户取消点胶")
                self.dry_run_paths_to_draw = []
                self.render_current_frame()

        except Exception as e:
            self.log(f"❌ 点胶准备出错: {e}")
            import traceback
            traceback.print_exc()

    def on_batch_run(self):
        """
        按钮：批量运行
        打开批量运行配置对话框
        """
        # 1. 基础检查
        if self.cam2mach_matrix is None:
            QMessageBox.warning(self, "未标定", "请先完成【标定坐标系】！")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "无图像", "请先启动相机！")
            return
            
        dlg = BatchRunDialog(self)
        dlg.exec_()

    def on_test_valve(self):
        """
        按钮：测试出胶 (点动测试)
        打开阀门 0.5秒 后关闭
        """
        self.log("🔧 测试出胶：打开阀门 0.5s ...")
        self.machine.valve_on()
        # 注意：这里会阻塞 UI 0.5s，简单测试没问题，复杂逻辑建议用 Worker
        import time
        time.sleep(0.5)
        self.machine.valve_off()
        self.log("🔧 测试出胶：关闭阀门。")

    # ================== 相机参数联动 ==================
    def sync_from_camera_to_ui(self):
        """
        Standardized sync: Camera Value -> UI.
        Exposure: us -> ms -> Slider(int)
        Gain: dB -> Slider(int)
        Gamma: float -> Slider(int)
        """
        params = self.camera_manager.get_params()

        if 'ExposureTime' in params:
            val = float(params['ExposureTime']) # us
            ms_val = val / 1000.0
            
            # Update SpinBox
            self.spn_exp.blockSignals(True)
            self.spn_exp.setValue(ms_val)
            self.spn_exp.blockSignals(False)
            
            # Simple Linear: Val(ms) * 10 -> Slider
            # Range 0-10ms -> 0-100
            pos = int(ms_val * 10)
            self.sld_exp.blockSignals(True)
            self.sld_exp.setValue(pos)
            self.sld_exp.blockSignals(False)

        if 'Gain' in params:
            val = float(params['Gain'])
            
            self.spn_gain.blockSignals(True)
            self.spn_gain.setValue(val)
            self.spn_gain.blockSignals(False)

            # Simple Linear: Val(dB) * 10 -> Slider
            # Range 0-10dB -> 0-100
            pos = int(val * 10)
            self.sld_gain.blockSignals(True)
            self.sld_gain.setValue(pos)
            self.sld_gain.blockSignals(False)

        if 'Gamma' in params:
            val = float(params['Gamma'])
            
            self.spn_gamma.blockSignals(True)
            self.spn_gamma.setValue(val)
            self.spn_gamma.blockSignals(False)

            # Simple Linear: Val * 10 -> Slider
            # Range 0.1-4.0 -> 1-40
            pos = int(val * 10)
            self.sld_gamma.blockSignals(True)
            self.sld_gamma.setValue(pos)
            self.sld_gamma.blockSignals(False)





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
        
        # 0. 检查参数是否设置
        if self.safe_z_product is None:
             QMessageBox.warning(self, "参数未设置", "检测到【产品安全高度】未设置！\n\n对刀需要使用该高度作为安全移动高度。\n请先在设置中配置。")
             return

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

    # ================== Halcon Integration ==================
    def on_halcon_toggled(self):
        enabled = self.chk_use_halcon.isChecked()
        self.btn_load_halcon.setEnabled(enabled)
        if enabled and not self.halcon_manager.is_initialized:
            # Try init
            if not self.halcon_manager.init_engine():
                QMessageBox.warning(self, "Error", "Failed to initialize Halcon Engine.\nEnsure mvtec-halcon is installed.")
                self.chk_use_halcon.setChecked(False)
                self.btn_load_halcon.setEnabled(False)

    def on_load_halcon_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Halcon Script", "config", "Halcon Scripts (*.hdev *.shm)")
        if path:
            if self.halcon_manager.load_script(path):
                self.log(f"✅ Halcon script loaded: {os.path.basename(path)}")
            else:
                self.log(f"❌ Failed to load Halcon script: {path}")

    def get_vision_result(self, gray_img):
        """
        Unified vision interface.
        Returns: (paths_pixels, debug_img_bgr)
        """
        if self.chk_use_halcon.isChecked():
            # Use Halcon
            res = self.halcon_manager.process_image(gray_img)
            debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            found_paths = []
            
            # ROI Center from Halcon Script (Hardcoded based on user's ROI)
            # ROI: 448.829, 936.949, 854.524, 1289.7
            # Center Row = 651.6765, Col = 1113.3245
            ref_row, ref_col = 651.6765, 1113.3245
            
            if res and "matches" in res:
                for match in res["matches"]:
                    # Halcon: Row=Y, Col=X
                    cx, cy = match['col'], match['row']
                    score = match['score']
                    angle = match['angle'] # Radians
                    
                    # 1. Draw Match Center
                    cx_i, cy_i = int(cx), int(cy)
                    cv2.drawMarker(debug_img, (cx_i, cy_i), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)

                    # 2. Transform Golden Path
                    if not hasattr(self, 'halcon_path_template') or self.halcon_path_template is None:
                        self._compute_halcon_template()

                    mold_segments = []
                    
                    if hasattr(self, 'halcon_path_template') and self.halcon_path_template is not None:
                         try:
                            path_segments = self.halcon_path_template
                            
                            # 3. Transform & Draw Each Segment
                            for pts in path_segments:
                                # pts is already Centered relative to Halcon ROI Center (Reference Point)
                                
                                # b. Rotate
                                # Matrix: [[cos, -sin], [sin, cos]]
                                sys_ang = angle
                                cos_a = np.cos(sys_ang)
                                sin_a = np.sin(sys_ang)
                                
                                x_new = pts[:, 0] * cos_a - pts[:, 1] * sin_a
                                y_new = pts[:, 0] * sin_a + pts[:, 1] * cos_a
                                
                                pts_rotated = np.stack([x_new, y_new], axis=1)
                                
                                # c. Translate to machine center (Halcon Match)
                                pts_transformed = pts_rotated + np.array([cx, cy])
                                
                                # d. Draw (Magenta)
                                pts_draw = pts_transformed.astype(np.int32)
                                is_closed = False # Dots might be tiny closed loops or just points
                                # Heuristic: if len < 3, draw as point
                                if len(pts_draw) == 1:
                                     cv2.circle(debug_img, tuple(pts_draw[0]), 5, (255, 0, 255), -1)
                                else:
                                     cv2.polylines(debug_img, [pts_draw], is_closed, (255, 0, 255), 2, cv2.LINE_AA)
                                
                                mold_segments.append(pts_transformed)
                                
                         except Exception as e:
                            print(f"[ERROR] Path draw failed: {e}")
                            
                    else:
                         mold_segments.append(np.array([[cx, cy]]))
                    
                    if mold_segments:
                        found_paths.append(mold_segments)
                    
                    # Draw Score
                    text = f"S:{score:.2f}"
                    cv2.putText(debug_img, text, (cx_i + 25, cy_i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
            return found_paths, debug_img
        else:
            # Use Legacy OpenCV
            from core.mold_detector import get_paths_and_image
            return get_paths_and_image(gray_img)

    def _compute_halcon_template(self):
        """
        One-time computation:
        1. Load Latest Capture or mold2.png
        2. Find Big Box -> Get Minv (Warped -> Raw)
        3. Load disp_poly & disp_dots (Warped Space)
        4. Project them to Raw Space
        5. Subtract Halcon ROI Center to get 'Relative Path' for Halcon
        """
        try:
            print("[Halcon] Computing offline path template...")
            import cv2
            import numpy as np
            from core.mold_detector import _find_big_mold
            
            # --- BASE DIR ---
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(base_dir, "config")
            
            # A. Select Source Image
            gray_gold = None
            used_path = "mold2.png"
            
            # 1. Try latest capture first (User Request)
            # latest_capture_path is likely absolute if from QFileDialog, but let's be safe
            if hasattr(self, 'latest_capture_path') and self.latest_capture_path and os.path.exists(self.latest_capture_path):
                 print(f"[Halcon] Using latest capture: {self.latest_capture_path}")
                 img_raw = cv2.imread(self.latest_capture_path)
                 if img_raw is not None:
                      # ⚠️ CRITICAL: Camera is mounted upside down, so we normally rotate 180 for "Golden" usage
                      # The paths are likely drawn on the ROTATED image.
                      # So we must rotate this capture to match the path coordinate space.
                      img_rotated = cv2.flip(img_raw, -1)
                      gray_gold = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
                      used_path = self.latest_capture_path
            
            # 2. Fallback to mold2.png (Absolute Path)
            mold2_path = os.path.join(config_dir, "mold2.png")
            if gray_gold is None:
                if os.path.exists(mold2_path):
                    print(f"[Halcon] Using {mold2_path}")
                    gray_gold = cv2.imread(mold2_path, cv2.IMREAD_GRAYSCALE)
                    used_path = mold2_path
                else:
                    print(f"[Halcon] ERR: No source image found. Checked: {mold2_path}")
                    # Also checked latest capture if available
                    return

            # B. Get Perspective Matrix (Warped -> Raw)
            rect, big_box = _find_big_mold(gray_gold)
            if big_box is None: 
                print("[Halcon] ERR: Could not find big box in source image")
                return
            
            # Recalculate Minv (Same logic as mold_detector)
            warp_size_path = os.path.join(config_dir, "warp_size.npy")
            if os.path.exists(warp_size_path):
                warp_size = np.load(warp_size_path)
                goldW, goldH = int(warp_size[0]), int(warp_size[1])
            else:
                 # Fallback if missing (should not happen if trained)
                 goldW, goldH = 1000, 1000 # Guess or fail?
            
            pts = big_box.astype(np.float32)
            s = pts.sum(axis=1)
            tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
            rect_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            dst_pts = np.array([[0, 0], [goldW - 1, 0], [goldW - 1, goldH - 1], [0, goldH - 1]], dtype=np.float32)
            
            # Minv: Warped (dst) -> Raw (rect)
            Minv = cv2.getPerspectiveTransform(dst_pts, rect_pts)
            
            # C. Load Paths (Warped Space)
            segments = []
            
            # Poly
            poly_path = os.path.join(config_dir, "disp_poly_cell0.npy")
            if os.path.exists(poly_path):
                raw = np.load(poly_path, allow_pickle=True)
                if isinstance(raw, np.ndarray):
                     if raw.dtype == object: 
                         segments.extend([s.astype(np.float32) for s in raw if s is not None])
                     elif raw.ndim == 2:
                         segments.append(raw.astype(np.float32))
                elif isinstance(raw, list):
                    segments.extend([np.array(s, dtype=np.float32) for s in raw])
            else:
                 print(f"[Halcon] Poly file not found: {poly_path}")
            
            # Dots
            dots_path = os.path.join(config_dir, "disp_dots_cell0.npy")
            if os.path.exists(dots_path):
                dots = np.load(dots_path, allow_pickle=True)
                if dots.size > 0:
                    # Treat each dot as a 1-point segment (1, 2)
                     if dots.ndim == 2:
                         for d in dots:
                             segments.append(d.reshape(1, 2).astype(np.float32))

            if not segments:
                print("[Halcon] ERR: No paths loaded")
                return

            # D. Project and Offset
            # Halcon ROI Center (Raw Coords)
            # From: 448.8, 936.9, 854.5, 1289.7
            ref_row, ref_col = 651.6765, 1113.3245
            ref_center = np.array([ref_col, ref_row]) # (x, y)
            
            final_template = []
            for seg in segments:
                # 1. Project Warped -> Raw
                pts_w = seg.reshape(-1, 1, 2)
                pts_raw = cv2.perspectiveTransform(pts_w, Minv).reshape(-1, 2)
                
                # 2. Subtract ROI Center
                pts_rel = pts_raw - ref_center
                final_template.append(pts_rel)
                
            self.halcon_path_template = final_template
            print(f"[Halcon] Template computed! {len(final_template)} segments.")
            
        except Exception as e:
            print(f"[Halcon] Template compute error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("🚀 App Starting...")
    
    # 1. 告诉 Windows 这是一个独立的应用程序，以便在任务栏显示图标
    if sys.platform == 'win32':
        myappid = 'mycompany.myproduct.subproduct.version' # 任意字符串
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    try:
        app = QApplication(sys.argv)
        
        # 2. 给 QApplication 也设置图标
        from PyQt5.QtGui import QIcon
        if os.path.exists("logo.svg"):
            app.setWindowIcon(QIcon("logo.svg"))
            
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
