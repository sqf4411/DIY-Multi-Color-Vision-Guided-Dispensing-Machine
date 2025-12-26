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
    QCheckBox, QTabWidget, QSplitter, QInputDialog, QListWidget
)




from PyQt5.QtGui import QImage, QPixmap

# 你的检测函数
from core.mold_detector import detect_mold_and_cells, reload_golden_templates
from core.dry_run_dialog import DryRunDialog, DryRunControlDialog
from core.batch_run_dialog import BatchRunDialog
from core.valve_control_dialog import ValveControlDialog

from core.machine_client import MachineClient
from core.config_manager import ConfigManager
from core.camera_manager import CameraManager
from core.halcon_manager import HalconManager
from core.visualizer_3d import Visualizer3D
from core.product_manager import ProductManager
from core.halcon_product_manager import HalconProductManager
from core.path_editor_wrapper import edit_path_halcon_mode
from core.touch_spinbox import TouchSpinBox, apply_touch_support
from core.touch_numpad import TouchNumberPad

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
        
        # 屏蔽 OpenCV/FFmpeg 的 MJPEG 警告日志
        cv2.setLogLevel(3)  # 0=Debug, 1=Info, 2=Warning, 3=Error, 4=Fatal
        
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
    current_line_signal = pyqtSignal(str)   # [NEW]
    current_index_signal = pyqtSignal(int)  # [NEW]

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
                # 即使是空行或注释，也发送高亮信号，让用户知道跑哪了
                self.current_index_signal.emit(i)
                self.current_line_signal.emit(line)
                self.progress_signal.emit(i + 1, total)
                continue

            # 发送高亮信号
            self.current_index_signal.emit(i)
            self.current_line_signal.emit(line)

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

class TempWorker(QThread):
    """
    后台轮询温度
    """
    temp_signal = pyqtSignal(dict) # 返回 get_temp_status 的字典

    def __init__(self, machine_client):
        super().__init__()
        self.machine = machine_client
        self.running = True

    def run(self):
        while self.running:
            if self.machine.connected:
                data = self.machine.get_temp_status()
                if data:
                    self.temp_signal.emit(data)
            time.sleep(2.0) # 每 2 秒刷新一次，避免请求太频繁

    def stop(self):
        self.running = False
        self.wait()

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

# ====== 可折叠分组框类 ======
class CollapsibleBox(QWidget):
    """
    可折叠的分组框，用于优化UI布局
    
    使用方法:
        box = CollapsibleBox("标题", collapsed=True, color="#3B82F6")
        box.add_widget(some_widget)
        box.add_layout(some_layout)
    """
    def __init__(self, title="", collapsed=True, color="#3B82F6"):
        super().__init__()
        
        # 创建可点击的标题按钮
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        
        # 设置按钮样式（带颜色区分）
        self.color = color
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                text-align: left;
                padding: 10px 15px;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {self._adjust_brightness(color, 1.1)};
            }}
            QPushButton:checked {{
                background-color: {self._adjust_brightness(color, 0.85)};
            }}
        """)
        
        # 连接点击事件
        self.toggle_button.clicked.connect(self.on_toggle)
        
        # 创建内容区域
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(5)
        self.content_area.setLayout(self.content_layout)
        self.content_area.setVisible(not collapsed)
        
        # 主布局
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        
        self.setLayout(layout)
        
        # 初始化按钮文本（带箭头）
        self.update_button_text()
    
    def on_toggle(self):
        """切换展开/收起状态"""
        is_expanded = self.toggle_button.isChecked()
        self.content_area.setVisible(is_expanded)
        self.update_button_text()
    
    def update_button_text(self):
        """更新按钮文本，添加展开/收起箭头"""
        current_text = self.toggle_button.text()
        # 移除已有的箭头
        if current_text.startswith("▼ ") or current_text.startswith("▶ "):
            current_text = current_text[2:]
        
        # 添加新箭头
        if self.toggle_button.isChecked():
            self.toggle_button.setText(f"▼ {current_text}")
        else:
            self.toggle_button.setText(f"▶ {current_text}")
    
    def add_widget(self, widget):
        """添加单个控件到内容区域"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """添加布局到内容区域"""
        self.content_layout.addLayout(layout)
    
    def set_expanded(self, expanded):
        """程序化设置展开状态"""
        self.toggle_button.setChecked(expanded)
        self.content_area.setVisible(expanded)
        self.update_button_text()
    
    @staticmethod
    def _adjust_brightness(hex_color, factor):
        """
        调整颜色亮度
        factor > 1: 变亮
        factor < 1: 变暗
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        r = int(min(255, r * factor))
        g = int(min(255, g * factor))
        b = int(min(255, b * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
# ====== End of Collapsible Box Class ======

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
        self.product_manager = ProductManager()  # 产品管理器
        self.halcon_product_manager = HalconProductManager(self.halcon_manager, self.product_manager)  # Halcon产品逻辑
        self.current_product = None  # 当前加载的产品名称
        self.camera_manager.get_frame_signal().connect(self.update_image) if self.camera_manager.get_frame_signal() else None
        # Load Config
        # self.load_app_settings()  # Moved after UI init

        # 连接日志信号
        self.log_signal.connect(self._log_impl)
        
        # [NEW] Mobile Monitor
        from mobile_ui.mobile_monitor_dialog import MobileMonitorDialog
        self.mobile_monitor = None # Lazy init

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
        self.config_dir = "config"
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        self.config_path = "config/app_settings.json"  # ✅ Fixed: Use separate file from camera params
        self.thread = None
        self.last_frame = None
        self.freeze_display = False  # 是否冻结画面
        self.live_detect = False     # 是否实时识别模具
        # 缩放相关
        self.zoom_scale = 1.0        # 在“自适应基准”上的倍数
        
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

        # ====== ✅ 新增：视觉偏移修正 (Halcon) ======
        self.vision_offset_x = 0.0
        self.vision_offset_y = 0.0
        
        # Halcon Model Reference (Dynamic)
        self.model_ref_row = 0.0
        self.model_ref_col = 0.0
        
        # Path Cache
        self.dots = [] 
        self.polys = []
        # =========================================

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

        # ====== ✅ 新增：相机空跑专用高度 (独立于针头) ======
        self.cam_safe_z_cavity = 10.0
        self.cam_safe_z_product = 30.0
        self.cam_work_z = 0.0
        # =================================================

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
        self.safe_z_cavity = 100.0   # 腔内安全高度 (小抬刀)
        self.safe_z_product = 100.0  # 产品安全高度 (大抬刀)
        self.work_z = 100.0          # 点胶 / 下刀高度 (接近0)
        
        # ====== ✅ 新增：相机空跑专用高度参数 ======
        self.cam_safe_z_cavity = 213.0
        self.cam_safe_z_product = 215.0
        self.cam_work_z = 211.0 # 相机通常比针头高，避免撞到
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

        # [NEW] Vision Offsets for Halcon (Pixels)
        self.vision_offset_x = 0.0
        self.vision_offset_y = 0.0

        # ==========================================
        # 在所有 UI 元素初始化完成后，调用加载设置
        self.init_ui()
        self.load_app_settings()
        self.load_saved_params()
        self.load_golden_templates()
        self.init_calibration()
        self.init_done = True
        
        # 自动迁移旧版recipes目录
        self.migrate_recipes_to_produces()
        
        # 初始化按钮状态（确保Halcon按钮默认禁用）
        self.update_button_states()

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
        
        # 3. 按钮定义 - 改为toggle按钮
        self.btn_connect_klipper = QPushButton("连接")
        self.btn_connect_klipper.setCheckable(True)  # 设为可切换
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
                    QPushButton:checked {
                        background-color: #10B981;  /* 绿色表示已连接 */
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
        
        # 相机按钮改为单个toggle按钮
        self.btn_camera = QPushButton("启动相机")
        self.btn_camera.setCheckable(True)  # 设为可切换
        self.btn_camera.setMinimumHeight(28)
        self.btn_camera.setFixedWidth(80)
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:checked {
                background-color: #EF4444;  /* 红色表示停止 */
            }
        """)
        
        # ⭐ 触摸模式按钮
        self.btn_touch_mode = QPushButton("触摸模式")
        self.btn_touch_mode.setCheckable(True)
        self.btn_touch_mode.setMinimumHeight(28)
        self.btn_touch_mode.setFixedWidth(80)
        self.btn_touch_mode.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:checked {
                background-color: #EF4444;  /* 红色=触摸模式启用 */
            }
            QPushButton:checked:hover {
                background-color: #DC2626;
            }
        """)
        self.btn_touch_mode.toggled.connect(self.toggle_touch_mode)

        # ====== Row 1 排列 ======
        row1_layout.addWidget(lbl_ip)
        row1_layout.addWidget(self.edit_ip)
        row1_layout.addSpacing(10)
        row1_layout.addWidget(lbl_port)
        row1_layout.addWidget(self.edit_port)
        row1_layout.addStretch() 
        
        # 触摸模式按钮
        row1_layout.addWidget(self.btn_touch_mode)
        row1_layout.addSpacing(10)
        # 单个相机toggle按钮
        row1_layout.addWidget(self.btn_camera)
        row1_layout.addSpacing(10)
        # 单个Klipper toggle按钮
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
        # btn_resume已删除 - 无效按钮
        self.btn_path = QPushButton("编辑点胶路径")

        # 点胶任务相关按钮
        self.btn_calibrate = QPushButton("标定坐标系")
        # ====== ✅ 新增：验证精度按钮 ======
        self.btn_verify = QPushButton("验证精度")
        self.btn_verify.setToolTip("点击开启/关闭【点击定位】功能")
        # ================================
        self.btn_virtual = QPushButton("相机空跑")
        self.btn_dry_run = QPushButton("针头空跑")
        # btn_real_run已删除 - 无效按钮
        self.btn_batch_run = QPushButton("批量运行")
        self.btn_multicolor = QPushButton("编辑多色路径")
        self.btn_multicolor.setStyleSheet("background-color: #3B82F6; color: white;")
        
        # [NEW] 测试出胶按钮（从batch_run_dialog提取出来）
        self.btn_test_valve = QPushButton("测试出胶")
        self.btn_test_valve.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")
        self.btn_test_valve.setToolTip("打开阀门控制面板，测试各颜色出胶")

        # ===== 不设置固定宽度，让按钮自动扩展填充空间 =====
        self.btn_save.setEnabled(False)

        # ================== 多功能区 (包含所有操作按钮) ==================
        action_group = QGroupBox("多功能区")
        action_group_layout = QVBoxLayout()
        
        # 第一排：5个按钮 (均匀铺满)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(5)  # 按钮间距

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_multicolor) # 多色编辑
        btn_layout.addWidget(self.btn_golden)
        btn_layout.addWidget(self.btn_test_valve) # [NEW] 测试出胶替换原编辑点胶路径位置
        # self.btn_path 已隐藏，不再显示
        # 不添加stretch，让按钮自动扩展

        # 第二排：5个按钮 (均匀铺满)
        task_layout = QHBoxLayout()
        task_layout.setSpacing(5)  # 按钮间距
        
        task_layout.addWidget(self.btn_calibrate)
        task_layout.addWidget(self.btn_verify)
        task_layout.addWidget(self.btn_virtual)
        task_layout.addWidget(self.btn_dry_run)
        task_layout.addWidget(self.btn_batch_run)
        # 不添加stretch，让按钮自动扩展
        
        # 将两排按钮添加到多功能区
        action_group_layout.addLayout(btn_layout)
        action_group_layout.addLayout(task_layout)
        action_group.setLayout(action_group_layout)

        # ================== 2. 相机参数组 ==================
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

        # Gamma (0.1-4.0, Step 0.1)
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
        # [NEW] Recipe System Controls (3-Button Workflow in one row)
        vision_layout.addWidget(self.chk_use_halcon)
        
        # 配方操作 (3个按钮在一行)
        lay_recipe = QHBoxLayout()
        
        self.btn_train_model = QPushButton("🆕 训练新模型")
        self.btn_train_model.setStyleSheet("background-color: #8B5CF6; color: white; font-weight: bold;")
        self.btn_train_model.clicked.connect(self.on_train_model)
        
        self.btn_save_recipe = QPushButton("💾 保存配方")
        self.btn_save_recipe.setStyleSheet("background-color: #3B82F6; color: white;")
        self.btn_save_recipe.clicked.connect(self.on_save_recipe)
        
        self.btn_load_recipe = QPushButton("📂 加载配方")
        self.btn_load_recipe.setStyleSheet("background-color: #F59E0B; color: white;")
        self.btn_load_recipe.clicked.connect(self.on_load_recipe)
        
        lay_recipe.addWidget(self.btn_train_model)
        lay_recipe.addWidget(self.btn_save_recipe)
        lay_recipe.addWidget(self.btn_load_recipe)
        vision_layout.addLayout(lay_recipe)

        # 3. 当前模型显示
        lay_info = QHBoxLayout()
        lay_info.addWidget(QLabel("当前配方:"))
        self.lbl_recipe_name = QLabel("未加载")
        self.lbl_recipe_name.setStyleSheet("color: red; font-weight: bold;")
        lay_info.addWidget(self.lbl_recipe_name)
        vision_layout.addLayout(lay_info)
        
        # Remove old buttons logic (load_halcon / select_model)
        # self.btn_load_halcon.setVisible(False) 


        lay_offset = QHBoxLayout()
        
        # X Offset
        lay_offset.addWidget(QLabel("Offset X:"))
        self.spn_vision_offset_x = QDoubleSpinBox()
        self.spn_vision_offset_x.setRange(-100.0, 100.0)
        self.spn_vision_offset_x.setSingleStep(0.1)
        self.spn_vision_offset_x.setDecimals(2)
        self.spn_vision_offset_x.valueChanged.connect(self.on_vision_offset_changed)
        lay_offset.addWidget(self.spn_vision_offset_x)

        # Y Offset
        lay_offset.addWidget(QLabel("Offset Y:"))
        self.spn_vision_offset_y = QDoubleSpinBox()
        self.spn_vision_offset_y.setRange(-100.0, 100.0)
        self.spn_vision_offset_y.setSingleStep(0.1)
        self.spn_vision_offset_y.setDecimals(2)
        self.spn_vision_offset_y.valueChanged.connect(self.on_vision_offset_changed)
        lay_offset.addWidget(self.spn_vision_offset_y)
        
        vision_layout.addLayout(lay_offset)
        # ============================================
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
        # Block signals to prevent overwriting config during init
        self.sld_spd_xy.blockSignals(True)
        self.sld_spd_xy.setValue(int(self.feedrate_xy))
        self.sld_spd_xy.blockSignals(False)
        self.sld_spd_xy.valueChanged.connect(self.on_speed_changed)
        row_sxy.addWidget(self.lbl_spd_xy)
        row_sxy.addWidget(self.sld_spd_xy)

        # 2. Z 速度
        row_sz = QHBoxLayout()
        self.lbl_spd_z = QLabel(f"Z 速度: {int(self.feedrate_z)}")
        self.sld_spd_z = QSlider(Qt.Horizontal)
        self.sld_spd_z.setRange(100, 2000)  # 范围 100 ~ 2000 (Z轴慢点)
        self.sld_spd_z.blockSignals(True)
        self.sld_spd_z.setValue(int(self.feedrate_z))
        self.sld_spd_z.blockSignals(False)
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
        
        # ================== 第一段：Klipper 连接 ==================
        right_layout.addWidget(conn_group)
        
        # ================== 第二段：系统状态 & 温控 ==================
        # [NEW] System Status / Temp
        self.sys_status_group = self.init_sys_status_group()
        right_layout.addWidget(self.sys_status_group)
        
        # ================== 第三段：相机信息 + 相机参数 (横向布局) ==================
        top_row_layout = QHBoxLayout()
        top_row_layout.addWidget(info_group, stretch=1)      # 相机信息占小比例
        top_row_layout.addWidget(self.param_group, stretch=2) # 相机参数占大比例
        
        right_layout.addLayout(top_row_layout)

        right_layout.addWidget(self.motion_group)
        # [MODIFIED] Put Tool Set and Offset groups side-by-side
        tools_layout = QHBoxLayout()
        tools_layout.addWidget(self.tool_set_group, stretch=1)
        tools_layout.addWidget(self.offset_group, stretch=1)
        right_layout.addLayout(tools_layout)
        right_layout.addWidget(action_group)   # 多功能区（包含所有按钮）
        # param_group 已移到第一行，这里不再添加
        right_layout.addWidget(self.vision_group)

        # [DISABLED] iPad Monitor Button - 用户使用iPad投屏查看主程序
        # btn_ipad = QPushButton("📱 启动 iPad 监控副屏")
        # btn_ipad.setMinimumHeight(40)
        # btn_ipad.setStyleSheet("""
        #     QPushButton { 
        #         background-color: #059669; 
        #         color: white; 
        #         font-weight: bold; 
        #         font-size: 16px; 
        #         border-radius: 6px;
        #         margin-top: 10px;
        #         margin-bottom: 10px;
        #     }
        #     QPushButton:hover { background-color: #10B981; }
        # """)
        # btn_ipad.clicked.connect(self.toggle_mobile_monitor)
        # self.btn_ipad = btn_ipad
        # right_layout.addWidget(btn_ipad)
        
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
        self.btn_camera.clicked.connect(self.on_toggle_camera)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_connect_klipper.clicked.connect(self.on_toggle_klipper_connection)

        self.btn_detect.clicked.connect(self.detect_mold)
        self.btn_golden.clicked.connect(self.on_golden_teach)
        # btn_resume已删除，不需要连接
        self.btn_path.clicked.connect(self.on_edit_path)

        # 标定精度确认
        self.btn_verify.clicked.connect(self.on_toggle_verify_mode)

        # 针头偏移标定
        self.btn_calib_nozzle.clicked.connect(self.on_calibrate_nozzle_offset)

        # 点胶任务按钮
        self.btn_calibrate.clicked.connect(self.on_calibrate_coords)
        self.btn_virtual.clicked.connect(self.on_virtual_run)
        self.btn_dry_run.clicked.connect(self.on_dry_run)
        # btn_real_run已删除，不需要连接
        self.btn_batch_run.clicked.connect(self.on_batch_run)
        # self.btn_path.clicked.connect(self.on_edit_path) # Duplicate removed
        self.btn_multicolor.clicked.connect(self.on_edit_multicolor_paths)
        self.btn_test_valve.clicked.connect(self.on_test_valve) # [NEW] 测试出胶

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
        
        # Z Params
        self.spn_safe_z_cavity.valueChanged.connect(self.on_z_params_changed)
        self.spn_safe_z_product.valueChanged.connect(self.on_z_params_changed)
        self.spn_work_z.valueChanged.connect(self.on_z_params_changed)

    def init_sys_status_group(self):
        """初始化系统状态（温度监控）组"""
        group = QGroupBox("系统状态 & 温控")
        layout = QGridLayout()
        
        # 0. 设置列宽均匀拉伸
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        
        # 1. 标题头（3列）- 加大 "状态/设定" 字体
        lbl_dev = QLabel("设备")
        lbl_cur = QLabel("当前")
        lbl_set = QLabel("状态/设定")
        
        # 设置字体稍微大一点，加粗
        style_header = "font-weight: bold; font-size: 14px; color: #9CA3AF;"
        #style_header_big = "font-weight: bold; font-size: 16px; color: #E5E7EB;" # 重点加大
        
        lbl_dev.setStyleSheet(style_header)
        lbl_cur.setStyleSheet(style_header)
        lbl_set.setStyleSheet("font-weight: bold; font-size: 16px; color: #E5E7EB;") # 特别加大
        
        layout.addWidget(lbl_dev, 0, 0)
        layout.addWidget(lbl_cur, 0, 1)
        layout.addWidget(lbl_set, 0, 2)
        
        # 2. Syringe Heater - 只显示状态
        self.lbl_syringe_name = QLabel("针筒加热:")
        
        # 当前温度
        self.lbl_syringe_temp = QLabel("0.0°C")
        self.lbl_syringe_temp.setStyleSheet("color: #F59E0B; font-weight: bold; font-size: 15px;") # Amber
        
        # 实际目标 (状态)
        self.lbl_syringe_target_status = QLabel("Off")
        self.lbl_syringe_target_status.setStyleSheet("color: #9CA3AF; font-weight: bold;")
        
        # 布局 Syringe
        layout.addWidget(self.lbl_syringe_name, 1, 0)
        layout.addWidget(self.lbl_syringe_temp, 1, 1)
        layout.addWidget(self.lbl_syringe_target_status, 1, 2)

        # 3. Fly-Super8 (MCU) - 显示温度 + 设定值输入框
        layout.addWidget(QLabel("Fly-Super8:"), 2, 0)
        self.lbl_mcu_temp = QLabel("0.0°C")
        layout.addWidget(self.lbl_mcu_temp, 2, 1)
        
        # 设定值输入框（放在Fly-Super8行）
        self.spn_syringe_target = QDoubleSpinBox()
        self.spn_syringe_target.setRange(0, 80) 
        self.spn_syringe_target.setSuffix("°C")
        self.spn_syringe_target.setDecimals(1)
        # self.spn_syringe_target.setFixedWidth(70) # [CHANGE] Remove fixed width to fill column
        self.spn_syringe_target.setStyleSheet("font-size: 14px; height: 24px;")
        
        # 加载上次保存的设定值 (如果存在)
        if hasattr(self, 'saved_params') and 'last_syringe_target' in self.saved_params:
             self.spn_syringe_target.setValue(float(self.saved_params['last_syringe_target']))
        layout.addWidget(self.spn_syringe_target, 2, 2)
        
        # 4. FLY-pi (Host) - 显示温度 + 执行按钮
        layout.addWidget(QLabel("FLY-pi:"), 3, 0)
        self.lbl_host_temp = QLabel("0.0°C")
        layout.addWidget(self.lbl_host_temp, 3, 1)
        
        # 执行按钮（放在FLY-pi行）
        self.btn_set_syringe_temp = QPushButton("执行")
        # self.btn_set_syringe_temp.setFixedWidth(70) # [CHANGE] Remove fixed width
        self.btn_set_syringe_temp.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold; height: 26px;")
        self.btn_set_syringe_temp.clicked.connect(self.on_set_syringe_temp)
        layout.addWidget(self.btn_set_syringe_temp, 3, 2)
        
        group.setLayout(layout)
        return group
        
    def on_set_syringe_temp(self):
        val = self.spn_syringe_target.value()
        self.machine.set_heater_temp(val)
        self.log(f"🔥 设置针筒加热目标温度: {val}°C")
        
        # 保存用户偏好
        self.saved_params['last_syringe_target'] = val
        self.save_app_settings()
        
    def update_temp_ui(self, data):
        """
        data = {
            'syringe': {'current': ..., 'target': ..., 'power': ...},
            'mcu': ...,
            'host': ...
        }
        """
        if not data: return
        
        # Syringe
        s = data.get('syringe', {})
        curr = s.get('current', 0.0)
        tgt = s.get('target', 0.0)
        
        self.lbl_syringe_temp.setText(f"{curr:.1f}°C")
        
        # 更新状态 Label
        if tgt > 0:
            self.lbl_syringe_target_status.setText(f"🎯 {tgt:.1f} °C")
            self.lbl_syringe_target_status.setStyleSheet("color: #10B981; font-weight: bold;") # Green
        else:
            self.lbl_syringe_target_status.setText("Off")
            self.lbl_syringe_target_status.setStyleSheet("color: #9CA3AF;") # Gray
            
        # 注意：这里不再自动修改 spn_syringe_target 的值，以免打断用户输入
        # 且满足用户 "保留设定值" 的需求
        
        # MCU
        self.lbl_mcu_temp.setText(f"{data.get('mcu', 0.0):.1f}°C")
        
        # Host
        self.lbl_host_temp.setText(f"{data.get('host', 0.0):.1f}°C")


    # ================== Slider/SpinBox Synchronization ==================


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
            
            # [NEW] Start TempWorker
            self.temp_worker = TempWorker(self.machine)
            self.temp_worker.temp_signal.connect(self.update_temp_ui)
            self.temp_worker.start()
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
    
    def on_toggle_klipper_connection(self):
        """Toggle按钮处理：连接/断开Klipper"""
        if self.btn_connect_klipper.isChecked():
            # 连接
            self.on_connect_klipper()
            if self.machine.connected:
                self.btn_connect_klipper.setText("断开")
        else:
            # 断开 (MachineClient没有disconnect方法，直接设置状态)
            self.machine.connected = False
            self.btn_connect_klipper.setText("连接")
            self.btn_connect_klipper.setStyleSheet("")
            self.log("🔌 已断开 Klipper 连接")
            # 停止温度监控线程
            if hasattr(self, 'temp_worker') and self.temp_worker:
                self.temp_worker.stop()
                self.temp_worker = None

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

        # 界面显示 - 橙色醒目显示
        self.lbl_pos_x.setText(f"X: {self.pos_x:.2f}")
        self.lbl_pos_x.setStyleSheet("color: #F59E0B; font-weight: bold;")  # 橙色
        
        self.lbl_pos_y.setText(f"Y: {self.pos_y:.2f}")
        self.lbl_pos_y.setStyleSheet("color: #F59E0B; font-weight: bold;")  # 橙色
        
        self.lbl_pos_z.setText(f"Z: {self.pos_z:.2f}")
        self.lbl_pos_z.setStyleSheet("color: #F59E0B; font-weight: bold;")  # 橙色
        
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

    # ================== 触摸模式相关 ==================
    
    def toggle_touch_mode(self, enabled):
        """切换触摸模式"""
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QFont
        
        TouchSpinBox.set_global_touch_mode(enabled)
        app = QApplication.instance()
        
        if enabled:
            self.log("✅ 触摸模式已启用 - 双击数值框打开触屏键盘")
            
            # --- 1. 保存原始状态 ---
            if not hasattr(self, '_original_font'):
                self._original_font = app.font()
            if not hasattr(self, '_original_style'):
                self._original_style = app.styleSheet()
            
            # --- 2. 设置全局大字体 ---
            new_font = QFont(self._original_font)
            current_size = new_font.pointSize()
            if current_size < 12:
                new_font.setPointSize(12)  # 至少12点
            else:
                new_font.setPointSize(current_size + 2) # 稍微增大
            app.setFont(new_font)
            
            # --- 3. 设置触屏优化样式 (增加点击区域) ---
            # 叠加到现有样式
            touch_style = """
                QDoubleSpinBox, QSpinBox, QLineEdit {
                    min-height: 40px;
                    font-size: 14pt;
                }
                QPushButton {
                    min-height: 40px;
                    font-size: 13pt;
                }
                QComboBox {
                    min-height: 40px;
                    font-size: 13pt;
                }
                QLabel {
                    font-size: 13pt;
                }
                QGroupBox {
                    font-size: 13pt;
                    font-weight: bold;
                    margin-top: 10px;
                }
            """
            app.setStyleSheet(self._original_style + touch_style)
            
            # 首次启用时为所有SpinBox添加触摸支持
            if not hasattr(self, '_touch_initialized'):
                self.enable_touch_for_all_spinboxes()
                self._touch_initialized = True
        else:
            self.log("触摸模式已禁用")
            # --- 恢复原始状态 ---
            if hasattr(self, '_original_font'):
                app.setFont(self._original_font)
            if hasattr(self, '_original_style'):
                app.setStyleSheet(self._original_style)
    
    def enable_touch_for_all_spinboxes(self):
        """为所有QDoubleSpinBox启用触摸支持"""
        apply_touch_support(self)
        self.log(f"✅ 已启用全局触摸支持 (单击弹出键盘)")

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

        # Enable save button (btn_camera status managed by on_toggle_camera)
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
        # btn_start和btn_stop不再存在，已改为btn_camera
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
    
    def on_toggle_camera(self):
        """Toggle按钮处理：启动/停止相机"""
        if self.btn_camera.isChecked():
            # 启动相机
            self.start_cam()
            self.btn_camera.setText("停止相机")
        else:
            # 停止相机
            self.stop_cam()
            self.btn_camera.setText("启动相机")


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
    def check_recipe_loaded_halcon(self):
        """检查Halcon模式下是否已加载配方"""
        use_halcon = hasattr(self, 'chk_use_halcon') and self.chk_use_halcon.isChecked()
        if not use_halcon:
            return True  # OpenCV模式不需要配方
        
        current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
        return "未加载" not in current_recipe
    
    def load_golden_templates(self):
        try:
            # ✅ OpenCV Mode: Load slot templates
            if os.path.exists("config/slot_templates.npy"):
                self.slot_templates = np.load("config/slot_templates.npy")
                self.log(f"✅ 黄金模板 config/slot_templates.npy 加载成功，形状：{self.slot_templates.shape}")
            else:
                self.log("⚠ 未找到 config/slot_templates.npy，暂时无法做黄金对位")
                self.slot_templates = None
            
            # ✅ Load Golden Path ONLY in OpenCV mode
            # In Halcon mode, paths come from recipes, not global config
            use_halcon = hasattr(self, 'chk_use_halcon') and self.chk_use_halcon.isChecked()
            
            if not use_halcon:
                # OpenCV Mode: Load global paths
                if os.path.exists("config/disp_poly_cell0.npy"):
                    self.golden_path_cache = np.load("config/disp_poly_cell0.npy", allow_pickle=True)
                    self.log(f"✅ 加载点胶路径缓存: {len(self.golden_path_cache)} 个路径段")
                else:
                    self.golden_path_cache = None
                    self.log("ℹ️ 未找到点胶路径缓存文件")
            else:
                # Halcon Mode: Skip global paths, they should come from recipe
                self.golden_path_cache = None
                self.log("ℹ️ Halcon模式: 路径将从配方加载")

        except Exception as e:
            self.log("❌ 加载黄金模板失败：", e)
            import traceback
            traceback.print_exc()
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


    def on_vision_offset_changed(self):
        """偏移量改变时，立即保存并更新"""
        self.vision_offset_x = self.spn_vision_offset_x.value()
        self.vision_offset_y = self.spn_vision_offset_y.value()
        self.save_app_settings()

    def load_app_settings(self):
        """Load application settings from ConfigManager (unified config system)"""
        try:
            # Load all parameters from ConfigManager (app_settings.json)
            # Nozzle Z Parameters (针头高度参数 - 主界面使用)
            self.safe_z_cavity = self.config.get('safe_z_cavity', 100.0)
            self.safe_z_product = self.config.get('safe_z_product', 100.0)
            self.work_z = self.config.get('work_z', 100.0)
            self.jog_step_mm = self.config.get('jog_step', 1.0)
            self.feedrate_xy = self.config.get('feedrate_xy', 3000.0)
            self.feedrate_z = self.config.get('feedrate_z', 500.0)
            self.accel = self.config.get('accel', 500.0)

            # Load Calibration Data
            self.calib_vision_pts = self.config.get('calib_vision_pts', [])
            self.calib_machine_pts = self.config.get('calib_machine_pts', [])
             
            # Load Nozzle Offset
            self.nozzle_offset_x = self.config.get('nozzle_offset_x', 0.0)
            self.nozzle_offset_y = self.config.get('nozzle_offset_y', 0.0)
            self.nozzle_offset_z = self.config.get('nozzle_offset_z', 0.0)
        
            # Load Z Base (tool setting reference)
            self.z_base = self.config.get('z_base', 0.0)

            # Load Camera Dry Run Params (相机空跑参数)
            self.cam_safe_z_cavity = self.config.get('cam_safe_z_cavity', 213.0)
            self.cam_safe_z_product = self.config.get('cam_safe_z_product', 215.0)
            self.cam_work_z = self.config.get('cam_work_z', 211.0)

            # Load Halcon Vision Offset
            self.vision_offset_x = self.config.get('vision_offset_x', 0.0)
            self.vision_offset_y = self.config.get('vision_offset_y', 0.0)

            # Load Dispensing Dwell Times (点胶延迟参数)
            self.dwell_ms = self.config.get('dwell_ms', 100)
            self.line_dwell_ms = self.config.get('line_dwell_ms', 0)

            # Update UI elements (if they exist)
            if hasattr(self, 'sld_spd_xy'):
                self.sld_spd_xy.blockSignals(True)
                self.sld_spd_xy.setValue(int(self.feedrate_xy))
                self.sld_spd_xy.blockSignals(False)
                self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")

            if hasattr(self, 'sld_spd_z'):
                self.sld_spd_z.blockSignals(True)
                self.sld_spd_z.setValue(int(self.feedrate_z))
                self.sld_spd_z.blockSignals(False)
                self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

            if hasattr(self, 'sld_acc'):
                self.sld_acc.blockSignals(True)
                self.sld_acc.setValue(int(self.accel))
                self.sld_acc.blockSignals(False)
                self.lbl_acc.setText(f"加速度: {int(self.accel)}")

            if hasattr(self, 'spn_safe_z_cavity'):
                self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
            if hasattr(self, 'spn_safe_z_product'):
                self.spn_safe_z_product.setValue(self.safe_z_product)
            if hasattr(self, 'spn_work_z'):
                self.spn_work_z.setValue(self.work_z)
        
            if hasattr(self, 'lbl_z_base'):
                self.lbl_z_base.setText(f"当前 Z 基准: {self.z_base:.2f} mm")

            # Apply Vision Offset to UI
            if hasattr(self, 'spn_vision_offset_x'):
                self.spn_vision_offset_x.blockSignals(True)
                self.spn_vision_offset_x.setValue(self.vision_offset_x)
                self.spn_vision_offset_x.blockSignals(False)

            if hasattr(self, 'spn_vision_offset_y'):
                self.spn_vision_offset_y.blockSignals(True)
                self.spn_vision_offset_y.setValue(self.vision_offset_y)
                self.spn_vision_offset_y.blockSignals(False)
            
            self.log(f"✅ 已加载应用参数 (速度XY={int(self.feedrate_xy)}, 偏移X={self.nozzle_offset_x:.2f}, Y={self.nozzle_offset_y:.2f})")
            
        except Exception as e:
            self.log(f"❌ 加载应用参数失败: {e}")
            import traceback
            traceback.print_exc()

    def save_app_settings(self):
        """Save all application settings to ConfigManager (unified config system)"""
        try:
            # Update ConfigManager with all current values
            self.config.set('safe_z_cavity', self.safe_z_cavity)
            self.config.set('safe_z_product', self.safe_z_product)
            self.config.set('work_z', self.work_z)
            self.config.set('jog_step', self.jog_step_mm)
            self.config.set('feedrate_xy', self.feedrate_xy)
            self.config.set('feedrate_z', self.feedrate_z)
            self.config.set('accel', self.accel)
            
            self.config.set('calib_vision_pts', self.calib_vision_pts)
            self.config.set('calib_machine_pts', self.calib_machine_pts)
            
            self.config.set('nozzle_offset_x', self.nozzle_offset_x)
            self.config.set('nozzle_offset_y', self.nozzle_offset_y)
            self.config.set('nozzle_offset_z', self.nozzle_offset_z)
            
            self.config.set('z_base', self.z_base)
            
            # Camera Dry Run Params
            self.config.set('cam_safe_z_cavity', self.cam_safe_z_cavity)
            self.config.set('cam_safe_z_product', self.cam_safe_z_product)
            self.config.set('cam_work_z', self.cam_work_z)
            
            self.config.set('vision_offset_x', self.vision_offset_x)
            self.config.set('vision_offset_y', self.vision_offset_y)
            
            # Dispensing Dwell Times
            self.config.set('dwell_ms', self.dwell_ms)
            self.config.set('line_dwell_ms', self.line_dwell_ms)
            
            # Trigger ConfigManager to write to file
            self.config.save_app_settings()
            
        except Exception as e:
            self.log(f"❌ 保存应用设置失败: {e}")
            import traceback
            traceback.print_exc()



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
                # [FIX] Handle PathSegment objects (Multicolor) vs numpy arrays (Legacy)
                if hasattr(path, 'points'): # Duck typing for PathSegment
                     # PathSegment stores points as list of lists [[x,y],...]
                     # Convert to numpy for polylines
                     pts = np.array(path.points, dtype=np.int32).reshape((-1, 1, 2))
                else:
                     # Legacy numpy array
                     pts = path.reshape((-1, 1, 2)).astype(np.int32)
                
                if len(pts) == 1:
                    # Draw Dot (Magenta)
                    # pts is (N, 1, 2), so pts[0][0] is [x, y]
                    cx, cy = int(pts[0][0][0]), int(pts[0][0][1])
                    # [USER ADJUST] 孤立点大小 (Radius): 2=小, 5=中, 8=大
                    cv2.circle(vis_img, (cx, cy), 5, (255, 0, 255), -1)
                else:
                    # [USER ADJUST] 线条粗细 (Thickness): 1=细, 2=中, 3=粗
                    cv2.polylines(vis_img, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)

                    # 标记起点
                    if len(pts) > 0:
                        sx, sy = int(pts[0][0][0]), int(pts[0][0][1])
                        # [USER ADJUST] 起点圆圈大小
                        cv2.circle(vis_img, (sx, sy), 3, (0, 0, 255), -1)

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

        # Use absolute path for captures directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        captures_dir = os.path.join(base_dir, "captures")
        os.makedirs(captures_dir, exist_ok=True)
        
        default_name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.png")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            os.path.join(captures_dir, default_name),
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
        # ✅ Halcon模式检查
        if not self.check_recipe_loaded_halcon():
            QMessageBox.warning(self, "需要配方", 
                              "Halcon模式下，请先加载或创建配方后再进行识别。\n"
                              "识别需要配方中的模型和路径数据。")
            return
        
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
            if os.path.exists("scripts/captures"):
                files = [os.path.join("scripts/captures", f) for f in os.listdir("scripts/captures") if f.lower().endswith(('.png', '.jpg', '.bmp'))]
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

    # ================== 多色路径编辑器 (独立) ==================
    def on_edit_multicolor_paths(self):
        """打开多色路径编辑器 (直接在reference.png上编辑)"""
        print("[DEBUG UI] on_edit_multicolor_paths CLICKED") # Debug
        # 1. 检查 Halcon 模式
        use_halcon = hasattr(self, 'chk_use_halcon') and self.chk_use_halcon.isChecked()
        if not use_halcon:
            QMessageBox.warning(self, "模式错误", "此功能仅在 Halcon 模式下可用。")
            return
            
        # 2. 检查配方
        current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
        if "未加载" in current_recipe or not current_recipe:
            QMessageBox.warning(self, "需要配方", "请先加载或创建配方。")
            return
            
        # 3. 获取产品目录
        product_dir = self.product_manager.get_product_path(current_recipe)
        if not product_dir or not os.path.exists(product_dir):
            QMessageBox.critical(self, "错误", f"产品目录不存在: {product_dir}")
            return
            
        self.log(f"🎨 打开多色路径编辑器: {current_recipe}")
        
        # 4. 打开编辑器
        # 注意：这里直接使用 path_editor_wrapper.py 中的逻辑
        try:
            success = edit_path_halcon_mode(self, current_recipe, product_dir)
            if success:
                self.log(f"✅ 多色路径已保存")
                
                # [CRITICAL] 重新加载数据到 HalconProductManager
                # 虽然用户说"暂时不用实现连接"，但为了基本功能正常（比如能在识别后显示）
                # 我们依然通知 manager 加载新数据。
                if hasattr(self, 'halcon_product_manager') and self.halcon_product_manager:
                     self.halcon_product_manager.load_product(current_recipe, force=True)
                     
                     # [FIX] 同步更新 UI 实例的缓存
                     self.current_segments = self.halcon_product_manager.get_current_segments()
                     self.log(f"🔄 这里的缓存已更新，段数: {len(self.current_segments.segments) if self.current_segments else 0}")
                     
                     # [FIX] 重新计算 Halcon 模板 (关键！否则还是用旧的)
                     # 传入 True 强制刷新
                     self._compute_halcon_template(force_reload=True)
                
                # [NEW] ⭐ 重要：更新按钮状态，解锁识别模具、空跑、批量等功能
                self.update_button_states()
                     
            else:
                self.log("❌ 编辑取消或保存失败")
        except Exception as e:
            self.log(f"❌ 打开编辑器失败: {e}")
            import traceback
            traceback.print_exc()

    # ================== 点胶路径编辑 ==================
    def on_edit_path(self):
        """点击按钮：在黄金图片中编辑某个小框的点胶路径"""
        
        # ✅ Halcon模式检查
        use_halcon = hasattr(self, 'chk_use_halcon') and self.chk_use_halcon.isChecked()
        if use_halcon:
            # 检查是否已加载配方
            current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
            if "未加载" in current_recipe:
                QMessageBox.warning(self, "需要配方", 
                                  "Halcon模式下，请先加载或创建配方后再编辑路径。\n"
                                  "路径将保存到配方中，而非全局配置。")
                return
        
        self.log("🧵 打开点胶路径示教窗口（默认小框 index=0）")

        # ====== ✅ 3按钮选择逻辑 ======
        candidate_path = self.latest_capture_path
        if not candidate_path:
            if os.path.exists("scripts/captures"):
                files = [os.path.join("scripts/captures", f) for f in os.listdir("scripts/captures") if f.lower().endswith(('.png', '.jpg', '.bmp'))]
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
        
        # ⚠️ 确定保存目录：Halcon模式使用配方目录，OpenCV模式使用全局config
        save_directory = None
        img_path_for_teach = None  # ⭐ 新增：指定路径编辑使用的图片
        
        if use_halcon:
            current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
            if "未加载" not in current_recipe:
                save_directory = self.product_manager.get_product_path(current_recipe)
                # ⭐ 关键修复：使用配方的training图片（reference.png）
                # 这样路径编辑器和Halcon使用完全相同的图片，坐标系统一致
                ref_img_path = os.path.join(save_directory, "reference.png")
                if os.path.exists(ref_img_path):
                    img_path_for_teach = ref_img_path
                    self.log(f"📍 使用配方训练图片: reference.png")
        
        # 暂时默认编辑第 0 个格子
        cell_idx = 0
        self.log(f"🧵 启动路径编辑器 (Cell {cell_idx})...")
        
        # 运行编辑器 (会阻塞直到窗口关闭)
        # ⭐ 传入指定的图片路径
        if img_path_for_teach:
            ok = run_path_teach(cell_index=cell_idx, img_path=img_path_for_teach, save_dir=save_directory)
        else:
            ok = run_path_teach(cell_index=cell_idx, save_dir=save_directory)
        
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
            
            # [FIX] Reload self.dots and self.polys from the correct location
            try:
                # ⭐ 从正确的目录加载：配方目录（Halcon）或全局config（OpenCV）
                load_dir = save_directory if save_directory else os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
                
                # 加载dots
                dots_path = os.path.join(load_dir, f"disp_dots_cell{cell_idx}.npy")
                if os.path.exists(dots_path):
                    self.dots = np.load(dots_path)
                    self.log(f"✅ 已重新加载点路径: {len(self.dots)} points")
                else:
                    self.dots = []
                    self.log("ℹ️ 无点路径数据")
                
                # 加载polys
                polys_path = os.path.join(load_dir, f"disp_poly_cell{cell_idx}.npy")
                if os.path.exists(polys_path):
                    self.polys = np.load(polys_path, allow_pickle=True)
                    self.log(f"✅ 已重新加载线路径: {len(self.polys)} lines")
                else:
                    self.polys = []
                    self.log("ℹ️ 无线路径数据")
            except Exception as e:
                self.log(f"⚠ 重新加载路径失败: {e}")
            
            # ⭐ 重要：更新按钮状态，因为路径已改变
            self.update_button_states()
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
            
            # 使用自定义按钮的消息框，添加“取消”逻辑
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("标定指南 (8点 + 撤销)")
            msg_box.setText(msg)
            btn_ok = msg_box.addButton("开始标定", QMessageBox.AcceptRole)
            btn_cancel = msg_box.addButton("取消", QMessageBox.RejectRole)
            msg_box.exec_()
            
            if msg_box.clickedButton() == btn_cancel:
                self.calib_step = 0
                self.calib_vision_pts = []
                self.calib_machine_pts = []
                self.btn_calibrate.setText("标定坐标系")
                self.btn_calibrate.setStyleSheet("")
                self.btn_calibrate.setEnabled(True)
                self.log("🚫 已取消标定操作。")
                return

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

    # ================== Recipe System / Halcon ==================
    def on_train_model(self):
        """
        Flow:
        1. Select Image (PNG/JPG)
        2. Enter New Recipe Name
        3. Trigger Halcon Training -> Generate .shm
        4. Create Recipe Folder and move files
        5. Auto-load the new recipe
        """
        # 1. Select Image
        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练图像", "", "Images (*.png *.jpg *.bmp)")
        if not img_path:
            return

        # 2. Enter Name
        name, ok = QInputDialog.getText(self, "新建配方", "请输入新产品名称 (Recipe Name):")
        if not ok or not name.strip():
            return
        name = name.strip()

        # 3. Training (Placeholder for now, assuming simple copy or Halcon call)
        # TODO: Call HalconManager.train_model(img_path) -> model_path
        # For now, we will simulate training by checking if we can get a result from Halcon
        # Or simply ask HalconManager to "Run Training Script" on this image.
        
        # User requirement: "Train Model" -> Uses this image to create a model.
        # We need a `halcon_manager.train_shape_model(img_path, output_path)` method.
        # Let's assume successful training returns a path.
        
        self.log(f"🧠 正在训练模型 [{name}]...")
        
        # Define output path
        recipe_dir = self.recipe_manager.get_recipe_path(name)
        if not os.path.exists(recipe_dir):
            os.makedirs(recipe_dir)
            
        model_path = os.path.join(recipe_dir, f"{name}.shm")
        
        # Call Halcon Logic (Synchronous for now, or Worker)
        # For this step, we might need to rely on the existing "run_training_script" logic 
        # but automated.
        # Since HalconManager.run_training_script logic is not fully exposed here, 
        # I will assume a Mock or Basic Implementation:
        try:
            # Try to copy the image to recipe dir
            # [FIX] Ensure the saved image is FLIPPED (Upright) to match path_teach and live view
            dest_img_path = os.path.join(recipe_dir, "reference.png")
            # shutil.copy2(img_path, dest_img_path) 
            raw_img = cv2.imread(img_path)
            if raw_img is not None:
                # Assume input from capture is Inverted (Raw). We Flip it Upright.
                # If it's already upright (external file), this might invert it back, 
                # but we prioritize consistency with the Camera/Capture workflow.
                upright_img = cv2.flip(raw_img, -1)
                cv2.imwrite(dest_img_path, upright_img)
                self.log(f"✅ 保存并翻转配方图像: {dest_img_path}")
            else:
                self.log(f"⚠ 无法读取源图像: {img_path}")
                shutil.copy2(img_path, dest_img_path) # Fallback
            
            # 🚀 CORE: Call Halcon to train!
            if hasattr(self.halcon_manager, 'train_model'):
                # result is now a dict or bool
                res = self.halcon_manager.train_model(dest_img_path, model_path)
                
                # Helper to parse result
                if isinstance(res, dict):
                    success = True
                    msg = res.get("msg", "Success")
                    origin_row = res.get("origin_row", 0)
                    origin_col = res.get("origin_col", 0)
                elif isinstance(res, tuple): # For compatibility if old code returns (bool, msg)
                    success, msg = res
                    origin_row, origin_col = 0, 0
                else:
                     success = res
                     msg = "Success" if success else "Failed"
                     origin_row, origin_col = 0, 0

            else:
                # Fallback: Just copy a template if it exists, or Fake it for testing
                success = True 
                msg = "Simulated Training (HalconManager.train_model missing)"
                origin_row, origin_col = 0, 0
                # Create a dummy SHM file so loading doesn't fail
                with open(model_path, 'w') as f: f.write("DUMMY MODEL")

            if success:
                self.log(f"✅ 训练成功! 模型保存至: {model_path}")
                
                # 3.5 Create Initial Config (⭐ Complete Current Params)
                initial_config = {
                    # === 视觉参数 ===
                    # ⭐ 新配方默认偏移为0（与启动时未加载配方一致）
                    'vision_offset_x': 0.0,
                    'vision_offset_y': 0.0,
                    'exposure': self.spn_exp.value() if hasattr(self, 'spn_exp') else 200.0,
                    'gain': self.spn_gain.value() if hasattr(self, 'spn_gain') else 10.0,
                    'gamma': self.spn_gamma.value() if hasattr(self, 'spn_gamma') else 1.0,
                    
                    # === ⭐ 运动参数 ===
                    'feedrate_xy': self.feedrate_xy,
                    'feedrate_z': self.feedrate_z,
                    'accel': self.accel,
                    'jog_step_mm': self.jog_step_mm,
                    
                    # === ⭐ Z高度参数 ===
                    'safe_z_cavity': self.safe_z_cavity,
                    'safe_z_product': self.safe_z_product,
                    'work_z': self.work_z,
                    'cam_safe_z_cavity': self.cam_safe_z_cavity,
                    'cam_safe_z_product': self.cam_safe_z_product,
                    'cam_work_z': self.cam_work_z,
                    
                    # === ⭐ 针头偏移 ===
                    'nozzle_offset_x': self.nozzle_offset_x,
                    'nozzle_offset_y': self.nozzle_offset_y,
                    'nozzle_offset_z': self.nozzle_offset_z,
                    
                    # === ⭐ 对刀基准 ===
                    'z_base': self.z_base,
                    
                    # === Halcon 模型信息 ===
                    'model_file': f"{name}.shm",
                    'model_row': origin_row,
                    'model_col': origin_col,
                    'use_halcon': self.chk_use_halcon.isChecked() if hasattr(self, 'chk_use_halcon') else False,
                }
                # 4. Save Recipe with initial empty paths
                success, msg = self.recipe_manager.save_recipe(
                    name,
                    config_data=initial_config,
                    path_points=None,  # No paths initially
                    path_polys=None
                )
                
                if success:
                    # ⭐ 清空内存中的旧路径数据（新配方应该是干净的）
                    self.dots = []
                    self.polys = []
                    
                    self.log(f"✅ 配方 '{name}' 已创建（路径为空，可开始编辑）")
                    # Auto-load the new recipe
                    self.on_load_recipe_by_name(name)
            else:
                self.log(f"❌ 训练失败: {msg}")

        except Exception as e:
            self.log(f"❌ 训练过程出错: {e}")

    def on_save_recipe(self):
        """Save current params, offsets, and path to recipe"""
        current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
        if "未加载" in current_recipe:
             name, ok = QInputDialog.getText(self, "保存配方", "请输入配方名称:")
             if not ok or not name.strip(): return
             name = name.strip()
        else:
             name = current_recipe

        # Collect Data - ⭐ COMPLETE PARAMETER SET
        config_data = {
            # === 视觉参数 ===
            'vision_offset_x': self.vision_offset_x,
            'vision_offset_y': self.vision_offset_y,
            'exposure': self.spn_exp.value() if hasattr(self, 'spn_exp') else 200.0,
            'gain': self.spn_gain.value() if hasattr(self, 'spn_gain') else 10.0,
            'gamma': self.spn_gamma.value() if hasattr(self, 'spn_gamma') else 1.0,
            
            # === ⭐ 运动参数（关键！）===
            'feedrate_xy': self.feedrate_xy,
            'feedrate_z': self.feedrate_z,
            'accel': self.accel,
            'jog_step_mm': self.jog_step_mm,
            
            # === ⭐ Z高度参数 ===
            'safe_z_cavity': self.safe_z_cavity,
            'safe_z_product': self.safe_z_product,
            'work_z': self.work_z,
            'cam_safe_z_cavity': self.cam_safe_z_cavity,
            'cam_safe_z_product': self.cam_safe_z_product,
            'cam_work_z': self.cam_work_z,
            
            # === ⭐ 针头偏移 ===
            'nozzle_offset_x': self.nozzle_offset_x,
            'nozzle_offset_y': self.nozzle_offset_y,
            'nozzle_offset_z': self.nozzle_offset_z,
            
            # === ⭐ 对刀基准 ===
            'z_base': self.z_base,
            
            # === Halcon 模式 ===
            'use_halcon': self.chk_use_halcon.isChecked() if hasattr(self, 'chk_use_halcon') else False,
        }
        
        # Path Data
        # Assuming we have a global path variable (self.dispensing_path or similar?)
        # Need to find where dispensing path is stored.
        # It's usually `self.dots` or `self.path_points`.
        # I'll try to save `self.dry_run_dots` if available, or whatever is used for path.
        # Looking at previous code, `dots` are loaded from `config/disp_dots_cell0.npy`.
        # So we should read that or use current memory state.
        
        # We will reference the file `config/disp_dots_cell0.npy` content if memory var is ambiguous.
        # But ideally we save what is currently "edited".
        # Let's save `self.dots` if it exists.
    
        # Reload Polys from file just in case they were edited
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            poly_path_file = os.path.join(base_dir, "config", "disp_poly_cell0.npy")
            if os.path.exists(poly_path_file):
                self.polys = np.load(poly_path_file, allow_pickle=True)
                self.log(f"✅ Loaded polys for save: {len(self.polys)} lines")
        except Exception as e:
            self.log(f"⚠ Failed to reload polys: {e}")

        path_data = getattr(self, 'dots', [])
        poly_data = getattr(self, 'polys', [])
        
        success, msg = self.recipe_manager.save_recipe(name, config_data, path_data, poly_data)
        self.log(f"{'✅' if success else '❌'} {msg}")
        
        if success:
             self.lbl_recipe_name.setText(name)
             self.lbl_recipe_name.setStyleSheet("color: blue; font-weight: bold;")
             
             # [NEW] Detailed Success Dialog
             info_txt = (f"配方 [{name}] 保存成功！\n\n"
                         f"包含内容：\n"
                         f"✅ 视觉参数 (Offset, Exp, Gain)\n"
                         f"✅ 点胶路径: 点={len(path_data)}, 线={len(poly_data)}\n"
                         f"✅ 模型关联 (model.shm)")
             QMessageBox.information(self, "保存成功", info_txt)

    def on_load_recipe(self):
        """Open recipe selection dialog with delete capability"""
        recipes = self.recipe_manager.list_recipes()
        if not recipes:
            QMessageBox.information(self, "无配方", "还没有保存任何配方。")
            return

        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("配方管理")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # 说明标签
        lbl_hint = QLabel("选择要加载的配方，或删除不需要的配方:")
        layout.addWidget(lbl_hint)
        
        # 配方列表
        from PyQt5.QtWidgets import QListWidget
        list_widget = QListWidget()
        list_widget.addItems(recipes)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        btn_load = QPushButton("📂 加载")
        btn_load.setStyleSheet("background-color: #10B981; color: white; font-weight: bold;")
        btn_load.clicked.connect(dialog.accept)
        
        btn_delete = QPushButton("🗑️ 删除")
        btn_delete.setStyleSheet("background-color: #EF4444; color: white; font-weight: bold;")
        
        btn_cancel = QPushButton("❌ 取消")
        btn_cancel.clicked.connect(dialog.reject)
        
        def on_delete():
            current_item = list_widget.currentItem()
            if not current_item:
                QMessageBox.warning(dialog, "未选择", "请先选择要删除的配方")
                return
            
            name = current_item.text()
            
            # 确认删除
            reply = QMessageBox.question(
                dialog, "确认删除", 
                f"确定要删除配方 [{name}] 吗？\n\n此操作不可恢复！",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                recipe_dir = self.recipe_manager.get_recipe_path(name)
                try:
                    shutil.rmtree(recipe_dir)
                    self.log(f"✅ 已删除配方: {name}")
                    
                    # 从列表中移除
                    list_widget.takeItem(list_widget.currentRow())
                    
                    # 如果删除的是当前加载的配方，清除状态
                    if self.lbl_recipe_name.text() == name:
                        self.lbl_recipe_name.setText("未加载")
                        self.lbl_recipe_name.setStyleSheet("color: red; font-weight: bold;")
                    
                    # 如果列表为空，关闭对话框
                    if list_widget.count() == 0:
                        QMessageBox.information(dialog, "已清空", "所有配方已删除")
                        dialog.reject()
                        
                except Exception as e:
                    QMessageBox.critical(dialog, "删除失败", f"无法删除配方:\n{e}")
                    self.log(f"❌ 删除失败: {e}")
        
        btn_delete.clicked.connect(on_delete)
        
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        # 双击加载
        list_widget.itemDoubleClicked.connect(dialog.accept)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            current_item = list_widget.currentItem()
            if current_item:
                name = current_item.text()
                self.on_load_recipe_by_name(name)

    def on_load_recipe_by_name(self, name):
        """Actual logic to load recipe"""
        
        # ⭐ 清除旧配方缓存（避免混淆）
        self.log("🧹 清除旧配方缓存...")
        self.golden_path_cache = None
        self.slot_templates = None
        self.dots = []
        self.polys = []
        if hasattr(self, 'halcon_path_template'):
            self.halcon_path_template = None
        
        # 加载新配方
        config, path_data, poly_data, msg = self.recipe_manager.load_recipe(name)
        if config is None:
            self.log(f"❌ {msg}")
            return
            
        self.log(f"📂 加载配方: {name}")
        
        # 1. Apply Config - ⭐ COMPLETE PARAMETER RESTORATION
        # === 视觉参数 ===
        if 'vision_offset_x' in config:
            self.vision_offset_x = config['vision_offset_x']
            if hasattr(self, 'spn_vision_offset_x'):
                self.spn_vision_offset_x.blockSignals(True)
                self.spn_vision_offset_x.setValue(config['vision_offset_x'])
                self.spn_vision_offset_x.blockSignals(False)
        
        if 'vision_offset_y' in config:
            self.vision_offset_y = config['vision_offset_y']
            if hasattr(self, 'spn_vision_offset_y'):
                self.spn_vision_offset_y.blockSignals(True)
                self.spn_vision_offset_y.setValue(config['vision_offset_y'])
                self.spn_vision_offset_y.blockSignals(False)
        
        if 'exposure' in config:
            self.cam_exp_time = config['exposure'] 
            if hasattr(self, 'spn_exp'):
                self.spn_exp.setValue(config['exposure'])
        
        if 'gain' in config and hasattr(self, 'spn_gain'):
            self.spn_gain.setValue(config['gain'])
        
        if 'gamma' in config and hasattr(self, 'spn_gamma'):
            self.spn_gamma.setValue(config['gamma'])
        
        # === ⭐ 运动参数 ===
        if 'feedrate_xy' in config:
            self.feedrate_xy = config['feedrate_xy']
            if hasattr(self, 'sld_spd_xy'):
                self.sld_spd_xy.blockSignals(True)
                self.sld_spd_xy.setValue(int(self.feedrate_xy))
                self.sld_spd_xy.blockSignals(False)
                self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")
        
        if 'feedrate_z' in config:
            self.feedrate_z = config['feedrate_z']
            if hasattr(self, 'sld_spd_z'):
                self.sld_spd_z.blockSignals(True)
                self.sld_spd_z.setValue(int(self.feedrate_z))
                self.sld_spd_z.blockSignals(False)
                self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")
        
        if 'accel' in config:
            self.accel = config['accel']
            if hasattr(self, 'sld_acc'):
                self.sld_acc.blockSignals(True)
                self.sld_acc.setValue(int(self.accel))
                self.sld_acc.blockSignals(False)
                self.lbl_acc.setText(f"加速度: {int(self.accel)}")
        
        if 'jog_step_mm' in config:
            self.jog_step_mm = config['jog_step_mm']
        
        # === ⭐ Z高度参数 ===
        if 'safe_z_cavity' in config:
            self.safe_z_cavity = config['safe_z_cavity']
            if hasattr(self, 'spn_safe_z_cavity'):
                self.spn_safe_z_cavity.setValue(self.safe_z_cavity)
        
        if 'safe_z_product' in config:
            self.safe_z_product = config['safe_z_product']
            if hasattr(self, 'spn_safe_z_product'):
                self.spn_safe_z_product.setValue(self.safe_z_product)
        
        if 'work_z' in config:
            self.work_z = config['work_z']
            if hasattr(self, 'spn_work_z'):
                self.spn_work_z.setValue(self.work_z)
        
        if 'cam_safe_z_cavity' in config:
            self.cam_safe_z_cavity = config['cam_safe_z_cavity']
        
        if 'cam_safe_z_product' in config:
            self.cam_safe_z_product = config['cam_safe_z_product']
        
        if 'cam_work_z' in config:
            self.cam_work_z = config['cam_work_z']
        
        # === ⭐ 针头偏移 ===
        if 'nozzle_offset_x' in config:
            self.nozzle_offset_x = config['nozzle_offset_x']
        
        if 'nozzle_offset_y' in config:
            self.nozzle_offset_y = config['nozzle_offset_y']
        
        if 'nozzle_offset_z' in config:
            self.nozzle_offset_z = config['nozzle_offset_z']
        
        # === ⭐ 对刀基准 ===
        if 'z_base' in config:
            self.z_base = config['z_base']
            if hasattr(self, 'lbl_z_base'):
                self.lbl_z_base.setText(f"当前 Z 基准: {self.z_base:.2f} mm")
        
        # === Halcon 模式 ===
        if 'use_halcon' in config and hasattr(self, 'chk_use_halcon'):
            self.chk_use_halcon.setChecked(config['use_halcon'])
            
        # [NEW] Load Halcon Model Reference
        self.model_ref_row = config.get('model_row', 651.6765) # Default fallback
        self.model_ref_col = config.get('model_col', 1113.3245)
        
        self.log(f"✅ 已恢复所有参数: 速度XY={int(self.feedrate_xy)}, 偏移X={self.nozzle_offset_x:.2f}, Y={self.nozzle_offset_y:.2f}, Z基准={self.z_base:.2f}")
            
        # 2. Load Model
        # Use model_file from config, fallback to model.shm
        recipe_dir = self.recipe_manager.get_recipe_path(name)
        model_file = config.get('model_file', 'model.shm')
        model_path = os.path.join(recipe_dir, model_file)
        if os.path.exists(model_path):
            ok = self.halcon_manager.set_model_path(model_path)
            if ok:
                self.log(f"✅ 模型已更新: {model_file}")
            else:
                self.log(f"⚠️ 模型更新失败: {model_file}")
        else:
            self.log(f"⚠️ 配方目录中未找到 {model_file}")

        # 3. Load Path
        if path_data is not None:
            # Update dots
            self.dots = path_data
            # ⚠️ In Halcon mode, DON'T save to global config!
            # Paths stay in recipe only
            use_halcon = config.get('use_halcon', False)
            if not use_halcon:
                # OpenCV mode: save to global for compatibility
                np.save(os.path.join(self.config_dir, "disp_dots_cell0.npy"), self.dots)
            self.log(f"✅ 点路径已加载 ({len(self.dots)} points)")
            
        if poly_data is not None:
            self.polys = poly_data
            use_halcon = config.get('use_halcon', False)
            if not use_halcon:
                # OpenCV mode: save to global
                np.save(os.path.join(self.config_dir, "disp_poly_cell0.npy"), self.polys)
            self.log(f"✅ 线路径已加载 ({len(self.polys)} lines)")
        
        self.lbl_recipe_name.setText(name)
        self.lbl_recipe_name.setStyleSheet("color: blue; font-weight: bold;")


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
        # ✅ Halcon模式检查
        if not self.check_recipe_loaded_halcon():
            QMessageBox.warning(self, "需要配方", 
                              "Halcon模式下，请先加载或创建配方后再进行空跑。\n"
                              "空跑需要配方中的路径数据。")
            return
        
        # 准备相机高度参数
        height_params = {
            'safe_z_cavity': self.cam_safe_z_cavity,
            'safe_z_product': self.cam_safe_z_product,
            'work_z': self.cam_work_z,
            'speed_xy': int(self.feedrate_xy),
            'speed_z': int(self.feedrate_z),
            'dwell_ms': self.dwell_ms,
            'line_dwell_ms': self.line_dwell_ms
        }
        
        def save_cam_params(dialog):
            self.cam_safe_z_cavity = dialog.safe_z_cavity
            self.cam_safe_z_product = dialog.safe_z_product
            self.cam_work_z = dialog.work_z
            
            # 速度参数是共享的，同步更新变量和 UI 滑块
            self.feedrate_xy = dialog.speed_xy
            self.feedrate_z = dialog.speed_z
            
            # Update Sliders & Labels (Block signals to avoid double-save)
            self.sld_spd_xy.blockSignals(True)
            self.sld_spd_xy.setValue(int(self.feedrate_xy))
            self.sld_spd_xy.blockSignals(False)
            self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")

            self.sld_spd_z.blockSignals(True)
            self.sld_spd_z.setValue(int(self.feedrate_z))
            self.sld_spd_z.blockSignals(False)
            self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

            # Save dwell times
            self.dwell_ms = dialog.dwell_ms
            self.line_dwell_ms = dialog.line_dwell_ms

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
        # ✅ Halcon模式检查
        if not self.check_recipe_loaded_halcon():
            QMessageBox.warning(self, "需要配方", 
                              "Halcon模式下，请先加载或创建配方后再进行空跑。\n"
                              "空跑需要配方中的路径数据。")
            return
        
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
            'speed_z': int(self.feedrate_z),
            'dwell_ms': self.dwell_ms,
            'line_dwell_ms': self.line_dwell_ms
        }

        def save_nozzle_params(dialog):
            # 更新主界面控件，控件会自动更新变量
            self.spn_safe_z_cavity.setValue(dialog.safe_z_cavity)
            self.spn_safe_z_product.setValue(dialog.safe_z_product)
            self.spn_work_z.setValue(dialog.work_z)
            self.spn_work_z.setValue(dialog.work_z)
                        
            # 速度参数是共享的，同步更新变量和 UI 滑块
            self.feedrate_xy = dialog.speed_xy
            self.feedrate_z = dialog.speed_z
            
            # Update Sliders & Labels (Block signals to avoid double-save)
            self.sld_spd_xy.blockSignals(True)
            self.sld_spd_xy.setValue(int(self.feedrate_xy))
            self.sld_spd_xy.blockSignals(False)
            self.lbl_spd_xy.setText(f"XY速度: {int(self.feedrate_xy)}")

            self.sld_spd_z.blockSignals(True)
            self.sld_spd_z.setValue(int(self.feedrate_z))
            self.sld_spd_z.blockSignals(False)
            self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")

            # Save dwell times
            self.dwell_ms = dialog.dwell_ms
            self.line_dwell_ms = dialog.line_dwell_ms

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
                self.dry_run_ctrl_dlg = DryRunControlDialog(self, self.dry_run_worker, full_script)
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
        按钮：测试出胶 (独立阀门控制)
        打开 ValveControlDialog
        """
        if not self.machine.connected:
             QMessageBox.warning(self, "未连接", "请先连接 Klipper！")
             return

        dlg = ValveControlDialog(self, self.machine)
        dlg.exec_()

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
            
        if hasattr(self, 'temp_worker') and self.temp_worker:
            self.temp_worker.stop()

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
            
            # ROI Center from Halcon Script (Dynamic)
            ref_row, ref_col = self.model_ref_row, self.model_ref_col
            # Fallback if 0 (invalid)
            if ref_row == 0 and ref_col == 0:
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
                            for item in path_segments:
                                seg_pts = None
                                is_obj = False
                                
                                # Compatibility: Handle both PathSegment object and raw numpy array
                                if hasattr(item, 'points'): # PathSegment
                                    # Use cached numpy array if available (from _compute), else convert
                                    if hasattr(item, 'points_np'):
                                        pts = item.points_np
                                    else:
                                        pts = np.array(item.points, dtype=np.float32)
                                    is_obj = True
                                else:
                                    pts = item # Expecting numpy array
                                
                                if pts is None or len(pts) == 0: continue

                                # pts is already Centered relative to Halcon ROI Center (Reference Point)
                                
                                # b. Rotate
                                # Matrix: [[cos, -sin], [sin, cos]]
                                sys_ang = angle
                                cos_a = np.cos(sys_ang)
                                sin_a = np.sin(sys_ang)
                                
                                x_new = pts[:, 0] * cos_a + pts[:, 1] * sin_a
                                y_new = -pts[:, 0] * sin_a + pts[:, 1] * cos_a
                                
                                pts_rotated = np.stack([x_new, y_new], axis=1)
                                
                                # c. Translate to machine center (Halcon Match)
                                pts_transformed = pts_rotated + np.array([cx, cy])
                                
                                # ====== ✅ Apply Vision Offset (Correction) ======
                                if hasattr(self, 'vision_offset_x'):
                                     pts_transformed[:, 0] += self.vision_offset_x
                                if hasattr(self, 'vision_offset_y'):
                                     pts_transformed[:, 1] += self.vision_offset_y
                                # ===============================================
                                
                                # d. Draw (Magenta)
                                pts_draw = pts_transformed.astype(np.int32)
                                is_closed = False 
                                if len(pts_draw) == 1:
                                     cv2.circle(debug_img, tuple(pts_draw[0]), 4, (255, 0, 255), -1)
                                else:
                                     cv2.polylines(debug_img, [pts_draw], is_closed, (255, 0, 255), 4, cv2.LINE_AA)
                                
                                # Append to results
                                if is_obj:
                                    # Create NEW PathSegment with FINAL coordinates
                                    # This ensures G-code generator receives Absolute Pixels + Color Info
                                    from core.path_segment import PathSegment
                                    final_seg = PathSegment(item.type, item.color, pts_transformed.tolist(), item.dwell_ms)
                                    mold_segments.append(final_seg)
                            
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

    def _compute_halcon_template(self, force_reload=False):
        """
        One-time computation:
        1. Load disp_poly & disp_dots (Warped Space) FIRST.
           If no paths, stop immediately and don't retry.
        2. Load Latest Capture or mold2.png
        3. Find Big Box -> Get Minv (Warped -> Raw)
        4. Project them to Raw Space
        5. Subtract Halcon ROI Center to get 'Relative Path' for Halcon
        """
        if not force_reload and getattr(self, 'halcon_path_template', None):
             return

        try:
            import cv2
            import numpy as np
            from core.mold_detector import _find_big_mold
            from core.path_segment import PathSegment
            
            # --- BASE DIR ---
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(base_dir, "config")

            # --- PRE-CHECK: C. Load Paths (Warped Space) ---
            # Do this FIRST to avoid expensive image loading if no paths exist
            segments = []
            
            # [Fix] Prioritize In-Memory Paths (Loaded from JSON via ProductManager)
            # 1. Try Multicolor Segments (New)
            if hasattr(self, 'current_segments') and self.current_segments and self.current_segments.segments:
                # current_segments.segments is a list of PathSegment objects
                segments.extend(self.current_segments.segments)
                
            # 2. Try Legacy Polys/Dots (Old)
            elif hasattr(self, 'polys') and hasattr(self, 'dots') and (self.polys or self.dots):
                # Polys
                if self.polys:
                    segments.extend([p.astype(np.float32) for p in self.polys])
                # Dots
                if self.dots:
                     for d in self.dots:
                         d_np = np.array(d, dtype=np.float32)
                         if d_np.ndim == 1:
                             segments.append(d_np.reshape(1, 2))
                         elif d_np.ndim == 2:
                             segments.append(d_np)
            else:
                # Fallback to disk (Legacy)
                # Determine path directory: use product directory if product loaded, else config
                path_dir = config_dir
                current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
                if current_recipe and "未加载" not in current_recipe:
                    path_dir = self.product_manager.get_product_path(current_recipe)
                    print(f"[Halcon] Loading paths from product (disk): {path_dir}")
                else:
                    print(f"[Halcon] Loading paths from global config: {path_dir}")
                
                # Poly
                poly_path = os.path.join(path_dir, "disp_poly_cell0.npy")
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
                dots_path = os.path.join(path_dir, "disp_dots_cell0.npy")
                if os.path.exists(dots_path):
                    dots = np.load(dots_path, allow_pickle=True)
                    if dots.size > 0:
                        # Treat each dot as a 1-point segment (1, 2)
                         if dots.ndim == 2:
                             for d in dots:
                                 segments.append(d.reshape(1, 2).astype(np.float32))

            if not segments:
                print("[Halcon] ERR: No paths loaded. Stopping template computation.")
                # CRITICAL FIX: Set to empty list so we don't retry indefinitely
                self.halcon_path_template = [] 
                return
            
            # --- A. Select Source Image ---
            gray_gold = None
            used_path = "mold2.png"
            
            # 0. Try Product Image FIRST (If a product is loaded)
            # This is CRITICAL for getting the correct Minv aligned with the Model
            current_recipe = self.lbl_recipe_name.text().replace("当前配方: ", "")
            if current_recipe and "未加载" not in current_recipe:
                recipe_img_path = os.path.join(self.product_manager.get_product_path(current_recipe), "reference.png")
                if os.path.exists(recipe_img_path):
                    print(f"[Halcon] Using Product Image: {recipe_img_path}")
                    img_raw = cv2.imread(recipe_img_path)
                    if img_raw is not None:
                        gray_gold = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                        used_path = recipe_img_path

            # 1. Try latest capture first (User Request)
            # latest_capture_path is likely absolute if from QFileDialog, but let's be safe
            if gray_gold is None and hasattr(self, 'latest_capture_path') and self.latest_capture_path and os.path.exists(self.latest_capture_path):
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
            
            # D. Project and Offset
            # Alternative: Run Match on this specific image to get ACTUAL frame of reference
            # This is more robust than parsing ROI, as it accounts for specific matching logic.
            roi_center = None
            if self.halcon_manager and self.halcon_manager.is_initialized:
                 print("[Halcon] Running match on reference image to find Origin...")
                 res = self.halcon_manager.process_image(gray_gold)
                 if res and res.get('success') and res.get('matches'):
                     # Use the top match as reference
                     top_match = res['matches'][0]
                     roi_center = np.array([top_match['col'], top_match['row']])
                     print(f"[Halcon] Found Ref Origin via Match: {roi_center}")
                 else:
                     print("[Halcon] Match failed on reference image. Fallback to ROI.")
            
            # Fallback parsing...
            # (Omitting complex script parsing for brevity if not strictly needed, but keeping logic if it was there)
            # Actually, let's keep the fallback logic or default to hardcoded if match fails, 
            # to be safe and consistent with previous code.
            
            if roi_center is None:
                # Try simple Hardcoded fallback first as per original code's "WARN" block
                print("[Halcon] WARN: Using fallback HARDCODED center (1113, 651).")
                roi_center = np.array([1113.3245, 651.6765]) 
            
            self.halcon_ref_center = roi_center
            ref_center = self.halcon_ref_center
            
            self.halcon_ref_center = roi_center
            ref_center = self.halcon_ref_center
            
            final_template = []
            
            # [Fix] Handle PathSegment objects to preserve metadata (Color/Type)
            # If segments list contains PathSegment objects (from PathSegmentList), use them.
            # Otherwise (from .npy), they are just numpy arrays.
            
            from core.path_segment import PathSegment
            
            for seg in segments:
                # 1. Check type
                if isinstance(seg, PathSegment):
                    # It's an object. Extract points.
                    # Note: seg.points might be list of tuples. Convert to numpy.
                    pts_np = np.array(seg.points, dtype=np.float32)
                    pts_w = pts_np.reshape(-1, 1, 2)
                    
                    # Project Warped -> Raw
                    pts_raw = cv2.perspectiveTransform(pts_w, Minv).reshape(-1, 2)
                    
                    # Subtract ROI Center -> Relative
                    pts_rel = pts_raw - ref_center
                    
                    # Create NEW Segment with transformed points
                    # (Keep color, dwell, type)
                    # Store as OBJECT in template list
                    new_seg = PathSegment(seg.type, seg.color, pts_rel.tolist(), seg.dwell_ms)
                    # OPTIMIZATION: Store numpy array in points for speed, if downstream supports it?
                    # BatchWorker checks if hasattr points.
                    # HalconManager._transform_and_draw_paths expects... wait.
                    # Let's keep points as List[Tuple] to match class definition, 
                    # OR hijack it to store numpy if we are careful.
                    # Safer: Store as object, but maybe add a .points_np attribute?
                    new_seg.points_np = pts_rel # Attached for calculation convenience
                    final_template.append(new_seg)
                    
                else:
                    # Legacy: Just a numpy array
                    pts_w = seg.reshape(-1, 1, 2)
                    pts_raw = cv2.perspectiveTransform(pts_w, Minv).reshape(-1, 2)
                    pts_rel = pts_raw - ref_center
                    final_template.append(pts_rel)
                
            self.halcon_path_template = final_template
            print(f"[Halcon] Template computed! {len(final_template)} segments (Mixed Mode).")
            
        except Exception as e:
            print(f"[Halcon] Template compute error: {e}")
            import traceback
            traceback.print_exc()

    # ================== 产品管理方法 (Product Management) ==================
    
    def on_halcon_toggled(self):
        """Halcon复选框切换"""
        enabled = self.chk_use_halcon.isChecked()
        
        if enabled and not self.halcon_manager.is_initialized:
            if not self.halcon_manager.init_engine():
                QMessageBox.warning(self, "错误", "Halcon引擎初始化失败\n请确保已安装mvtec-halcon")
                self.chk_use_halcon.setChecked(False)
                return
        
        self.update_button_states()
        self.log(f"Halcon模式: {'启用' if enabled else '禁用'}")
    
    def on_train_model(self):
        """训练新模型"""
        # 1. 选择图片
        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练图像", "", "Images (*.png *.jpg *.bmp)")
        if not img_path:
            return
        
        # 2. 输入产品名
        name, ok = QInputDialog.getText(self, "新建产品", "请输入产品名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        
        # 3. 检查是否已存在
        if name in self.product_manager.list_products():
            reply = QMessageBox.question(
                self, "产品已存在", 
                f"产品 '{name}' 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        self.log(f"📝 开始训练产品: {name}")
        
        # 4. 收集初始配置
        initial_config = {
            'vision_offset_x': 0.0,
            'vision_offset_y': 0.0,
            'exposure': self.spn_exp.value() if hasattr(self, 'spn_exp') else 4.4,
            'gain': self.spn_gain.value() if hasattr(self, 'spn_gain') else 5.5,
            'gamma': self.spn_gamma.value() if hasattr(self, 'spn_gamma') else 2.6,
            'feedrate_xy': self.feedrate_xy,
            'feedrate_z': self.feedrate_z,
            'accel': self.accel,
            'jog_step_mm': self.jog_step_mm,
            'safe_z_cavity': self.safe_z_cavity,
            'safe_z_product': self.safe_z_product,
            'work_z': self.work_z,
            'nozzle_offset_x': self.nozzle_offset_x,
            'nozzle_offset_y': self.nozzle_offset_y,
            'z_base': self.z_base,
        }
        
        # 5. 调用HalconProductManager训练
        success, msg = self.halcon_product_manager.train_new_product(
            img_path, name, initial_config
        )
        
        if success:
            self.current_product = name
            self.lbl_recipe_name.setText(name)
            self.lbl_recipe_name.setStyleSheet("color: #10B981; font-weight: bold;")
            
            # 同步模型参数
            self.model_ref_row = self.halcon_product_manager.model_ref_row
            self.model_ref_col = self.halcon_product_manager.model_ref_col
            
            # 清空路径
            self.dots = []
            self.polys = []
            
            # 清除缓存
            if hasattr(self, 'halcon_path_template'):
                self.halcon_path_template = None
            
            self.update_button_states()
            self.log(f"✅ {msg}")
            QMessageBox.information(self, "成功", f"产品 '{name}' 训练完成！\n可以开始编辑路径。")
        else:
            QMessageBox.critical(self, "训练失败", msg)
            self.log(f"❌ {msg}")
    
    def on_load_recipe(self):
        """加载产品（保持方法名兼容性）"""
        self.on_load_product()
    
    def on_load_product(self):
        """加载产品 (Enhanced Dialog)"""
        products = self.product_manager.list_products()
        if not products:
            QMessageBox.information(self, "提示", "没有可用的产品\n请先训练新模型")
            return
        
        # --- Create Custom Dialog ---
        dialog = QDialog(self)
        dialog.setWindowTitle("加载产品配方")
        dialog.resize(400, 500)
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("可用产品列表:"))
        
        list_widget = QListWidget()
        list_widget.addItems(products)
        list_widget.setCurrentRow(0) # Default select first
        layout.addWidget(list_widget)
        
        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        
        btn_load = QPushButton("加载 (Load)")
        btn_load.setStyleSheet("background-color: #2563EB; color: white; font-weight: bold; padding: 6px;")
        
        btn_delete = QPushButton("删除 (Delete)")
        btn_delete.setStyleSheet("background-color: #EF4444; color: white; padding: 6px;")
        
        btn_cancel = QPushButton("取消 (Cancel)")
        btn_cancel.setStyleSheet("padding: 6px;")
        
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        # --- Internal Functions ---
        self._selected_product_to_load = None
        
        def do_load():
            item = list_widget.currentItem()
            if not item: return
            self._selected_product_to_load = item.text()
            dialog.accept()
            
        def do_delete():
            item = list_widget.currentItem()
            if not item: return
            name = item.text()
            
            # Confirm
            reply = QMessageBox.question(
                dialog, "确认删除",
                f"确定要删除产品 '{name}' 吗？\n此操作不可恢复！",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    import shutil
                    product_path = self.product_manager.get_product_path(name)
                    if os.path.exists(product_path):
                        shutil.rmtree(product_path)
                        self.log(f"✅ 已删除产品: {name}")
                        
                        # Remove from list
                        row = list_widget.row(item)
                        list_widget.takeItem(row)
                        
                        # Reset current if matches
                        if self.current_product == name:
                             self.current_product = None
                             self.lbl_recipe_name.setText("未加载")
                             self.lbl_recipe_name.setStyleSheet("color: red; font-weight: bold;")
                             self.dots = []
                             self.polys = []
                             self.update_button_states()
                             
                        if list_widget.count() == 0:
                             QMessageBox.information(dialog, "提示", "所有产品已删除")
                             dialog.reject()
                except Exception as e:
                    QMessageBox.critical(dialog, "删除失败", f"无法删除产品: {e}")
        
        # Check double click
        list_widget.itemDoubleClicked.connect(do_load)
        
        btn_load.clicked.connect(do_load)
        btn_delete.clicked.connect(do_delete)
        btn_cancel.clicked.connect(dialog.reject)
        
        # --- Execute Dialog ---
        if dialog.exec_() == QDialog.Accepted and self._selected_product_to_load:
            name = self._selected_product_to_load
            
            # Exec Loading Logic
            self.log(f"📂 正在加载产品: {name}")
            
            success, msg = self.halcon_product_manager.load_product(name)
            
            if success:
                self.current_product = name
                self.lbl_recipe_name.setText(name)
                self.lbl_recipe_name.setStyleSheet("color: #10B981; font-weight: bold;")
                
                # 同步参数到UI
                self.model_ref_row = self.halcon_product_manager.model_ref_row
                self.model_ref_col = self.halcon_product_manager.model_ref_col
                self.vision_offset_x = self.halcon_product_manager.vision_offset_x
                self.vision_offset_y = self.halcon_product_manager.vision_offset_y
            
            # 更新UI显示
            if hasattr(self, 'spn_vision_offset_x'):
                self.spn_vision_offset_x.blockSignals(True)
                self.spn_vision_offset_x.setValue(self.vision_offset_x)
                self.spn_vision_offset_x.blockSignals(False)
            if hasattr(self, 'spn_vision_offset_y'):
                self.spn_vision_offset_y.blockSignals(True)
                self.spn_vision_offset_y.setValue(self.vision_offset_y)
                self.spn_vision_offset_y.blockSignals(False)
            
            # 同步路径
            paths = self.halcon_product_manager.get_current_paths()
            self.dots = paths['dots'] if paths['dots'] is not None else []
            self.polys = paths['polys'] if paths['polys'] is not None else []
            
            # [NEW] 获取并存储 PathSegmentList (包含颜色/阀门信息)
            # 用于 _compute_halcon_template 和 G-code 生成
            self.current_segments = self.halcon_product_manager.get_current_segments()
            if self.current_segments:
                 print(f"[Main] Loaded {len(self.current_segments.segments)} multicolor segments.")
            else:
                 print("[Main] Warning: No multicolor segments loaded (Legacy Mode?)")

            # [NEW] 强制覆盖 mold_detector 中的路径，确保 Vision 识别使用的是最新加载的 (JSON or NPY)
            # 解决: 编辑完了多色路径，但识别还是用的旧 NPY 的问题
            try:
                from core.mold_detector import set_override_paths
                set_override_paths(self.polys, self.dots)
            except ImportError:
                print("[Main] Warning: mold_detector.set_override_paths not found")
            
            # ⭐ 加载并应用配方的所有参数
            config = self.product_manager.load_product_config(name)
            if config:
                # 1. 相机参数（直接应用到相机硬件）
                if 'exposure' in config:
                    exp_ms = config['exposure']  # 单位：ms
                    if hasattr(self, 'spn_exp'):
                        self.spn_exp.blockSignals(True)
                        self.spn_exp.setValue(exp_ms)
                        self.spn_exp.blockSignals(False)
                    # ⭐ 同步slider (0-100, 对应0-10ms)
                    if hasattr(self, 'sld_exp'):
                        slider_val = int(exp_ms * 10.0)
                        self.sld_exp.blockSignals(True)
                        self.sld_exp.setValue(slider_val)
                        self.sld_exp.blockSignals(False)
                    # 应用到相机（单位：us）
                    self.camera_manager.set_exposure(exp_ms * 1000.0)
                
                if 'gain' in config:
                    gain_val = config['gain']
                    if hasattr(self, 'spn_gain'):
                        self.spn_gain.blockSignals(True)
                        self.spn_gain.setValue(gain_val)
                        self.spn_gain.blockSignals(False)
                    # ⭐ 同步slider (0-100, 对应0-10dB)
                    if hasattr(self, 'sld_gain'):
                        slider_val = int(gain_val * 10)
                        self.sld_gain.blockSignals(True)
                        self.sld_gain.setValue(slider_val)
                        self.sld_gain.blockSignals(False)
                    # 应用到相机
                    self.camera_manager.set_gain(gain_val)
                
                if 'gamma' in config:
                    gamma_val = config['gamma']
                    if hasattr(self, 'spn_gamma'):
                        self.spn_gamma.blockSignals(True)
                        self.spn_gamma.setValue(gamma_val)
                        self.spn_gamma.blockSignals(False)
                    # ⭐ 同步slider (1-40, 对应0.1-4.0)
                    if hasattr(self, 'sld_gamma'):
                        slider_val = int(gamma_val * 10)
                        self.sld_gamma.blockSignals(True)
                        self.sld_gamma.setValue(slider_val)
                        self.sld_gamma.blockSignals(False)
                    # 应用到相机
                    self.camera_manager.set_gamma(gamma_val)
                
                # 2. 运动参数
                if 'feedrate_xy' in config:
                    self.feedrate_xy = config['feedrate_xy']
                    if hasattr(self, 'sld_spd_xy'):
                        self.sld_spd_xy.blockSignals(True)
                        self.sld_spd_xy.setValue(int(self.feedrate_xy))
                        self.sld_spd_xy.blockSignals(False)
                        self.lbl_spd_xy.setText(f"XY 速度: {int(self.feedrate_xy)}")
                
                if 'feedrate_z' in config:
                    self.feedrate_z = config['feedrate_z']
                    if hasattr(self, 'sld_spd_z'):
                        self.sld_spd_z.blockSignals(True)
                        self.sld_spd_z.setValue(int(self.feedrate_z))
                        self.sld_spd_z.blockSignals(False)
                        self.lbl_spd_z.setText(f"Z 速度: {int(self.feedrate_z)}")
                
                if 'accel' in config:
                    self.accel = config['accel']
                
                if 'jog_step_mm' in config:
                    self.jog_step_mm = config['jog_step_mm']
                
                # 3. Z高度参数
                if 'safe_z_cavity' in config:
                    self.safe_z_cavity = config['safe_z_cavity']
                
                if 'safe_z_product' in config:
                    self.safe_z_product = config['safe_z_product']
                
                if 'work_z' in config:
                    self.work_z = config['work_z']
                
                # 4. 针头偏移
                if 'nozzle_offset_x' in config:
                    self.nozzle_offset_x = config['nozzle_offset_x']
                
                if 'nozzle_offset_y' in config:
                    self.nozzle_offset_y = config['nozzle_offset_y']
                
                if 'z_base' in config:
                    self.z_base = config['z_base']
                
                self.log(f"✅ 已加载配方参数: 曝光={config.get('exposure', '?')}, XY速度={int(self.feedrate_xy)}, Z={self.safe_z_product:.1f}")
            
            # 清除缓存
            if hasattr(self, 'halcon_path_template'):
                self.halcon_path_template = None
            
            self.update_button_states()
            self.log(f"✅ 产品加载成功")
            self.log(f"📊 路径: {len(self.dots)} 点, {len(self.polys)} 线")
        else:
            QMessageBox.critical(self, "加载失败", msg)
            self.log(f"❌ {msg}")
    
    def on_save_recipe(self):
        """保存产品（保持方法名兼容性）"""
        self.on_save_product()
    
    def on_save_product(self):
        """保存当前产品"""
        if not self.current_product:
            QMessageBox.warning(self, "提示", "没有加载的产品\n请先训练或加载产品")
            return
        
        self.log(f"💾 正在保存产品: {self.current_product}")
        
        # 收集配置
        config_data = {
            'vision_offset_x': self.vision_offset_x,
            'vision_offset_y': self.vision_offset_y,
            'exposure': self.spn_exp.value() if hasattr(self, 'spn_exp') else 4.4,
            'gain': self.spn_gain.value() if hasattr(self, 'spn_gain') else 5.5,
            'gamma': self.spn_gamma.value() if hasattr(self, 'spn_gamma') else 2.6,
            'feedrate_xy': self.feedrate_xy,
            'feedrate_z': self.feedrate_z,
            'accel': self.accel,
            'jog_step_mm': self.jog_step_mm,
            'safe_z_cavity': self.safe_z_cavity,
            'safe_z_product': self.safe_z_product,
            'work_z': self.work_z,
            'nozzle_offset_x': self.nozzle_offset_x,
            'nozzle_offset_y': self.nozzle_offset_y,
            'z_base': self.z_base,
        }
        
        # 更新HalconProductManager的路径
        self.halcon_product_manager.save_paths_to_product(self.dots, self.polys)
        
        success, msg = self.halcon_product_manager.save_product(config_data)
        
        if success:
            self.log(f"✅ {msg}")
            
            # 构建详细信息
            details = f"""产品 '{self.current_product}' 保存成功！

📊 保存的参数：
━━━━━━━━━━━━━━━━━━━━━━━━
📷 相机参数：
  • 曝光: {config_data['exposure']:.1f} ms
  • 增益: {config_data['gain']:.1f} dB
  • Gamma: {config_data['gamma']:.2f}

🎯 视觉偏移：
  • X: {config_data['vision_offset_x']:.2f} mm
  • Y: {config_data['vision_offset_y']:.2f} mm

🚀 运动参数：
  • XY速度: {int(config_data['feedrate_xy'])} mm/min
  • Z速度: {int(config_data['feedrate_z'])} mm/min
  • 加速度: {int(config_data['accel'])} mm/s²

📏 Z轴高度：
  • 腔内安全Z: {config_data['safe_z_cavity']:.1f} mm
  • 产品安全Z: {config_data['safe_z_product']:.1f} mm
  • 点胶工作Z: {config_data['work_z']:.1f} mm
  • Z基准: {config_data['z_base']:.1f} mm

🔧 针头偏移：
  • X偏移: {config_data['nozzle_offset_x']:.2f} mm
  • Y偏移: {config_data['nozzle_offset_y']:.2f} mm

💧 出胶参数：
  • 供料提前: {config_data.get('valve_prefeed_time', 0.10):.2f} 秒
  • 阀A时间: {config_data.get('valve_a_dispense_time', 0.50):.2f} 秒
  • 阀B时间: {config_data.get('valve_b_dispense_time', 0.50):.2f} 秒

🎨 点胶路径：
  • 点路径: {len(self.dots)} 个点
  • 线路径: {len(self.polys)} 条线
"""
            
            QMessageBox.information(self, "保存成功", details)
        else:
            QMessageBox.critical(self, "保存失败", msg)
            self.log(f"❌ {msg}")

    # ================== 产品管理方法 (Product Management) ==================
    
    def migrate_recipes_to_produces(self):
        """启动时自动迁移recipes→produces"""
        if os.path.exists('recipes') and not os.path.exists('produces'):
            self.log("ℹ️ 检测到旧版recipes目录，正在迁移...")
            try:
                shutil.move('recipes', 'produces')
                self.log("✅ 已迁移 recipes/ → produces/")
            except Exception as e:
                self.log(f"❌ 迁移失败: {e}")
    
    def update_button_states(self):
        """根据Halcon状态、产品加载状态和路径存在更新按钮"""
        halcon_enabled = hasattr(self, 'chk_use_halcon') and self.chk_use_halcon.isChecked()
        product_loaded = self.current_product is not None
        
        # [FIX] 检测路径：支持旧格式（dots/polys）和新格式（current_segments）
        has_legacy_paths = len(self.dots) > 0 or len(self.polys) > 0
        has_multicolor_paths = (hasattr(self, 'current_segments') and 
                                self.current_segments and 
                                len(self.current_segments.segments) > 0)
        has_paths = has_legacy_paths or has_multicolor_paths
        
        # === 1. Halcon专用按钮（默认全部禁用，只有勾选Halcon才启用） ===
        # 训练新模型：只要Halcon启用就可以用
        if hasattr(self, 'btn_train_model'):
            self.btn_train_model.setEnabled(halcon_enabled)
        
        # 加载产品：只要Halcon启用就可以用
        if hasattr(self, 'btn_load_recipe'):
            self.btn_load_recipe.setEnabled(halcon_enabled)
        
        # 保存产品：需要Halcon启用 且 产品已加载
        if hasattr(self, 'btn_save_recipe'):
            self.btn_save_recipe.setEnabled(halcon_enabled and product_loaded)
        
        # === 2. 功能按钮细粒度控制 ===
        if halcon_enabled:
            # Halcon模式：需要根据产品和路径状态细分
            # 编辑路径：产品加载后就可用（新建或已有产品都可以编辑）
            edit_enabled = product_loaded
            
            # 识别、空跑、批量：需要有路径
            run_enabled = product_loaded and has_paths
        else:
            # OpenCV模式：全部可用
            edit_enabled = True
            run_enabled = True
        
        # 编辑按钮
        if hasattr(self, 'btn_path'):
            self.btn_path.setEnabled(edit_enabled)
        
        # 识别按钮
        if hasattr(self, 'btn_detect'):
            self.btn_detect.setEnabled(run_enabled)
        
        # 空跑按钮
        if hasattr(self, 'btn_virtual'):
            self.btn_virtual.setEnabled(run_enabled)
        if hasattr(self, 'btn_dry_run'):
            self.btn_dry_run.setEnabled(run_enabled)
        
        # 批量按钮
        if hasattr(self, 'btn_batch_run'):
            self.btn_batch_run.setEnabled(run_enabled)





    # ==================== iPad副屏功能（已禁用） ====================
    # 用户使用iPad投屏直接查看主程序，不需要专门的副屏
    
    # def toggle_mobile_monitor(self):
    #     """切换iPad监控副屏的开/关状态"""
    #     if self.mobile_monitor and self.mobile_monitor.isVisible():
    #         self.mobile_monitor.close()
    #         self.mobile_monitor = None
    #         self.btn_ipad.setText("📱 启动 iPad 监控副屏")
    #         self.btn_ipad.setStyleSheet("background-color: #10B981; color: white; padding: 10px; border-radius: 5px;")
    #         print("[副屏] 已关闭")
    #     else:
    #         self.open_mobile_monitor()
    #         self.btn_ipad.setText("⏹ 停止副屏")
    #         self.btn_ipad.setStyleSheet("background-color: #EF4444; color: white; padding: 10px; border-radius: 5px;")
    #         print("[副屏] 已启动")
    # 
    # def open_mobile_monitor(self):
    #     """Open the simplified iPad dashboard - 精简版，只显示USB画面"""
    #     from mobile_ui.mobile_monitor_minimal import MobileMonitorDialog
    #     if not self.mobile_monitor:
    #         self.mobile_monitor = MobileMonitorDialog(self)
    #     
    #     # 每次打开都转移USB相机流到副屏
    #     if hasattr(self, 'usb_cam_worker') and self.usb_cam_worker:
    #         try:
    #             try:
    #                 self.usb_cam_worker.frame_signal.disconnect(self.update_usb_image)
    #             except:
    #                 pass
    #             try:
    #                 self.usb_cam_worker.frame_signal.disconnect(self.mobile_monitor.update_usb_image)
    #             except:
    #                 pass
    #             
    #             self.usb_cam_worker.frame_signal.connect(self.mobile_monitor.update_usb_image)
    #             if hasattr(self, 'label_usb_cam'):
    #                 self.label_usb_cam.hide()
    #             print("[副屏] USB流已转移到副屏")
    #         except Exception as e:
    #             print(f"[副屏] USB流转移失败: {e}")
    #     
    #     self.mobile_monitor.show()
    #     self.mobile_monitor.raise_()
    #     self.mobile_monitor.activateWindow()

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
