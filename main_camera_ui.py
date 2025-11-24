import os
import sys

def add_mvs_dll_path():
    dll_dir = os.getenv("MVS_DLL_DIR")
    if dll_dir and os.path.isdir(dll_dir):
        # Windows 10+ 推荐用这个
        if sys.platform == "win32":
            os.add_dll_directory(dll_dir)
        else:
            # 理论上不会走到这里，预防一下
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[INFO] MVS DLL directory added: {dll_dir}")
    else:
        print("[WARN] 环境变量 MVS_DLL_DIR 未设置或路径不存在，请检查。")

add_mvs_dll_path()

import sys
import os
import ctypes
from datetime import datetime
from machine_client import MachineClient

import numpy as np
import cv2
import json

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QTextEdit,
    QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap

# 你的检测函数
from mold_detector import detect_mold_and_cells, reload_golden_templates

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

# ================== 主窗口 ==================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
        QWidget {
            background-color: #111827;        /* 整体深灰背景 */
            color: #E5E7EB;                   /* 文字偏亮 */
            font-family: "Microsoft YaHei";
            font-size: 10pt;
        }

        /* 分组卡片 */
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
        self.thread = None
        self.last_frame = None
        self.freeze_display = False  # 是否冻结画面

        self.live_detect = False     # 是否实时识别模具

        # ==== 相机标定 / 去畸变 ====
        self.K = None          # 内参矩阵
        self.D = None          # 畸变系数
        self.map1 = None       # remap 映射
        self.map2 = None
        self.calib_size = None # (w, h)
        self._calib_loaded = False

        # 识别结果
        self.big_box = None
        self.cell_boxes = []
        self.cell_centers = []

        # 配置文件
        self.config_path = "camera_params.json"
        self.saved_params = {}

        # 黄金模板
        self.slot_templates = None   # (4,4,2)

        # 参数范围
        self.exp_min = 100.0
        self.exp_max = 10000.0
        self.gain_min = 0.0
        self.gain_max = 10.0
        self.gamma_min = 0.5
        self.gamma_max = 2.0

        # ==== 运动 / 坐标相关 ====
        self.machine = MachineClient()
        self.machine.connect()

        # 当前坐标（mm）
        self.pos_x, self.pos_y, self.pos_z = self.machine.get_position()

        # 点动步长（mm）
        self.jog_step_mm = 1.0

        # 针头相对「相机参考点」的 XY 偏移量（mm）
        self.nozzle_offset_x = 0.0
        self.nozzle_offset_y = 0.0

        # 相机像素 -> 机器 XY 的映射矩阵
        self.cam2mach_matrix = None

        # 安全高度（Z）
        self.safe_z = 10.0
        self.feedrate_xy = 50.0   # XY 速度
        self.feedrate_z = 20.0    # Z 速度

        self.init_ui()
        self.load_saved_params()
        self.load_golden_templates()

    # ================== UI ==================
    def init_ui(self):
        # ---- 左侧大画面 ----
        self.label = QLabel("无图像")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(800, 600)
        self.label.setStyleSheet("""
            background-color: #000000;
            color: #9CA3AF;
            border-radius: 8px;
            border: 1px solid #374151;
        """)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label)

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

        # ---- 控制按钮 ----
        self.btn_start = QPushButton("启动相机")
        self.btn_stop = QPushButton("停止相机")
        self.btn_save = QPushButton("保存图片")
        self.btn_detect = QPushButton("识别模具")
        self.btn_golden = QPushButton("黄金示教")
        self.btn_resume = QPushButton("恢复预览")
        self.btn_path = QPushButton("编辑点胶路径")

        # 点胶任务相关按钮
        self.btn_calibrate = QPushButton("标定坐标系")
        self.btn_virtual = QPushButton("虚拟跑")
        self.btn_dry_run = QPushButton("空跑")
        self.btn_real_run = QPushButton("开始点胶")
        self.btn_test_valve = QPushButton("测试出胶")

        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        # 第一排：相机 & 示教
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_golden)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_path)

        # 第二排：点胶任务
        task_layout = QHBoxLayout()
        task_layout.addWidget(self.btn_calibrate)
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

        # 当前坐标显示
        coord_layout = QHBoxLayout()
        self.lbl_pos_x = QLabel("X: 0.000")
        self.lbl_pos_y = QLabel("Y: 0.000")
        self.lbl_pos_z = QLabel("Z: 0.000")
        coord_layout.addWidget(self.lbl_pos_x)
        coord_layout.addWidget(self.lbl_pos_y)
        coord_layout.addWidget(self.lbl_pos_z)
        motion_layout.addLayout(coord_layout)

        # 各轴归位按钮
        home_layout = QHBoxLayout()
        self.btn_home_x = QPushButton("X归位")
        self.btn_home_y = QPushButton("Y归位")
        self.btn_home_z = QPushButton("Z归位")
        home_layout.addWidget(self.btn_home_x)
        home_layout.addWidget(self.btn_home_y)
        home_layout.addWidget(self.btn_home_z)
        motion_layout.addLayout(home_layout)

        # 步长设置
        step_layout = QHBoxLayout()
        lbl_step = QLabel("步长 (mm)")
        self.spn_step = QDoubleSpinBox()
        self.spn_step.setRange(0.01, 100.0)
        self.spn_step.setDecimals(3)
        self.spn_step.setSingleStep(0.1)
        self.spn_step.setValue(self.jog_step_mm)
        step_layout.addWidget(lbl_step)
        step_layout.addWidget(self.spn_step)
        motion_layout.addLayout(step_layout)

        # X 轴按钮
        row_x = QHBoxLayout()
        self.btn_x_minus = QPushButton("X-")
        self.btn_x_plus = QPushButton("X+")
        row_x.addWidget(self.btn_x_minus)
        row_x.addWidget(self.btn_x_plus)
        motion_layout.addLayout(row_x)

        # Y 轴按钮
        row_y = QHBoxLayout()
        self.btn_y_minus = QPushButton("Y-")
        self.btn_y_plus = QPushButton("Y+")
        row_y.addWidget(self.btn_y_minus)
        row_y.addWidget(self.btn_y_plus)
        motion_layout.addLayout(row_y)

        # Z 轴按钮
        row_z = QHBoxLayout()
        self.btn_z_minus = QPushButton("Z-")
        self.btn_z_plus = QPushButton("Z+")
        row_z.addWidget(self.btn_z_minus)
        row_z.addWidget(self.btn_z_plus)
        motion_layout.addLayout(row_z)

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

        # ---- Z 高度 / 对刀 ----
        z_group = QGroupBox("Z 高度 / 对刀")
        z_layout = QVBoxLayout()
        self.lbl_z_ref = QLabel("当前 Z 基准: -.- mm")
        self.btn_tool_probe = QPushButton("一键对刀")
        z_layout.addWidget(self.lbl_z_ref)
        z_layout.addWidget(self.btn_tool_probe)
        z_group.setLayout(z_layout)
        self.z_group = z_group

        # ---- 日志窗口 ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)

        # ---- 右侧整体布局 ----
        right_layout = QVBoxLayout()
        right_layout.addWidget(info_group)
        right_layout.addWidget(self.motion_group)   # XYZ 控制
        right_layout.addWidget(self.offset_group)   # 针头偏移
        right_layout.addWidget(self.z_group)        # Z 高度 / 对刀
        right_layout.addLayout(btn_layout)
        right_layout.addLayout(task_layout)         # 点胶任务按钮排
        right_layout.addWidget(param_group)
        right_layout.addWidget(self.log_box)
        right_layout.addStretch(1)

        # ---- 总布局 ----
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        self.setLayout(main_layout)

        # ================== 信号连接 ==================
        self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop.clicked.connect(self.stop_cam)
        self.btn_save.clicked.connect(self.save_image)

        self.btn_detect.clicked.connect(self.detect_mold)
        self.btn_golden.clicked.connect(self.on_golden_teach)
        self.btn_resume.clicked.connect(self.resume_preview)
        self.btn_path.clicked.connect(self.on_edit_path)

        # 针头偏移标定
        self.btn_calib_nozzle.clicked.connect(self.on_calibrate_nozzle_offset)

        # 一键对刀
        self.btn_tool_probe.clicked.connect(self.on_tool_probe_clicked)

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
        self.btn_home_x.clicked.connect(lambda: self.on_home_axis('x'))
        self.btn_home_y.clicked.connect(lambda: self.on_home_axis('y'))
        self.btn_home_z.clicked.connect(lambda: self.on_home_axis('z'))

        # 初始化一次坐标显示 & 针头偏移显示
        self.update_xyz_display()
        self.update_nozzle_offset_display()

    # ================== 运动 / 坐标相关 ==================
    def on_jog(self, axis: str, direction: int):
        """X± / Y± / Z± 点动按钮的回调，direction 为 +1 或 -1。"""
        step = float(self.spn_step.value())
        dx = dy = dz = 0.0

        if axis == 'x':
            dx = direction * step
        elif axis == 'y':
            dy = direction * step
        elif axis == 'z':
            dz = direction * step

        self.move_machine_rel(dx, dy, dz)

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
            f"dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f} -> "
            f"X={self.pos_x:.3f}, Y={self.pos_y:.3f}, Z={self.pos_z:.3f}"
        )

    def update_xyz_display(self, x=None, y=None, z=None):
        """
        更新 UI 上的坐标显示。
        如果传入 x/y/z，则顺便更新内部 pos_x/pos_y/pos_z。
        """
        if x is not None:
            self.pos_x = float(x)
        if y is not None:
            self.pos_y = float(y)
        if z is not None:
            self.pos_z = float(z)

        self.lbl_pos_x.setText(f"X: {self.pos_x:.3f}")
        self.lbl_pos_y.setText(f"Y: {self.pos_y:.3f}")
        self.lbl_pos_z.setText(f"Z: {self.pos_z:.3f}")

    def on_home_axis(self, axis: str):
        """单轴归位：X/Y/Z"""
        if axis == 'x':
            self.machine.home_x()
        elif axis == 'y':
            self.machine.home_y()
        elif axis == 'z':
            self.machine.home_z()
        else:
            return

        self.pos_x, self.pos_y, self.pos_z = self.machine.get_position()
        self.update_xyz_display()
        self.log("HOME:", f"Axis {axis.upper()} -> "
                           f"X={self.pos_x:.3f}, Y={self.pos_y:.3f}, Z={self.pos_z:.3f}")

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
        标定针头偏移的入口（目前先做占位，不真的去标定，只是保证按钮正常工作）。
        下一步我们会把“相机先对点，再针头对同一点”那套流程写进来。
        """
        QMessageBox.information(
            self,
            "标定针头偏移（占位）",
            "UI 已支持针头偏移显示，当前版本只是占位实现。\n\n"
            "后续步骤我们会实现：\n"
            "1. 先用相机对准一个物理点，并记录当前 XY；\n"
            "2. 再用针头对准同一个点，记录 XY；\n"
            "3. 计算相机→针头的 ΔX, ΔY 并保存。"
        )
        self.log("TODO", "标定针头偏移逻辑将在下一步实现。")

    # ================== 日志 ==================
    def log(self, *args):
        text = " ".join(str(a) for a in args)
        print(text)
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ================== 相机相关 ==================
    def start_cam(self):
        if self.cam:
            return

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
        self.apply_saved_params()

        self.cam.MV_CC_StartGrabbing()
        self.sync_from_camera_to_ui()

        self.thread = CameraThread(self.cam)
        self.thread.frame_signal.connect(self.update_image)
        self.thread.start()

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
            else:
                self.saved_params = {}
        except Exception as e:
            self.log("读取参数文件失败：", e)
            self.saved_params = {}

    def init_calibration(self):
        """从 calib 目录加载相机标定参数（camera_matrix.npy / dist_coeffs.npy）"""
        try:
            calib_dir = os.path.join(os.path.dirname(__file__), "calib")
            cam_path = os.path.join(calib_dir, "camera_matrix.npy")
            dist_path = os.path.join(calib_dir, "dist_coeffs.npy")

            if os.path.exists(cam_path) and os.path.exists(dist_path):
                self.K = np.load(cam_path)
                self.D = np.load(dist_path)
                self._calib_loaded = True
                msg = "✅ 已加载标定参数 camera_matrix.npy / dist_coeffs.npy"
            else:
                self._calib_loaded = False
                msg = "⚠ calib 目录下未找到 camera_matrix.npy 或 dist_coeffs.npy，将使用原始图像"

        except Exception as e:
            self._calib_loaded = False
            msg = f"⚠ 加载标定参数失败：{e}"

        if hasattr(self, "log_box"):
            self.log(msg)
        else:
            print(msg)

    # ===== 黄金模板 =====
    def load_golden_templates(self):
        try:
            if os.path.exists("slot_templates.npy"):
                self.slot_templates = np.load("slot_templates.npy")
                self.log(f"✅ 黄金模板 slot_templates.npy 加载成功，形状：{self.slot_templates.shape}")
            else:
                self.log("⚠ 未找到 slot_templates.npy，暂时无法做黄金对位")
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

    def update_image(self, img: np.ndarray):
        """
        临时关闭相机标定 / 去畸变逻辑，直接使用相机原始图像。
        """
        self.last_frame = img.copy()

        if self.freeze_display:
            return

        if self.live_detect:
            # 识别并画框
            color_img, big_box, cell_boxes, cell_centers = detect_mold_and_cells(self.last_frame)
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            # 灰度显示
            h, w = self.last_frame.shape
            qimg = QImage(self.last_frame.data, w, h, w, QImage.Format_Grayscale8).copy()

        qpix = QPixmap.fromImage(qimg)
        self.label.setPixmap(qpix.scaled(
            self.label.width(),
            self.label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

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

            # 打开识别时立刻对当前帧跑一次
            try:
                color_img, big_box, cell_boxes, cell_centers = detect_mold_and_cells(
                    self.last_frame
                )
                rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(
                    rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
                ).copy()
                qpix = QPixmap.fromImage(qimg)
                self.label.setPixmap(
                    qpix.scaled(
                        self.label.width(),
                        self.label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            except Exception as e:
                self.log(f"首次识别出错：{e}")
        else:
            self.btn_detect.setText("识别模具")
            self.log("⏹ 已关闭实时识别")

    # ================== 黄金示教 ==========================
    def on_golden_teach(self):
        """点击‘黄金示教’，使用本地 mold2.png 文件进行示教"""
        self.freeze_display = True

        if not os.path.exists("mold2.png"):
            self.log("❌ 未找到 mold2.png，请先准备好这张图片")
            self.freeze_display = False
            return

        from golden_teach import run_golden_teach
        ok = run_golden_teach()

        self.freeze_display = False

        if ok:
            self.log("✅ 黄金示教完成，重新加载模板")
            reload_golden_templates()
        else:
            self.log("⚠ 黄金示教取消或失败")

    # ================== 点胶路径编辑 ==================
    def on_edit_path(self):
        """点击按钮：在黄金图片中编辑某个小框的点胶路径"""
        self.log("🧵 打开点胶路径示教窗口（默认小框 index=0）")
        try:
            from path_teach import run_path_teach
            ok = run_path_teach(cell_index=0, img_path="mold2.png")
            if ok:
                self.log("✅ 点胶路径已保存到 disp_poly_cell0.npy / disp_dots_cell0.npy，已重新加载")
                from mold_detector import reload_golden_templates
                reload_golden_templates()
            else:
                self.log("⚠ 点胶路径编辑已取消或未保存")
        except Exception as e:
            self.log(f"❌ 打开路径示教失败：{e}")

    # ================== 点胶任务按钮（占位） ==================
    def on_calibrate_coords(self):
        """
        按钮：标定坐标系（相机像素 → 机器 XY）
        当前为占位，后面会实现 4 点标定。
        """
        self.log("🧮 标定坐标系：暂时只是占位，下一步我们会专门实现标定流程。")

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
        按钮：空跑
        只走 XYZ，不打开点胶阀。
        """
        if self.cam2mach_matrix is None:
            self.log("⚠ 还没有做坐标标定（cam2mach_matrix 为 None），暂不执行空跑。")
            return
        self.log("🏃 空跑：将来会发送 Gcode/运动指令给 MachineClient（当前为占位）。")

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
    def closeEvent(self, e):
        self.stop_cam()
        e.accept()

    def resume_preview(self):
        self.freeze_display = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec_())
