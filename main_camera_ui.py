import sys
import os
import ctypes
from datetime import datetime

import numpy as np
import cv2
import json

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap

# 你的检测函数
from mold_detector import detect_mold_and_cells, reload_golden_templates


# ---- 按你的实际路径修改 ----
sys.path.append(r"H:\MVS\Development\Samples\Python\MvImport")
from MvCameraControl_class import *


# ================== 线程：黄金示教 ==================


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
        self.setWindowTitle("HikRobot Camera (Python + Qt)")

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

        # self.init_calibration()  # 读取 calib/camera_matrix.npy 和 dist_coeffs.npy

        # 识别结果
        self.big_box = None


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

        self.init_ui()
        self.load_saved_params()
        self.load_golden_templates()

    # ================== UI ==================
    def init_ui(self):
        # ---- 左侧大画面 ----
        self.label = QLabel("无图像")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(600, 450)

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

        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_golden)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_path)

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

        # ---- 日志窗口 ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)

        # ---- 右侧整体布局 ----
        right_layout = QVBoxLayout()
        right_layout.addWidget(info_group)
        right_layout.addLayout(btn_layout)
        right_layout.addWidget(param_group)
        right_layout.addWidget(self.log_box)
        right_layout.addStretch(1)

        # ---- 总布局 ----
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)
        self.setLayout(main_layout)

        # ================== 信号连接 ==================
        self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop.clicked.connect(self.stop_cam)
        self.btn_save.clicked.connect(self.save_image)

        self.btn_detect.clicked.connect(self.detect_mold)
        self.btn_golden.clicked.connect(self.on_golden_teach)
        self.btn_resume.clicked.connect(self.resume_preview)
        self.btn_path.clicked.connect(self.on_edit_path)

        self.sld_exp.valueChanged.connect(self.on_exp_slider_changed)
        self.sld_gain.valueChanged.connect(self.on_gain_slider_changed)
        self.sld_gamma.valueChanged.connect(self.on_gamma_slider_changed)

        self.spn_exp.valueChanged.connect(self.on_exp_spin_changed)
        self.spn_gain.valueChanged.connect(self.on_gain_spin_changed)
        self.spn_gamma.valueChanged.connect(self.on_gamma_spin_changed)

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

        # log_box 还没创建时，用 print，创建后再用 log()
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

    # ================== 图像 / 保存 ==================

    # 带棋盘的去畸变参数
    # def update_image(self, img: np.ndarray):
    #     # -------- 1. 如有标定参数，先创建 remap 映射 --------
    #     frame = img
    #
    #     if self._calib_loaded:
    #         h, w = frame.shape
    #         if self.map1 is None or self.calib_size != (w, h):
    #             # 只在第一次/分辨率变化时初始化
    #             self.calib_size = (w, h)
    #             try:
    #                 newK, _ = cv2.getOptimalNewCameraMatrix(
    #                     self.K, self.D, self.calib_size, 1, self.calib_size
    #                 )
    #                 self.map1, self.map2 = cv2.initUndistortRectifyMap(
    #                     self.K, self.D, None, newK, self.calib_size, cv2.CV_16SC2
    #                 )
    #                 self.log(f"✅ 已创建去畸变映射，分辨率 = {w} x {h}")
    #             except Exception as e:
    #                 self.log(f"⚠ 创建去畸变映射失败：{e}")
    #                 self.map1, self.map2 = None, None
    #
    #         # 真正做一次去畸变
    #         if self.map1 is not None and self.map2 is not None:
    #             frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
    #
    #     # -------- 2. 把“去畸变后的图像”作为 last_frame --------
    #     self.last_frame = frame.copy()
    #
    #     if self.freeze_display:
    #         return
    #
    #     # -------- 3. 后面的逻辑完全保持不变，只是换成 frame --------
    #     if self.live_detect:
    #         color_img, big_box, cell_boxes, cell_centers = detect_mold_and_cells(self.last_frame)
    #         rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgb.shape
    #         bytes_per_line = ch * w
    #         qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    #     else:
    #         h, w = self.last_frame.shape
    #         qimg = QImage(self.last_frame.data, w, h, w, QImage.Format_Grayscale8).copy()
    #
    #     qpix = QPixmap.fromImage(qimg)
    #     self.label.setPixmap(qpix.scaled(
    #         self.label.width(),
    #         self.label.height(),
    #         Qt.KeepAspectRatio,
    #         Qt.SmoothTransformation
    #     ))
    def update_image(self, img: np.ndarray):
        """
        临时关闭相机标定 / 去畸变逻辑，直接使用相机原始图像。
        """
        # 直接用原图
        self.last_frame = img.copy()

        if self.freeze_display:
            return

        if self.live_detect:
            # 和你原来一样，直接把 last_frame 丢给检测
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

        # 切换实时识别开关
        self.live_detect = not self.live_detect

        if self.live_detect:
            self.btn_detect.setText("停止识别")
            self.log("✅ 已开启实时识别（每帧都会画大框+小框）")

            # ⭐ 关键：刚刚打开识别时，立刻对当前帧跑一次识别，
            # 避免黄金示教后第一次点击看不到大小框
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

    # ================== 黄金示教 ==============================
    def on_golden_teach(self):
        """点击‘黄金示教’，使用本地 mold2.png 文件进行示教"""

        # ⭐ 不再停止相机线程，只是冻结左侧画面，避免刷新
        self.freeze_display = True

        # 如果你是用固定的 mold2.png，就只检查文件是否存在
        if not os.path.exists("mold2.png"):
            self.log("❌ 未找到 mold2.png，请先准备好这张图片")
            self.freeze_display = False
            return

        # 如果你想用当前画面当黄金，也可以这样保存（可选）
        # if self.last_frame is not None:
        #     cv2.imwrite("mold2.png", self.last_frame)

        # 同步调用黄金示教
        from golden_teach import run_golden_teach
        ok = run_golden_teach()

        # ⭐ 黄金示教结束，解除冻结，让画面继续刷新
        self.freeze_display = False

        if ok:
            self.log("✅ 黄金示教完成，重新加载模板")
            # 刷新 mold_detector 里的缓存，保证识别用最新模板
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
                # ⭐ 路径更新后，刷新 mold_detector 的缓存
                from mold_detector import reload_golden_templates
                reload_golden_templates()
            else:
                self.log("⚠ 点胶路径编辑已取消或未保存")
        except Exception as e:
            self.log(f"❌ 打开路径示教失败：{e}")

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
