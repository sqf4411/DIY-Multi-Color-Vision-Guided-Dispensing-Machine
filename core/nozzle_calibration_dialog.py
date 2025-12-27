from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

class NozzleCalibrationDialog(QDialog):
    # 定义信号，用于通知主窗口保存设置
    # args: offset_x, offset_y
    calibration_finished = pyqtSignal(float, float)

    def __init__(self, parent=None, machine=None, current_offset_x=0.0, current_offset_y=0.0):
        super().__init__(parent)
        self.machine = machine
        self.setWindowTitle("针头偏移标定")
        self.resize(400, 300)
        # 设置为非模态，允许用户操作主窗口（移动XYZ）
        self.setWindowModality(Qt.NonModal)
        
        # 状态数据
        self.cam_x = None
        self.cam_y = None
        self.nozzle_x = None
        self.nozzle_y = None
        
        # ✅ 保存当前偏移值
        self.current_offset_x = current_offset_x
        self.current_offset_y = current_offset_y
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # 说明
        lbl_desc = QLabel(
            "标定流程：\n"
            "1. 移动 XYZ，使【相机中心】对准一个清晰的特征点。\n"
            "2. 点击“记录相机坐标”。\n"
            "3. 移动 XYZ，使【针头尖端】对准同一个特征点。\n"
            "4. 点击“记录针头坐标”。\n"
            "5. 确认偏移量无误后，点击“保存并应用”。"
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #9CA3AF; font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(lbl_desc)

        # --- 步骤 1: 相机 ---
        grp_cam = QGroupBox("步骤 1: 相机对位")
        layout_cam = QVBoxLayout()
        
        self.lbl_cam_pos = QLabel("未记录")
        self.lbl_cam_pos.setStyleSheet("color: #F59E0B; font-weight: bold;")
        
        self.btn_record_cam = QPushButton("记录相机坐标")
        self.btn_record_cam.clicked.connect(self.on_record_cam)
        
        layout_cam.addWidget(self.lbl_cam_pos)
        layout_cam.addWidget(self.btn_record_cam)
        grp_cam.setLayout(layout_cam)
        layout.addWidget(grp_cam)

        # --- 步骤 2: 针头 ---
        grp_nozzle = QGroupBox("步骤 2: 针头对位")
        layout_nozzle = QVBoxLayout()
        
        self.lbl_nozzle_pos = QLabel("未记录")
        self.lbl_nozzle_pos.setStyleSheet("color: #6B7280;") # 初始灰色
        
        self.btn_record_nozzle = QPushButton("记录针头坐标")
        self.btn_record_nozzle.clicked.connect(self.on_record_nozzle)
        self.btn_record_nozzle.setEnabled(False) # 初始禁用
        
        layout_nozzle.addWidget(self.lbl_nozzle_pos)
        layout_nozzle.addWidget(self.btn_record_nozzle)
        grp_nozzle.setLayout(layout_nozzle)
        layout.addWidget(grp_nozzle)

        # --- 结果 ---
        grp_res = QGroupBox("标定结果")
        layout_res = QFormLayout()
        self.lbl_res_x = QLabel(f"{self.current_offset_x:.3f} mm")
        self.lbl_res_y = QLabel(f"{self.current_offset_y:.3f} mm")
        layout_res.addRow("ΔX:", self.lbl_res_x)
        layout_res.addRow("ΔY:", self.lbl_res_y)
        grp_res.setLayout(layout_res)
        layout.addWidget(grp_res)

        # --- 底部按钮 ---
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存并应用")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("background-color: #10B981; color: white; font-weight: bold;")
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def on_record_cam(self):
        if not self.machine: return
        x, y, z = self.machine.get_position()
        self.cam_x = x
        self.cam_y = y
        
        self.lbl_cam_pos.setText(f"X: {x:.3f}, Y: {y:.3f}")
        self.lbl_cam_pos.setStyleSheet("color: #10B981; font-weight: bold;") # 绿色
        
        # 切换状态
        self.btn_record_cam.setEnabled(False)
        self.btn_record_cam.setText("已记录")
        
        self.btn_record_nozzle.setEnabled(True)
        self.lbl_nozzle_pos.setStyleSheet("color: #F59E0B; font-weight: bold;") # 激活下个步骤颜色

    def on_record_nozzle(self):
        if not self.machine: return
        x, y, z = self.machine.get_position()
        self.nozzle_x = x
        self.nozzle_y = y
        
        self.lbl_nozzle_pos.setText(f"X: {x:.3f}, Y: {y:.3f}")
        self.lbl_nozzle_pos.setStyleSheet("color: #10B981; font-weight: bold;")
        
        self.btn_record_nozzle.setEnabled(False)
        self.btn_record_nozzle.setText("已记录")
        
        # 计算结果
        if self.cam_x is not None:
            dx = self.nozzle_x - self.cam_x
            dy = self.nozzle_y - self.cam_y
            self.lbl_res_x.setText(f"{dx:.3f} mm")
            self.lbl_res_y.setText(f"{dy:.3f} mm")
            
            self.btn_save.setEnabled(True)

    def on_save(self):
        if self.cam_x is not None and self.nozzle_x is not None:
            dx = self.nozzle_x - self.cam_x
            dy = self.nozzle_y - self.cam_y
            # 发送信号
            self.calibration_finished.emit(dx, dy)
            self.accept()
