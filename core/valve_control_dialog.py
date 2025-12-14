from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QDoubleSpinBox, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt, QTimer

class ValveControlDialog(QDialog):
    def __init__(self, parent=None, machine_client=None):
        super().__init__(parent)
        self.setWindowTitle("电磁阀独立控制测试")
        self.resize(550, 350)
        self.machine = machine_client
        
        layout = QVBoxLayout()
        
        # Grid for valves
        grid_group = QGroupBox("阀门测试")
        grid = QGridLayout()
        
        # Headers
        grid.addWidget(QLabel("阀门名称"), 0, 0)
        grid.addWidget(QLabel("PIN ID"), 0, 1)
        grid.addWidget(QLabel("状态"), 0, 2)
        grid.addWidget(QLabel("时长(s)"), 0, 3)
        grid.addWidget(QLabel("操作"), 0, 4)
        
        # Valve Definitions
        # format: (Display Name, PIN Macro Name, Default Time)
        self.valves = [
            ("颜色阀 A1", "A1_VALVE", 0.5),
            ("颜色阀 A2", "A2_VALVE", 0.5),
            ("颜色阀 A3", "A3_VALVE", 0.5),
            ("颜色阀 A4", "A4_VALVE", 0.5),
            ("总控阀 A", "VALVE_A_CTRL", 0.5),
            ("总控阀 B", "VALVE_B_CTRL", 0.5),
        ]
        
        self.spin_boxes = {}
        self.status_labels = {} # Store labels to update color
        
        for i, (name, pin, default_t) in enumerate(self.valves):
            row = i + 1
            # 1. Name
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("font-weight: bold; font-size: 14px;")
            grid.addWidget(lbl_name, row, 0)
            
            # 2. PIN
            lbl_pin = QLabel(pin)
            lbl_pin.setStyleSheet("color: #6B7280; font-family: Consolas;")
            grid.addWidget(lbl_pin, row, 1)

            # 3. Status Light
            lbl_status = QLabel("●")
            # Default Gray
            lbl_status.setStyleSheet("color: #9CA3AF; font-size: 24px;") 
            lbl_status.setAlignment(Qt.AlignCenter)
            self.status_labels[pin] = lbl_status
            grid.addWidget(lbl_status, row, 2)
            
            # 4. Time
            spn = QDoubleSpinBox()
            spn.setRange(0.01, 10.0)
            spn.setValue(default_t)
            spn.setSingleStep(0.1)
            spn.setFixedWidth(60)
            self.spin_boxes[pin] = spn
            grid.addWidget(spn, row, 3)
            
            # 5. Test Button
            btn = QPushButton("开启测试")
            btn.setStyleSheet("background-color: #3B82F6; color: white; padding: 4px 8px;")
            # Use closure
            btn.clicked.connect(lambda checked, p=pin: self.test_single_valve(p))
            grid.addWidget(btn, row, 4)
            
        grid_group.setLayout(grid)
        layout.addWidget(grid_group)
        
        layout.addStretch()
        
        # All Off Button
        btn_all_off = QPushButton("⛔ 全部关闭 (ALL OFF)")
        btn_all_off.setMinimumHeight(45)
        btn_all_off.setStyleSheet("""
            QPushButton {
                background-color: #EF4444; 
                color: white; 
                font-weight: bold; 
                font-size: 15px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #DC2626; }
            QPushButton:pressed { background-color: #B91C1C; }
        """)
        btn_all_off.clicked.connect(self.all_valves_off)
        layout.addWidget(btn_all_off)
        
        self.setLayout(layout)

    def test_single_valve(self, pin):
        t = self.spin_boxes[pin].value()
        
        # 1. UI: Turn Green
        if pin in self.status_labels:
            self.status_labels[pin].setStyleSheet("color: #10B981; font-size: 24px;") # Green

        # 2. Call Machine
        if self.machine:
             self.machine.test_valve_macro(pin, t)

        # 3. Timer to reset to Gray
        QTimer.singleShot(int(t * 1000), lambda: self._reset_status(pin))

    def _reset_status(self, pin):
        if pin in self.status_labels:
            self.status_labels[pin].setStyleSheet("color: #9CA3AF; font-size: 24px;") # Gray

    def all_valves_off(self):
        if self.machine:
            self.machine.all_valves_off()
        # Reset all lights immediately
        for pin in self.status_labels:
            self._reset_status(pin)
