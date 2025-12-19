from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import json
from pathlib import Path
from datetime import datetime

from core.touch_spinbox import apply_touch_support

class ValveControlDialog(QDialog):
    def __init__(self, parent=None, machine_client=None):
        super().__init__(parent)
        self.setWindowTitle("电磁阀测试与出胶控制")
        self.resize(700, 850)
        self.machine = machine_client
        
        # 阀门定义
        self.valves = [
            ("颜色阀 A1", "A1_VALVE", 0.5),
            ("颜色阀 A2", "A2_VALVE", 0.5),
            ("颜色阀 A3", "A3_VALVE", 0.5),
            ("颜色阀 A4", "A4_VALVE", 0.5),
            ("总控阀 A", "VALVE_A_CTRL", 0.5),
            ("总控阀 B", "VALVE_B_CTRL", 0.5),
        ]
        
        # 颜色配置
        self.color_configs = {
            "A1_VALVE": {"name": "红色", "color": "#EF4444", "emoji": "🔴"},
            "A2_VALVE": {"name": "绿色", "color": "#10B981", "emoji": "🟢"},
            "A3_VALVE": {"name": "蓝色", "color": "#3B82F6", "emoji": "🔵"},
            "A4_VALVE": {"name": "黄色", "color": "#F59E0B", "emoji": "🟡"},
        }
        
        self.spin_boxes = {}
        self.status_labels = {}
        self.all_test_buttons = []  # 存储所有测试按钮
        
        self.init_ui()
        
        # 🚀 启动时自动加载保存的参数
        self.load_params_from_file_silent()
    
    def showEvent(self, event):
        """对话框显示时重新加载参数（确保加载新产品后参数正确）"""
        super().showEvent(event)
        # 每次显示对话框时重新加载参数
        self.load_params_from_file_silent()
        # ⭐ 确保所有SpinBox都启用了触屏支持
        apply_touch_support(self)
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # ====== 1. 参数设置区域 ======
        params_group = QGroupBox("⚙️ 出胶参数设置")
        params_layout = QVBoxLayout()
        
        # 三个时间参数
        time_grid = QGridLayout()
        
        # 供料提前时间
        lbl_prefeed = QLabel("供料提前时间")
        lbl_prefeed.setStyleSheet("font-weight: bold;")
        self.spn_prefeed_time = QDoubleSpinBox()
        self.spn_prefeed_time.setRange(0.01, 2.0)
        self.spn_prefeed_time.setValue(0.10)
        self.spn_prefeed_time.setSingleStep(0.01)
        self.spn_prefeed_time.setSuffix(" 秒")
        time_grid.addWidget(lbl_prefeed, 0, 0)
        time_grid.addWidget(self.spn_prefeed_time, 0, 1)
        
        # 阀A出胶时间
        lbl_valve_a = QLabel("阀A出胶时间")
        lbl_valve_a.setStyleSheet("font-weight: bold; color: #3B82F6;")
        self.spn_valve_a_time = QDoubleSpinBox()
        self.spn_valve_a_time.setRange(0.01, 10.0)
        self.spn_valve_a_time.setValue(0.50)
        self.spn_valve_a_time.setSingleStep(0.05)
        self.spn_valve_a_time.setSuffix(" 秒")
        time_grid.addWidget(lbl_valve_a, 0, 2)
        time_grid.addWidget(self.spn_valve_a_time, 0, 3)
        
        # 阀B出胶时间
        lbl_valve_b = QLabel("阀B出胶时间")
        lbl_valve_b.setStyleSheet("font-weight: bold; color: #10B981;")
        self.spn_valve_b_time = QDoubleSpinBox()
        self.spn_valve_b_time.setRange(0.01, 10.0)
        self.spn_valve_b_time.setValue(0.50)
        self.spn_valve_b_time.setSingleStep(0.05)
        self.spn_valve_b_time.setSuffix(" 秒")
        time_grid.addWidget(lbl_valve_b, 1, 2)
        time_grid.addWidget(self.spn_valve_b_time, 1, 3)
        
        params_layout.addLayout(time_grid)
        
        # 参数管理按钮（移除加载按钮，因为加载配方时会自动加载）
        btn_layout = QHBoxLayout()
        
        btn_save = QPushButton("💾 保存参数（临时）")
        btn_save.setToolTip("保存当前参数供快速测试\n真正保存到配方需使用主UI的'保存配方'按钮")
        btn_save.clicked.connect(self.save_params_to_file)
        btn_save.setStyleSheet("background-color: #10B981; color: white; padding: 8px; font-weight: bold;")
        
        btn_reset = QPushButton("🔄 恢复默认")
        btn_reset.clicked.connect(self.reset_to_defaults)
        btn_reset.setStyleSheet("background-color: #6B7280; color: white; padding: 8px;")
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        
        params_layout.addLayout(btn_layout)
        
        # 添加提示标签
        lbl_hint = QLabel("💡 提示：加载产品配方时会自动应用该产品的阀门参数")
        lbl_hint.setStyleSheet("color: #6B7280; font-size: 11px; font-style: italic;")
        params_layout.addWidget(lbl_hint)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # ====== 2. 点胶阀A测试 ======
        valve_a_group = QGroupBox("🎯 点胶阀A测试（单阀调试，调节A出胶量）")
        valve_a_layout = QVBoxLayout()
        
        # 4个颜色按钮
        a_btn_layout = QHBoxLayout()
        for valve_pin in ["A1_VALVE", "A2_VALVE", "A3_VALVE", "A4_VALVE"]:
            config = self.color_configs[valve_pin]
            btn = QPushButton(f"{config['emoji']} A+{valve_pin[:2]}\n{config['name']}出胶")
            btn.setMinimumHeight(60)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {config['color']};
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    opacity: 0.8;
                }}
            """)
            btn.clicked.connect(lambda checked, p=valve_pin: self.test_valve_a_with_color(p))
            self.all_test_buttons.append(btn)
            a_btn_layout.addWidget(btn)
        
        valve_a_layout.addLayout(a_btn_layout)
        
        # 提示
        lbl_a_hint = QLabel("💡 提示：调节「阀A出胶时间」后测试，称重记录")
        lbl_a_hint.setStyleSheet("color: #6B7280; font-size: 11px; font-style: italic;")
        valve_a_layout.addWidget(lbl_a_hint)
        
        valve_a_group.setLayout(valve_a_layout)
        layout.addWidget(valve_a_group)
        
        # ====== 3. 点胶阀B测试 ======
        valve_b_group = QGroupBox("🎯 点胶阀B测试（单阀调试，调节B出胶量）")
        valve_b_layout = QVBoxLayout()
        
        btn_b_test = QPushButton("⚪ B单独测试\n(第二点胶阀)")
        btn_b_test.setMinimumHeight(60)
        btn_b_test.setFixedWidth(180)
        btn_b_test.setStyleSheet("""
            QPushButton {
                background-color: #8B5CF6;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #7C3AED;
            }
        """)
        btn_b_test.clicked.connect(self.test_valve_b_only)
        self.all_test_buttons.append(btn_b_test)
        
        b_btn_layout = QHBoxLayout()
        b_btn_layout.addWidget(btn_b_test)
        b_btn_layout.addStretch()
        valve_b_layout.addLayout(b_btn_layout)
        
        # 提示
        lbl_b_hint = QLabel("💡 提示：调节「阀B出胶时间」后测试，称重记录")
        lbl_b_hint.setStyleSheet("color: #6B7280; font-size: 11px; font-style: italic;")
        valve_b_layout.addWidget(lbl_b_hint)
        
        valve_b_group.setLayout(valve_b_layout)
        layout.addWidget(valve_b_group)
        
        # ====== 4. 双阀混合测试 ======
        dual_group = QGroupBox("🔬 双阀混合测试（1:1环氧胶生产）")
        dual_layout = QVBoxLayout()
        
        # 4个混合按钮
        dual_btn_layout = QHBoxLayout()
        for valve_pin in ["A1_VALVE", "A2_VALVE", "A3_VALVE", "A4_VALVE"]:
            config = self.color_configs[valve_pin]
            btn = QPushButton(f"{config['emoji']} A+B+{valve_pin[:2]}\n混合{config['name']}胶")
            btn.setMinimumHeight(60)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {config['color']};
                    color: white;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 8px;
                    border: 2px solid #FFFFFF;
                }}
                QPushButton:hover {{
                    opacity: 0.9;
                }}
            """)
            btn.clicked.connect(lambda checked, p=valve_pin: self.test_dual_valve_mixed(p))
            self.all_test_buttons.append(btn)
            dual_btn_layout.addWidget(btn)
        
        dual_layout.addLayout(dual_btn_layout)
        
        # 提示
        lbl_dual_hint = QLabel("💡 提示：使用已校准的A/B时间进行1:1混合点胶")
        lbl_dual_hint.setStyleSheet("color: #6B7280; font-size: 11px; font-style: italic;")
        dual_layout.addWidget(lbl_dual_hint)
        
        dual_group.setLayout(dual_layout)
        layout.addWidget(dual_group)
        
        # ====== 5. 单阀门独立测试 ======
        grid_group = QGroupBox("🔧 单阀门独立测试（低级调试）")
        grid = QGridLayout()
        
        # Headers
        grid.addWidget(QLabel("阀门名称"), 0, 0)
        grid.addWidget(QLabel("PIN ID"), 0, 1)
        grid.addWidget(QLabel("状态"), 0, 2)
        grid.addWidget(QLabel("时长(s)"), 0, 3)
        grid.addWidget(QLabel("操作"), 0, 4)
        
        for i, (name, pin, default_t) in enumerate(self.valves):
            row = i + 1
            # 1. Name
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("font-weight: bold; font-size: 12px;")
            grid.addWidget(lbl_name, row, 0)
            
            # 2. PIN  
            lbl_pin = QLabel(pin)
            lbl_pin.setStyleSheet("color: #6B7280; font-family: Consolas; font-size: 11px;")
            grid.addWidget(lbl_pin, row, 1)

            # 3. Status Light
            lbl_status = QLabel("●")
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
            btn.setStyleSheet("background-color: #3B82F6; color: white; padding: 4px 8px; font-size: 11px;")
            btn.clicked.connect(lambda checked, p=pin: self.test_single_valve(p))
            grid.addWidget(btn, row, 4)
            
        grid_group.setLayout(grid)
        layout.addWidget(grid_group)
        
        # ====== 6. 全部关闭按钮 ======
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
    
    # ========== 测试方法 ==========
    
    def test_valve_a_with_color(self, color_valve_pin):
        """点胶阀A + 颜色测试"""
        t_pre = self.spn_prefeed_time.value()
        t_dispense_a = self.spn_valve_a_time.value()
        
        self._disable_all_test_buttons()
        
        # 1. 启动颜色阀
        if self.machine:
            self.machine.set_pin(color_valve_pin, 1)
        self._set_valve_status(color_valve_pin, True)
        
        # 2. 延迟后启动A
        QTimer.singleShot(int(t_pre * 1000), 
            lambda: self._start_valve_a())
        
        # 3. 延迟后关闭A
        QTimer.singleShot(int((t_pre + t_dispense_a) * 1000), 
            lambda: self._stop_valve_a())
        
        # 4. 最后关闭颜色阀并恢复按钮
        QTimer.singleShot(int((t_pre + t_dispense_a + 0.1) * 1000), 
            lambda: self._cleanup(color_valve_pin))
    
    def test_valve_b_only(self):
        """点胶阀B单独测试"""
        t_dispense_b = self.spn_valve_b_time.value()
        
        self._disable_all_test_buttons()
        
        # 1. 启动B
        if self.machine:
            self.machine.set_pin("VALVE_B_CTRL", 1)
        self._set_valve_status("VALVE_B_CTRL", True)
        
        # 2. 延迟后关闭B并恢复按钮
        QTimer.singleShot(int(t_dispense_b * 1000), 
            lambda: self._cleanup("VALVE_B_CTRL"))
    
    def test_dual_valve_mixed(self, color_valve_pin):
        """双阀混合测试"""
        t_pre = self.spn_prefeed_time.value()
        t_dispense_a = self.spn_valve_a_time.value()
        t_dispense_b = self.spn_valve_b_time.value()
        
        self._disable_all_test_buttons()
        
        # 1. 启动颜色阀
        if self.machine:
            self.machine.set_pin(color_valve_pin, 1)
        self._set_valve_status(color_valve_pin, True)
        
        # 2. 延迟后同时启动A和B
        QTimer.singleShot(int(t_pre * 1000), 
            lambda: self._start_dual_valves())
        
        # 3. 根据A和B不同的时间分别关闭
        QTimer.singleShot(int((t_pre + t_dispense_a) * 1000), 
            lambda: self._stop_valve_a())
        QTimer.singleShot(int((t_pre + t_dispense_b) * 1000), 
            lambda: self._stop_valve_b())
        
        # 4. 最后关闭颜色阀（取较长时间+0.1）
        max_time = max(t_dispense_a, t_dispense_b)
        QTimer.singleShot(int((t_pre + max_time + 0.1) * 1000), 
            lambda: self._cleanup(color_valve_pin))
    
    def test_single_valve(self, pin):
        """单阀门独立测试"""
        t = self.spin_boxes[pin].value()
        
        if pin in self.status_labels:
            self.status_labels[pin].setStyleSheet("color: #EF4444; font-size: 24px;")  # 红色

        if self.machine:
             self.machine.test_valve_macro(pin, t)

        QTimer.singleShot(int(t * 1000), lambda: self._reset_status(pin))
    
    # ========== 辅助方法 ==========
    
    def _start_valve_a(self):
        if self.machine:
            self.machine.set_pin("VALVE_A_CTRL", 1)
        self._set_valve_status("VALVE_A_CTRL", True)
    
    def _stop_valve_a(self):
        if self.machine:
            self.machine.set_pin("VALVE_A_CTRL", 0)
        self._set_valve_status("VALVE_A_CTRL", False)
    
    def _start_valve_b(self):
        if self.machine:
            self.machine.set_pin("VALVE_B_CTRL", 1)
        self._set_valve_status("VALVE_B_CTRL", True)
    
    def _stop_valve_b(self):
        if self.machine:
            self.machine.set_pin("VALVE_B_CTRL", 0)
        self._set_valve_status("VALVE_B_CTRL", False)
    
    def _start_dual_valves(self):
        if self.machine:
            self.machine.set_pin("VALVE_A_CTRL", 1)
            self.machine.set_pin("VALVE_B_CTRL", 1)
        self._set_valve_status("VALVE_A_CTRL", True)
        self._set_valve_status("VALVE_B_CTRL", True)
    
    def _stop_dual_valves(self):
        if self.machine:
            self.machine.set_pin("VALVE_A_CTRL", 0)
            self.machine.set_pin("VALVE_B_CTRL", 0)
        self._set_valve_status("VALVE_A_CTRL", False)
        self._set_valve_status("VALVE_B_CTRL", False)
    
    def _cleanup(self, pin):
        if self.machine:
            self.machine.set_pin(pin, 0)
        self._set_valve_status(pin, False)
        self._enable_all_test_buttons()
    
    def _set_valve_status(self, pin, is_on):
        if pin in self.status_labels:
            color = "#EF4444" if is_on else "#9CA3AF"  # 红色=开启, 灰色=关闭
            self.status_labels[pin].setStyleSheet(f"color: {color}; font-size: 24px;")
    
    def _reset_status(self, pin):
        if pin in self.status_labels:
            self.status_labels[pin].setStyleSheet("color: #9CA3AF; font-size: 24px;")
    
    def _disable_all_test_buttons(self):
        for btn in self.all_test_buttons:
            btn.setEnabled(False)
    
    def _enable_all_test_buttons(self):
        for btn in self.all_test_buttons:
            btn.setEnabled(True)
    
    def all_valves_off(self):
        """全部关闭"""
        if self.machine:
            self.machine.all_valves_off()
        for pin in self.status_labels:
            self._reset_status(pin)
        self._enable_all_test_buttons()
    
    # ========== 参数管理 ==========
    
    def save_params_to_file(self):
        """保存当前参数到文件"""
        params = {
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "prefeed_time": self.spn_prefeed_time.value(),
                "valve_a_dispense_time": self.spn_valve_a_time.value(),
                "valve_b_dispense_time": self.spn_valve_b_time.value()
            },
            "notes": ""
        }
        
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        try:
            with open(config_dir / "valve_dispense_params.json", 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "成功", 
                f"参数已保存！\n\n"
                f"供料提前：{params['parameters']['prefeed_time']}秒\n"
                f"阀A时间：{params['parameters']['valve_a_dispense_time']}秒\n"
                f"阀B时间：{params['parameters']['valve_b_dispense_time']}秒")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")
    
    def load_params_from_file(self):
        """从文件加载参数"""
        config_file = Path("config") / "valve_dispense_params.json"
        
        if not config_file.exists():
            QMessageBox.warning(self, "提示", "未找到保存的参数文件")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            params = data["parameters"]
            self.spn_prefeed_time.setValue(params["prefeed_time"])
            self.spn_valve_a_time.setValue(params["valve_a_dispense_time"])
            self.spn_valve_b_time.setValue(params["valve_b_dispense_time"])
            
            last_updated = data.get("last_updated", "未知")
            QMessageBox.information(self, "成功", 
                f"参数已加载！\n\n上次保存时间：{last_updated}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载参数失败：{e}")
    
    def reset_to_defaults(self):
        """恢复默认参数"""
        reply = QMessageBox.question(self, "确认", 
            "确定要恢复默认参数吗？",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.spn_prefeed_time.setValue(0.10)
            self.spn_valve_a_time.setValue(0.50)
            self.spn_valve_b_time.setValue(0.50)
            QMessageBox.information(self, "成功", "已恢复默认参数")
    
    def load_params_from_file_silent(self):
        """静默加载参数（启动时用）"""
        config_file = Path("config") / "valve_dispense_params.json"
        print(f"[ValveControl] 尝试加载参数文件: {config_file.absolute()}")
        print(f"[ValveControl] 文件存在: {config_file.exists()}")
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                params = data["parameters"]
                
                print(f"[ValveControl] 读取到的参数: {params}")
                
                self.spn_prefeed_time.setValue(params["prefeed_time"])
                self.spn_valve_a_time.setValue(params["valve_a_dispense_time"])
                self.spn_valve_b_time.setValue(params["valve_b_dispense_time"])
                
                print(f"[ValveControl] ✅ 已加载参数: 供料={params['prefeed_time']}, A={params['valve_a_dispense_time']}, B={params['valve_b_dispense_time']}")
            except Exception as e:
                print(f"[ValveControl] ❌ 加载参数失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[ValveControl] ⚠️ 参数文件不存在，使用默认值")
