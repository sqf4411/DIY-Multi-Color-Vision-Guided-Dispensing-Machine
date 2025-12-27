# 对刀对话框集成示例
# 在 main_camera_ui.py 的 MainWindow 类中添加以下代码

"""
===========================================
1. 在文件顶部导入对话框
===========================================
"""
from core.tool_set_dialog import ToolSetDialog

"""
===========================================
2. 在 UI 初始化中添加对刀按钮
===========================================
例如在运动控制区域添加：
"""
def init_tool_set_button(self):
    # 在您的UI布局中添加此按钮
    btn_tool_set = QPushButton("🔧 一键对刀")
    btn_tool_set.setStyleSheet("background-color: #F59E0B; min-height: 35px;")
    btn_tool_set.clicked.connect(self.open_tool_set_dialog)
    # 添加到您的布局中，例如：
    # your_layout.addWidget(btn_tool_set)

"""
===========================================
3. 添加对话框打开方法
===========================================
"""
def open_tool_set_dialog(self):
    """打开对刀设置对话框"""
    # 获取当前机器位置
    current_pos = (self.pos_x, self.pos_y, self.pos_z)
    
    # 打开对话框
    dialog = ToolSetDialog(self, current_pos)
    
    if dialog.exec_() == QDialog.Accepted:
        # 检查用户选择的操作
        if hasattr(dialog, 'result_action') and dialog.result_action == "auto_tool_set":
            # 用户点击了"开始自动对刀"
            self.execute_auto_tool_set(dialog.get_tool_setter_pos())

"""
===========================================
4. 添加自动对刀执行方法
===========================================
"""
def execute_auto_tool_set(self, tool_setter_pos):
    """执行自动对刀"""
    if not self.machine.connected:
        QMessageBox.warning(self, "错误", "❌ 机器未连接！")
        return
    
    x, y, z = tool_setter_pos
    self.log(f"🎯 开始自动对刀，目标位置: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
    
    # 创建并启动自动对刀线程
    # 注意：现在传入 target_z 参数！
    self.tool_set_worker = AutoToolSetWorker(
        self.machine, 
        safe_z=150.0,  # 安全高度
        target_x=x, 
        target_y=y,
        target_z=z  # ← 新增：对刀仪Z坐标
    )
    
    self.tool_set_worker.finished_signal.connect(self.on_tool_set_finished)
    self.tool_set_worker.start()

"""
===========================================
5. 添加对刀完成回调
===========================================
"""
def on_tool_set_finished(self, success, new_z_base, message):
    """对刀完成回调"""
    if success:
        self.log(f"✅ {message}")
        self.log(f"📊 精确对刀高度: Z={new_z_base:.3f}")
        
        # 可以选择保存这个精确值
        QMessageBox.information(
            self, 
            "对刀成功", 
            f"✅ 对刀完成！\n\n精确高度: Z={new_z_base:.3f} mm\n\n{message}"
        )
    else:
        self.log(f"❌ {message}")
        QMessageBox.warning(self, "对刀失败", f"❌ {message}")

"""
===========================================
完整使用流程
===========================================
1. 用户点击"一键对刀"按钮
2. 如果是首次：保存当前位置 → 完成
3. 如果已有数据：
   - 点击"开始自动对刀" → 自动探测精确高度
   - 点击"重新标定位置" → 修改XYZ坐标 → 保存
4. 自动对刀完成后，显示精确Z高度
"""
