from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt
import sys

from machine_client import MachineClient


class TestKlipperWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klipper 连接测试（纯净版）")
        self.resize(500, 200)

        self.machine = MachineClient()
        self.default_base_url = "http://192.168.60.26"

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 一行：地址 + 按钮 + 状态
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("地址:"))

        self.edit_base_url = QLineEdit(self.default_base_url)
        self.edit_base_url.setPlaceholderText("例如: http://192.168.60.26:7125")
        conn_layout.addWidget(self.edit_base_url)

        self.btn_connect = QPushButton("连接")
        conn_layout.addWidget(self.btn_connect)

        self.lbl_status = QLabel("未连接")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        conn_layout.addWidget(self.lbl_status)

        main_layout.addLayout(conn_layout)

        # 日志框
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

        # 信号
        self.btn_connect.clicked.connect(self.on_connect_clicked)

    def log(self, text: str):
        self.log_box.append(text)

    def on_connect_clicked(self):
        base_url = self.edit_base_url.text().strip()
        if not base_url:
            QMessageBox.warning(self, "错误", "请先输入地址")
            return

        self.machine.configure(base_url)
        ok = self.machine.connect()
        if ok:
            self.lbl_status.setText("已连接(本地)")
            self.log(f"✅ (stub) 已标记连接到: {base_url}")
        else:
            self.lbl_status.setText("未连接")
            self.log(f"⚠ (stub) 连接失败: {base_url}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestKlipperWindow()
    w.show()
    sys.exit(app.exec_())
