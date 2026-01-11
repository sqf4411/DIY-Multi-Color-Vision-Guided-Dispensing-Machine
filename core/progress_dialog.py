from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QProgressBar, 
                             QPushButton, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt

class ProcessingDialog(QDialog):
    """
    通用进度对话框
    支持: 进度条, 状态文本, 暂停/继续, 停止
    """
    pause_signal = pyqtSignal()
    resume_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self, title="Processing", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(400)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint) # No close button X
        # Prevent closing by ESC
        self.setModal(True)

        self.layout = QVBoxLayout()
        
        # Status Label
        self.lbl_status = QLabel("正在准备...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.lbl_status)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                width: 20px;
            }
        """)
        self.layout.addWidget(self.progress_bar)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setCheckable(True) # Toggle
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #F59E0B; 
                color: white; 
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #10B981;
                text: "继续";
            }
            QPushButton:hover {
                background-color: #D97706;
            }
        """)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        btn_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #DC2626; 
                color: white; 
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #B91C1C;
            }
        """)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        btn_layout.addWidget(self.btn_stop)

        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)
        
        self.is_paused = False

    def update_status(self, msg):
        self.lbl_status.setText(msg)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setFormat(f"{current}/{total} ({percent}%)")

    def on_pause_clicked(self):
        if self.btn_pause.isChecked():
            # Paused -> Change to Resume
            self.btn_pause.setText("继续")
            self.lbl_status.setText("已暂停，等待继续...")
            self.pause_signal.emit()
            self.is_paused = True
        else:
            # Resumed -> Change to Pause
            self.btn_pause.setText("暂停")
            self.lbl_status.setText("正在恢复...")
            self.resume_signal.emit()
            self.is_paused = False

    def on_stop_clicked(self):
        ret = QMessageBox.question(self, "确认停止", "确定要停止当前任务吗？", 
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.lbl_status.setText("正在停止...")
            self.stop_signal.emit()
            # Don't close immediately, wait for thread to finish signal
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)

    def force_close(self):
        """Mandatory close (used by parent when task finished)"""
        self._allow_close = True
        self.accept()

    def closeEvent(self, event):
        if getattr(self, '_allow_close', False) or (self.isVisible() and not self.btn_stop.isEnabled()):
             super().closeEvent(event)
        else:
            # Ignore user close attempts (Alt+F4) unless explicitly allowed
            event.ignore()
