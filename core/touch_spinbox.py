"""
SpinBox触屏支持模块
为QDoubleSpinBox添加触屏输入功能
"""

from PyQt5.QtWidgets import QDoubleSpinBox, QWidget, QLineEdit
from PyQt5.QtCore import Qt, QObject, QEvent
from core.touch_numpad import TouchNumberPad


class TouchFilter(QObject):
    """触屏事件过滤器，用于拦截单击事件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def eventFilter(self, obj, event):
        # 监听鼠标释放事件（模拟单击）
        if event.type() == QEvent.MouseButtonRelease:
            # 只有在触屏模式开启，且是左键点击时才触发
            if TouchSpinBox.touch_mode_enabled and event.button() == Qt.LeftButton:
                # 找到所属的 SpinBox
                spinbox = None
                if isinstance(obj, QDoubleSpinBox):
                    spinbox = obj
                elif isinstance(obj, QLineEdit) and isinstance(obj.parent(), QDoubleSpinBox):
                    spinbox = obj.parent()
                
                if spinbox:
                    # 弹出键盘
                    title = spinbox.objectName() or "输入数值"
                    value, ok = TouchNumberPad.get_number(
                        title=title,
                        current_value=spinbox.value(),
                        min_val=spinbox.minimum(),
                        max_val=spinbox.maximum(),
                        parent=spinbox.window()
                    )
                    if ok:
                        spinbox.setValue(value)
                    
                    # 消费事件，防止默认行为（如弹出光标）
                    # 视情况而定，如果为了体验好，也可以 return False 让它继续获得焦点
                    # return True to consume
                    return True
                    
        return super().eventFilter(obj, event)


class TouchSpinBox(QDoubleSpinBox):
    """支持触屏输入的SpinBox"""
    
    # 全局触屏模式标志
    touch_mode_enabled = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._title = "输入数值"
        # 安装事件过滤器
        self._filter = TouchFilter(self)
        self.installEventFilter(self._filter)
        self.lineEdit().installEventFilter(self._filter)
    
    def set_title(self, title):
        """设置触屏键盘标题"""
        self._title = title
        return self
    
    @staticmethod
    def set_global_touch_mode(enabled):
        """全局启用/禁用触屏模式"""
        TouchSpinBox.touch_mode_enabled = enabled
        print(f"[TouchSpinBox] 触屏模式: {'启用' if enabled else '禁用'}")


def apply_touch_support(widget: QWidget):
    """
    为Widget内所有的QDoubleSpinBox添加触屏支持
    无论它们已经是TouchSpinBox还是普通的QDoubleSpinBox
    """
    filter_instance = TouchFilter(widget)
    # 保持引用防止被垃圾回收
    widget._touch_filter = filter_instance
    
    # 查找所有SpinBox
    spinboxes = widget.findChildren(QDoubleSpinBox)
    count = 0
    for sp in spinboxes:
        # 避免重复安装
        if getattr(sp, '_touch_hooked', False):
            continue
            
        sp.installEventFilter(filter_instance)
        sp.lineEdit().installEventFilter(filter_instance)
        sp._touch_hooked = True
        count += 1
        
    print(f"[TouchSpinBox] 已为 {widget.objectName() or 'Widget'} 的 {count} 个SpinBox安装触屏支持")
