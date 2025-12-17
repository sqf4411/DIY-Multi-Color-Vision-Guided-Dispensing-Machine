"""
路径绘制画布组件
支持点和线的绘制，多颜色显示
"""
from PyQt5.QtWidgets import QWidget, QScrollArea
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QColor, QBrush, QFont, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import cv2
import numpy as np
from .path_segment import PathSegment

class PathCanvas(QWidget):
    """路径绘制画布"""
    
    segment_added = pyqtSignal(PathSegment)  # 添加路径段信号
    segment_clicked = pyqtSignal(int)  # 点击路径段信号（索引）
    batch_edit_requested = pyqtSignal()  # 批量编辑请求信号
    
    def __init__(self, template_img, parent=None):
        super().__init__(parent)
        self.template = template_img
        self.segments = []  # List[PathSegment]
        
        # 当前绘制状态
        self.current_mode = 'dot'  # 'dot' 或 'line'
        self.current_color = 0     # 0-3
        self.current_dwell = 100   # ms
        self.current_line = []     # 正在绘制的线
        
        # 选中状态
        self.selected_indices = set()  # 选中的路径段索引
        
        # 缩放
        self.scale = 1.0
        self.calculate_scale()
        
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        
        # 背景色
        self.setStyleSheet("background-color: #2C2C2C;")

        # 拖拽相关
        self.last_pan_pos = QPoint()
        self.is_panning = False
    
    def calculate_scale(self):
        """计算缩放比例"""
        if self.template is None:
            return
        h, w = self.template.shape[:2]
        # 初始缩放：让图像更大一些
        self.scale = 2.0
        
        # 更新widget大小
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)
        self.setFixedSize(scaled_w, scaled_h)
    
    def set_mode(self, mode: str):
        """设置绘制模式"""
        self.current_mode = mode
        if mode == 'dot' and self.current_line:
            # 切换到点模式时，结束当前线
            self.finish_current_line()
    
    def set_color(self, color: int):
        """设置当前颜色"""
        self.current_color = color
    
    def set_dwell(self, dwell_ms: int):
        """设置出胶时间"""
        self.current_dwell = dwell_ms
    
    def set_segments(self, segments):
        """设置路径段列表"""
        self.segments = segments
        self.update()
    
    def set_selected_indices(self, indices):
        """设置选中的路径段索引"""
        self.selected_indices = set(indices)
        self.update()
    
    def finish_current_line(self):
        """完成当前正在绘制的线"""
        if len(self.current_line) >= 2:
            seg = PathSegment(
                seg_type='line',
                color=self.current_color,
                points=self.current_line.copy(),
                dwell_ms=self.current_dwell
            )
            self.segment_added.emit(seg)
            self.current_line = []
            self.update()
    
    def mousePressEvent(self, event):
        """鼠标点击"""
        # 转换坐标
        x = event.x() / self.scale
        y = event.y() / self.scale
        
        # 检测是否点击了已有路径段
        clicked_index = self.find_segment_at(x, y)
        
        # 中键拖拽逻辑
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.last_pan_pos = event.globalPos()
            self.is_panning = True
            return

        if event.button() == Qt.RightButton:
            # 右键：如果点击了已选中的路径段，什么都不做（保持选中）
            # 如果点击了未选中的路径段，选中它
            # 如果没点击路径段且在线模式，结束线
            if clicked_index is not None:
                # 点击了路径段
                if clicked_index not in self.selected_indices:
                    # 未选中，选中它（用于右键菜单）
                    self.segment_clicked.emit(clicked_index)
                # 如果已选中，什么都不做，保持选中状态等待右键菜单
                return
            else:
                # 没点击路径段，线模式下结束当前线
                if self.current_mode == 'line' and self.current_line:
                    self.finish_current_line()
                return
        
        # 左键逻辑
        if clicked_index is not None:
            # 点击了路径段，发送选择信号
            self.segment_clicked.emit(clicked_index)
            return
        
        # 没有点击已有路径段，说明是要添加新点/线
        # 清除选中状态
        self.selected_indices.clear()
        self.update()
        
        if event.button() == Qt.LeftButton:
            # 左键逻辑
            if self.current_mode == 'dot':
                # 点模式：添加点
                seg = PathSegment(
                    seg_type='dot',
                    color=self.current_color,
                    points=[(x, y)],
                    dwell_ms=self.current_dwell
                )
                self.segment_added.emit(seg)
                self.update()
            
            elif self.current_mode == 'line':
                # 线模式：添加线的节点
                self.current_line.append((x, y))
                self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动"""
        if self.is_panning:
            delta = event.globalPos() - self.last_pan_pos
            self.last_pan_pos = event.globalPos()
            
            # 查找父级 ScrollArea 并滚动
            parent = self.parent()
            while parent:
                if isinstance(parent, QScrollArea):
                    scrollbar_h = parent.horizontalScrollBar()
                    scrollbar_v = parent.verticalScrollBar()
                    scrollbar_h.setValue(scrollbar_h.value() - delta.x())
                    scrollbar_v.setValue(scrollbar_v.value() - delta.y())
                    break
                parent = parent.parent()
            return
            
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            self.is_panning = False
            
    def wheelEvent(self, event):
        """滚轮事件：缩放"""
        if self.template is None:
            return
        
        # 获取滚轮方向
        delta = event.angleDelta().y()
        
        # 计算新缩放比例
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_scale = self.scale * zoom_factor
        
        # 计算最小缩放比例（适应窗口大小）
        min_scale = 0.1
        parent = self.parent()
        while parent:
            if isinstance(parent, QScrollArea):
                viewport_w = parent.viewport().width()
                viewport_h = parent.viewport().height()
                img_h, img_w = self.template.shape[:2]
                
                # 计算宽高比
                w_ratio = viewport_w / img_w
                h_ratio = viewport_h / img_h
                
                # 取较小值作为最小缩放，保证能完全显示
                min_scale = min(w_ratio, h_ratio) * 0.9  # 稍微留点边距
                break
            parent = parent.parent()

        # 限制缩放范围
        # 最小：适应窗口
        # 最大：10.0 (放大可以限制)
        new_scale = max(min_scale, min(10.0, new_scale))
        
        if new_scale != self.scale:
            self.scale = new_scale
            h, w = self.template.shape[:2]
            scaled_w = int(w * self.scale)
            scaled_h = int(h * self.scale)
            self.setFixedSize(scaled_w, scaled_h)
            self.update()
    
    def contextMenuEvent(self, event):
        """右键菜单"""
        # 如果有选中的路径段，触发批量编辑
        if self.selected_indices:
            self.batch_edit_requested.emit()
        else:
            # 没有选中，执行默认行为
            super().contextMenuEvent(event)
    
    def find_segment_at(self, x, y):
        """查找坐标(x,y)处的路径段索引"""
        threshold = 10 / self.scale  # 点击阈值（像素）
        
        for i, seg in enumerate(self.segments):
            if seg.type == 'dot':
                # 检测点
                px, py = seg.points[0]
                dist = ((px - x)**2 + (py - y)**2) ** 0.5
                if dist < threshold:
                    return i
            else:  # line
                # 检测线段（检查是否靠近任一线段）
                for j in range(len(seg.points) - 1):
                    x1, y1 = seg.points[j]
                    x2, y2 = seg.points[j+1]
                    # 点到线段的距离
                    dist = self.point_to_line_distance(x, y, x1, y1, x2, y2)
                    if dist < threshold:
                        return i
        return None
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """计算点到线段的距离"""
        # 线段长度
        line_len = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        if line_len == 0:
            return ((px-x1)**2 + (py-y1)**2) ** 0.5
        
        # 投影参数t
        t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / (line_len**2)))
        # 投影点
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        # 距离
        return ((px-proj_x)**2 + (py-proj_y)**2) ** 0.5
    
    def paintEvent(self, event):
        """绘制"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. 绘制模板图像
        if self.template is not None:
            h, w = self.template.shape[:2]
            scaled_w = int(w * self.scale)
            scaled_h = int(h * self.scale)
            
            # 转换为QImage
            if len(self.template.shape) == 2:
                img_color = cv2.cvtColor(self.template, cv2.COLOR_GRAY2RGB)
            else:
                img_color = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
            
            qimg = QImage(img_color.data, w, h, w*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled_pixmap)
        
        # 2. 绘制所有已完成的路径段
        for idx, seg in enumerate(self.segments):
            r, g, b = seg.get_display_color_rgb()
            color = QColor(r, g, b)
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            # 检查是否被选中
            is_selected = idx in self.selected_indices
            
            if seg.type == 'dot':
                # 绘制点
                x, y = seg.points[0]
                px = int(x * self.scale)
                py = int(y * self.scale)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(px-5, py-5, 10, 10)
                
                # 选中高亮：额外绘制白色圆圈
                if is_selected:
                    painter.setPen(QPen(QColor(255, 255, 255), 3))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(px-10, py-10, 20, 20)
                
                # 绘制序号标签
                painter.setFont(QFont('Arial', 8, QFont.Bold))
                text = str(idx + 1)
                text_x = px + 8
                text_y = py - 5
                # 绘制紫色文字
                painter.setPen(QPen(QColor(255, 0, 255)))
                painter.drawText(text_x, text_y, text)
            
            else:  # line
                # 绘制线段
                for i in range(len(seg.points) - 1):
                    x1, y1 = seg.points[i]
                    x2, y2 = seg.points[i+1]
                    p1 = QPoint(int(x1 * self.scale), int(y1 * self.scale))
                    p2 = QPoint(int(x2 * self.scale), int(y2 * self.scale))
                    
                    # 选中高亮：先绘制白色粗线
                    if is_selected:
                        painter.setPen(QPen(QColor(255, 255, 255), 5))
                        painter.drawLine(p1, p2)
                        painter.setPen(QPen(color, 2))  # 恢复原色
                    
                    painter.drawLine(p1, p2)
                
                # 绘制节点
                painter.setBrush(QBrush(color))
                for x, y in seg.points:
                    px = int(x * self.scale)
                    py = int(y * self.scale)
                    painter.drawEllipse(px-4, py-4, 8, 8)
                    
                    # 选中高亮：节点外圈
                    if is_selected:
                        painter.setPen(QPen(QColor(255, 255, 255), 2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(px-7, py-7, 14, 14)
                        painter.setPen(QPen(color, 2))  # 恢复
                        painter.setBrush(QBrush(color))
                
                # 绘制序号标签（在线段中点）
                if len(seg.points) >= 2:
                    mid_idx = len(seg.points) // 2
                    mx, my = seg.points[mid_idx]
                    mpx = int(mx * self.scale)
                    mpy = int(my * self.scale)
                    
                    painter.setFont(QFont('Arial', 8, QFont.Bold))
                    text = str(idx + 1)
                    text_x = mpx + 8
                    text_y = mpy - 5
                    # 绘制紫色文字
                    painter.setPen(QPen(QColor(255, 0, 255)))
                    painter.drawText(text_x, text_y, text)
        
        # 3. 绘制正在绘制的线（虚线）
        if self.current_line:
            r, g, b = PathSegment.COLOR_RGB[self.current_color]
            color = QColor(r, g, b)
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            
            for i in range(len(self.current_line) - 1):
                x1, y1 = self.current_line[i]
                x2, y2 = self.current_line[i+1]
                p1 = QPoint(int(x1 * self.scale), int(y1 * self.scale))
                p2 = QPoint(int(x2 * self.scale), int(y2 * self.scale))
                painter.drawLine(p1, p2)
            
            # 绘制节点
            painter.setBrush(QBrush(color))
            for x, y in self.current_line:
                px = int(x * self.scale)
                py = int(y * self.scale)
                painter.drawEllipse(px-4, py-4, 8, 8)
