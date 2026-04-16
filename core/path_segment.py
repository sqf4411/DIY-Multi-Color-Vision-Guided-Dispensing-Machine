"""
路径段数据结构
支持多颜色、点和线、独立时间控制
"""
from typing import List, Tuple, Dict, Any
import numpy as np

class PathSegment:
    """单个路径段（点或线）"""
    
    # 颜色定义
    COLOR_NAMES = ['A1', 'A2', 'A3', 'A4']
    COLOR_BGR = [
        (0, 0, 255),    # 红
        (0, 255, 255),  # 黄  
        (255, 0, 0),    # 蓝
        (0, 255, 0)     # 绿
    ]
    COLOR_RGB = [
        (255, 0, 0),    # 红 (Qt用RGB)
        (255, 255, 0),  # 黄
        (0, 0, 255),    # 蓝
        (0, 255, 0)     # 绿
    ]
    
    def __init__(self, seg_type: str, color: int, points: List[Tuple[float, float]], dwell_ms: int = 100):
        """
        初始化路径段
        
        参数:
            seg_type: 'dot' 或 'line'
            color: 颜色索引 0-3 (A1-A4)
            points: 坐标列表 [(x1,y1)] 或 [(x1,y1), (x2,y2), ...]
            dwell_ms: 出胶时间(毫秒)
        """
        self.type = seg_type
        self.color = color
        self.points = points
        self.dwell_ms = dwell_ms
        self.id = id(self)  # 唯一ID
    
    def get_color_name(self) -> str:
        """获取颜色名称"""
        return self.COLOR_NAMES[self.color]
    
    def get_display_color_bgr(self) -> Tuple[int, int, int]:
        """获取OpenCV显示颜色(BGR)"""
        return self.COLOR_BGR[self.color]
    
    def get_display_color_rgb(self) -> Tuple[int, int, int]:
        """获取Qt显示颜色(RGB)"""
        return self.COLOR_RGB[self.color]
    
    def get_coord_str(self) -> str:
        """获取坐标字符串用于显示"""
        if self.type == 'dot':
            x, y = self.points[0]
            return f"({x:.1f}, {y:.1f})"
        else:  # line
            return f"{len(self.points)}点"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典保存"""
        # 优化：坐标保留2位小数，避免污染配置文件的git diff
        rounded_points = []
        for p in self.points:
            rounded_points.append([round(float(p[0]), 2), round(float(p[1]), 2)])
            
        return {
            'type': self.type,
            'color': self.color,
            'points': rounded_points,
            'dwell_ms': self.dwell_ms
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PathSegment':
        """从字典加载"""
        return PathSegment(
            seg_type=data['type'],
            color=data['color'],
            points=data['points'],
            dwell_ms=data['dwell_ms']
        )
    
    def __repr__(self):
        return f"PathSegment({self.type}, {self.get_color_name()}, {len(self.points)}pts, {self.dwell_ms}ms)"


class PathSegmentList:
    """路径段列表管理器"""
    
    def __init__(self):
        self.segments: List[PathSegment] = []
    
    def add(self, segment: PathSegment):
        """添加路径段"""
        self.segments.append(segment)
    
    def remove(self, index: int):
        """删除指定索引的路径段"""
        if 0 <= index < len(self.segments):
            self.segments.pop(index)
    
    def move_up(self, index: int):
        """上移"""
        if 0 < index < len(self.segments):
            self.segments[index], self.segments[index-1] = \
                self.segments[index-1], self.segments[index]
    
    def move_down(self, index: int):
        """下移"""
        if 0 <= index < len(self.segments) - 1:
            self.segments[index], self.segments[index+1] = \
                self.segments[index+1], self.segments[index]
    
    def clear(self):
        """清空"""
        self.segments.clear()
    
    def get_all(self) -> List[PathSegment]:
        """获取所有路径段"""
        return self.segments
    
    def count(self) -> int:
        """获取数量"""
        return len(self.segments)
    
    def to_dict_list(self) -> List[Dict]:
        """转换为字典列表"""
        return [seg.to_dict() for seg in self.segments]
    
    @staticmethod
    def from_dict_list(data: List[Dict]) -> 'PathSegmentList':
        """从字典列表加载"""
        plist = PathSegmentList()
        for item in data:
            plist.add(PathSegment.from_dict(item))
        return plist
    
    def get_legacy_format(self) -> Tuple[List, List]:
        """
        转换为旧版格式用于兼容
        返回: (polylines, dots)
        """
        polylines = []
        dots = []
        
        for seg in self.segments:
            if seg.type == 'dot':
                dots.extend(seg.points)
            else:  # line
                polylines.append(np.array(seg.points, dtype=np.float32))
        
        return polylines, dots
