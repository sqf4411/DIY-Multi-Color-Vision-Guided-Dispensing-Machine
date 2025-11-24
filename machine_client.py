# machine_client.py
from typing import Tuple


class MachineClient:
    """
    运动控制抽象层：
    - 现在只是本地模拟坐标
    - 将来可以改成：HTTP 调 Klipper、ssh 发 Gcode、串口、socket 等
    """

    def __init__(self):
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self.connected = False

    # ===== 连接相关 =====
    def connect(self):
        """将来在这里做真正的连接，现在只标记为已连接。"""
        self.connected = True

    def disconnect(self):
        self.connected = False

    # ===== 位置相关 =====
    def get_position(self) -> Tuple[float, float, float]:
        """返回当前坐标（单位 mm）"""
        return self._x, self._y, self._z

    def move_rel(self, dx: float, dy: float, dz: float):
        """
        相对运动（mm）

        TODO（将来）：
        - 发 G-code，比如 G91 / G0 X.. Y.. Z.. / G90
        - 或者通过 HTTP / WebSocket / 串口下发
        """
        if not self.connected:
            return

        self._x += dx
        self._y += dy
        self._z += dz

    def home_all(self):
        """全部回零"""
        if not self.connected:
            return
        self._x = self._y = self._z = 0.0

    # ==== 新增：单轴回零 ====
    def home_x(self):
        if not self.connected:
            return
        self._x = 0.0

    def home_y(self):
        if not self.connected:
            return
        self._y = 0.0

    def home_z(self):
        if not self.connected:
            return
        self._z = 0.0