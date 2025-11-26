# machine_client.py
from typing import Tuple, Optional
import requests


class MachineClient:
    """
    运动控制抽象层：通过 Moonraker HTTP 调 Klipper

    - configure(base_url)        配置 Moonraker 地址，例如 http://192.168.60.26:7125
    - connect()                  连接 /server/info，并同步一次 homed 状态 & 坐标
    - move_rel(dx,dy,dz)         相对运动（mm），自动用 G91/G90
    - get_position()             从 Klipper 读当前 X,Y,Z
    - home_all / home_x / ...    执行 G28 / G28 X / ...
    """

    def __init__(self):
        # 当前坐标缓存（mm）
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0

        # 连接相关
        self.connected: bool = False
        self.base_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.session = requests.Session()

        # 是否已经回零（由 Klipper 状态决定）
        self.is_homed: bool = False

        # 👉 新增：外部可以塞一个 logger 进来（比如 UI 的 self.log）
        self.logger = None

    # 小工具：统一用这个打日志
    def _log(self, *args):
        msg = " ".join(str(a) for a in args)
        if self.logger:
            self.logger(msg)      # 打到 UI
        else:
            print(msg)            # 没有 logger 时退回 print

    # ------------------------------------------------------------------ #
    # 配置 & 连接
    # ------------------------------------------------------------------ #
    def configure(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._log(f"[MachineClient] configure: base_url={self.base_url}, api_key={self.api_key}")

    def connect(self) -> bool:
        if not self.base_url:
            self._log("[MachineClient] connect: base_url 未配置")
            return False

        try:
            r = self.session.get(f"{self.base_url}/server/info", timeout=3)
            r.raise_for_status()
        except Exception as e:
            self._log(f"[MachineClient] 连接 Moonraker 失败: {e}")
            self.connected = False
            return False

        self._log(f"[MachineClient] 已连接 Moonraker: {self.base_url}")

        # 同步一次 homed 状态和坐标
        ok = self._sync_toolhead_status()

        self.connected = True
        if ok:
            # 成功拿到 homed_axes 时才根据结果提示
            if self.is_homed:
                self._log("[MachineClient] 当前已回零，可直接运动。")
            else:
                self._log("[MachineClient] 当前未回零，需要先执行 G28。")
        else:
            # 读不到 toolhead，就不要乱说“未回零”，只告诉你状态未知
            self._log(
                "[MachineClient] 无法从 Klipper 读取回零状态（toolhead=None），"
                "但已成功连接，你可以直接尝试移动 XYZ。"
            )


        return True


    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
        # ... (接在 force_fake_home 函数后面)

    def emergency_stop(self):
        """
        🚨 发送 M112 急停指令
        Klipper 会立即关闭 MCU，停止所有动作。
        恢复需要手动在 Fluidd/Mainsail 点击 Firmware Restart。
        """
        if not self.connected:
            return

        # M112 是最高优先级指令
        print("🚨🚨🚨 SENDING EMERGENCY STOP (M112) 🚨🚨🚨")
        self._post_gcode("M112")
        self.is_homed = False  # 急停后状态肯定丢失




    def _headers(self):
        if self.api_key:
            return {"X-Api-Key": self.api_key}
        return {}

    def _post_gcode(self, script: str):
        """
        向 /printer/gcode/script 发送一段 G-code 脚本
        """
        if not self.connected or not self.base_url:
            print("[MachineClient] 未连接，忽略发送 G-code。")
            return False

        url = f"{self.base_url}/printer/gcode/script"
        payload = {"script": script}
        try:
            # 👇👇👇 修改这里：把 timeout=5 改成 timeout=20 👇👇👇
            r = self.session.post(url, json=payload, headers=self._headers(), timeout=120)
            r.raise_for_status()
            return True
        except Exception as e:
            # 这里最好把 print 改成 self._log 方便在 UI 看，不过 print 也能在控制台看
            print(f"[MachineClient] 发送 G-code 失败: {e}  script={script!r}")
            return False

    def _sync_toolhead_status(self) -> bool:
        """
        从 Klipper 读取状态：
        优先尝试读取 'toolhead'，如果失败则尝试读取 'gcode_move'。
        """
        if not self.base_url:
            return False

        url = f"{self.base_url}/printer/objects/query"
        # ✅ 正确写法：/printer/objects/query?toolhead&gcode_move
        params = {"toolhead": "", "gcode_move": ""}

        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=5)
            r.raise_for_status()
            data = r.json()

            result = data.get("result", {})
            all_status = result.get("status", {})

            # --- 方案 A: 尝试读取 toolhead ---
            toolhead = all_status.get("toolhead")
            if toolhead:
                pos = toolhead.get("position", [0.0, 0.0, 0.0, 0.0])
                self._x, self._y, self._z = map(float, pos[:3])
                homed_axes = toolhead.get("homed_axes", "")
                self.is_homed = bool(homed_axes)
                self._log(f"[MachineClient] 同步成功 (toolhead): ({self._x:.2f}, {self._y:.2f}, {self._z:.2f})")
                return True

            # --- 方案 B: 尝试读取 gcode_move ---
            gcode_move = all_status.get("gcode_move")
            if gcode_move:
                pos = gcode_move.get("gcode_position", [0.0, 0.0, 0.0, 0.0])
                self._x, self._y, self._z = map(float, pos[:3])
                self._log(f"[MachineClient] 同步成功 (gcode_move): ({self._x:.2f}, {self._y:.2f}, {self._z:.2f})")
                return True

            # --- 两个都失败 ---
            self._log("[MachineClient] 读取状态失败：toolhead 和 gcode_move 均为空。status =", all_status)
            return False

        except Exception as e:
            self._log("[MachineClient] 读取状态请求异常:", e)
            return False

    # ------------------------------------------------------------------ #
    # 坐标查询
    # ------------------------------------------------------------------ #

    def get_position(self) -> Tuple[float, float, float]:
        if self.base_url and self.connected:
            try:
                self._sync_toolhead_status()
            except Exception as e:
                self._log("[MachineClient] get_position: 同步 Klipper 坐标失败，使用缓存坐标。", e)
        return self._x, self._y, self._z

    # ------------------------------------------------------------------ #
    # 运动控制
    # ------------------------------------------------------------------ #
    def move_rel(self, dx: float, dy: float, dz: float,
                 feedrate_xy: float = 6000.0, feedrate_z: float = 600.0):
        """
        相对运动（mm）：
        - 使用 G91 / G90 切换到相对坐标再切回绝对
        - 如果 Klipper 报“未回零”，会直接返回错误，不再在这里拦截
        """
        if not self.connected:
            print("[MachineClient] 未连接，忽略 move_rel。")
            return

        # 如果机器还没回零，让 Klipper 自己报错即可（你那边会看到 400 错误）
        # 我们不再在这里做人为禁止，这样重启 UI 但 Klipper 没重启时可以直接移动。
        f_xy = float(feedrate_xy)
        f_z = float(feedrate_z)

        cmd_list = ["G91"]  # 相对模式
        # 分开 XY 和 Z 的速度
        if dx != 0 or dy != 0:
            cmd_list.append(f"G0 X{dx:.3f} Y{dy:.3f} F{f_xy:.0f}")
        if dz != 0:
            cmd_list.append(f"G0 Z{dz:.3f} F{f_z:.0f}")
        cmd_list.append("G90")  # 切回绝对

        script = "\n".join(cmd_list)
        ok = self._post_gcode(script)
        if ok:
            # 不再从 Klipper 读状态，直接在本地累加软件坐标
            self._x += dx
            self._y += dy
            self._z += dz


    # ------------------------------------------------------------------ #
    # 回零
    # ------------------------------------------------------------------ #
    def home_all(self):
        if not self.connected:
            return
        if self._post_gcode("G28"):
            self.is_homed = True
            self._sync_toolhead_status()
            print("[MachineClient] 已执行 G28 全部回零。")

    def home_x(self):
        if not self.connected:
            return
        if self._post_gcode("G28 X"):
            self.is_homed = True
            self._sync_toolhead_status()
            print("[MachineClient] 已执行 G28 X。")

    def home_y(self):
        if not self.connected:
            return
        if self._post_gcode("G28 Y"):
            self.is_homed = True
            self._sync_toolhead_status()
            print("[MachineClient] 已执行 G28 Y。")

    def home_z(self):
        if not self.connected:
            return
        if self._post_gcode("G28 Z"):
            self.is_homed = True
            self._sync_toolhead_status()
            print("[MachineClient] 已执行 G28 Z。")
