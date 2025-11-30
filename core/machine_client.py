# machine_client.py
from typing import Tuple, Optional
import requests
import time


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

    def _post_gcode(self, script: str, timeout: int = 20):
        """
        向 /printer/gcode/script 发送一段 G-code 脚本
        """
        if not self.connected or not self.base_url:
            print("[MachineClient] 未连接，忽略发送 G-code。")
            return False

        url = f"{self.base_url}/printer/gcode/script"
        payload = {"script": script}
        try:
            # 使用传入的 timeout
            r = self.session.post(url, json=payload, headers=self._headers(), timeout=timeout)
            r.raise_for_status()
            return True
        except Exception as e:
            # 这里最好把 print 改成 self._log 方便在 UI 看，不过 print 也能在控制台看
            print(f"[MachineClient] 发送 G-code 失败: {e}  script={script!r}")
            return False

    def send_gcode(self, script: str):
        """
        发送 G-code (公开接口)
        """
        return self._post_gcode(script)

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
            # --- 方案 A: 尝试读取 toolhead ---
            if toolhead:
                pos = toolhead.get("position", [0.0, 0.0, 0.0, 0.0])
                self._x, self._y, self._z = map(float, pos[:3])
                homed_axes = toolhead.get("homed_axes", "")
                self.is_homed = bool(homed_axes)
                # 不再打印“同步成功”日志，避免定时器轮询时刷屏
                return True

            # --- 方案 B: 尝试读取 gcode_move ---
            gcode_move = all_status.get("gcode_move")
            if gcode_move:
                pos = gcode_move.get("gcode_position", [0.0, 0.0, 0.0, 0.0])
                self._x, self._y, self._z = map(float, pos[:3])
                # 同理这里也去掉频繁日志
                return True


            # --- 两个都失败 ---
            self._log("[MachineClient] 读取状态失败：toolhead 和 gcode_move 均为空。status =", all_status)
            return False

        except Exception as e:
            self._log("[MachineClient] 读取状态请求异常:", e)
        except Exception as e:
            self._log("[MachineClient] 读取状态请求异常:", e)
            return False

    def query_probe_status(self) -> str:
        """
        查询探针状态。
        优先查询 [gcode_macro TOOL_STATE] 的 state 变量 (用户自定义)。
        如果失败，再尝试查询 [probe] 对象。
        返回: 'open', 'triggered', or 'unknown'
        """
        if not self.base_url or not self.connected:
            return 'unknown'

        # --- 方案 A: 查询用户自定义宏 TOOL_STATE ---
        # URL: /printer/objects/query?gcode_macro TOOL_STATE
        url = f"{self.base_url}/printer/objects/query"
        # 注意空格要编码，requests 会自动处理 params
        params = {"gcode_macro TOOL_STATE": ""}
        
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=1)
            data = r.json()
            # 路径: result -> status -> gcode_macro TOOL_STATE -> state
            macro_status = data.get("result", {}).get("status", {}).get("gcode_macro TOOL_STATE", {})
            
            if macro_status:
                state = macro_status.get("state")
                # 用户配置: 1=Triggered, 0=Open
                if state is not None:
                    try:
                        val = int(state)
                        return 'triggered' if val == 1 else 'open'
                    except:
                        pass
        except Exception:
            pass

        # --- 方案 B: 标准 Probe 对象 (备用) ---
        # 1. 发送 QUERY_PROBE (可能会刷屏，仅作为备用)
        # self._post_gcode("QUERY_PROBE") 
        
        # 2. 读取 probe 对象
        params = {"probe": ""}
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=1)
            data = r.json()
            probe_data = data.get("result", {}).get("status", {}).get("probe", {})
            if probe_data:
                last_query = probe_data.get("last_query")
                if last_query is not None:
                    return 'triggered' if last_query else 'open'
        except Exception:
            pass
            
        return 'unknown'

    def get_probe_status(self) -> str:
        """
        查询探针状态。
        返回: 'open', 'triggered', or 'unknown'
        """
        if not self.base_url or not self.connected:
            return 'unknown'

        # Klipper 的 probe 对象通常叫 "probe"
        # 我们查询 /printer/objects/query?probe
        url = f"{self.base_url}/printer/objects/query"
        params = {"probe": ""}
        
        try:
            # 注意：如果不先发送 QUERY_PROBE，这个值可能不会更新。
            # 但频繁发送 QUERY_PROBE 会导致 Klipper 繁忙。
            # 我们先尝试直接读。如果用户发现不更新，可能需要手动发指令。
            # 或者，我们可以发送 QUERY_PROBE 命令，然后解析返回结果。
            # 这里为了实时性，我们尝试直接读对象。
            r = self.session.get(url, params=params, headers=self._headers(), timeout=2)
            data = r.json()
            status = data.get("result", {}).get("status", {}).get("probe", {})
            
            # last_query: True (Triggered) / False (Open) ? 
            # 或者 last_z_result?
            # Klipper 文档：probe.last_query: The result of the last QUERY_PROBE command.
            # 所以必须发 QUERY_PROBE 才能更新这个值。
            # 这对实时显示灯不太友好。
            
            # 替代方案：查询 endstops? 如果探针接在 endstop 上。
            # /printer/objects/query?query_endstops
            # 这也需要 QUERY_ENDSTOPS 命令。
            
            # 既然用户要求“针触碰到了对刀仪并触发，灯灭”，说明他可能在手动测试。
            # 我们可以每秒发一次 QUERY_PROBE 吗？
            # 让我们尝试发送 QUERY_PROBE 并解析 response。
            pass
        except:
            pass
            
        # 方案 C: 发送 QUERY_PROBE 并解析 console output? 太麻烦。
        # 方案 D: 假设用户配置了 [probe] 且 Klipper 会自动更新某些状态？
        # 通常不会。
        
        # 让我们先实现一个简单的：发送 QUERY_PROBE，然后读 output。
        # 或者，只在“手动对刀模式”下开启高频轮询。
        
        return 'unknown'

    def send_and_get_probe_status(self) -> str:
        """
        发送 QUERY_PROBE 并返回状态。
        """
        if not self.base_url: return 'unknown'
        
        # 这是一个 hack，因为我们没法直接拿到命令返回值（_post_gcode 不返回内容）
        # 但我们可以通过 objects/query?probe 来读 last_query
        # 前提是我们先发了 QUERY_PROBE
        
        self._post_gcode("QUERY_PROBE")
        
        # 稍等一下？或者直接读？
        # Klipper 处理命令是异步的。
        # 也许我们不需要实时显示灯？
        # 用户说：“还要显示这个对刀灯在ui界面...灯灭，说明触发了”
        # 这意味着必须实时。
        
        # 如果探针是接在 endstop 上的（例如 z_min），我们可以查询 query_endstops?
        # 还是需要发命令。
        
        # 让我们尝试读 probe 对象，假设它能反映状态。
        url = f"{self.base_url}/printer/objects/query"
        params = {"probe": ""}
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=1)
            data = r.json()
            val = data.get("result", {}).get("status", {}).get("probe", {}).get("last_query")
            # last_query: 1 (triggered) / 0 (open)
            if val is None: return 'unknown'
            return 'triggered' if val else 'open'
        except:
            return 'unknown'

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


    def move_z_absolute(self, z: float, feedrate: float = 600.0):
        """
        Z轴绝对运动
        """
        if not self.connected:
            return

        script = f"G90\nG0 Z{z:.3f} F{feedrate:.0f}"
        ok = self._post_gcode(script)
        if ok:
            self._z = z
            
    def move_xy_absolute(self, x: float, y: float, feedrate: float = 6000.0):
        """
        XY绝对运动
        """
        if not self.connected:
            return

        script = f"G90\nG0 X{x:.3f} Y{y:.3f} F{feedrate:.0f}"
        ok = self._post_gcode(script)
        if ok:
            self._x = x
            self._y = y

    def wait_for_move(self):
        """
        等待运动结束 (M400)
        """
        if not self.connected:
            return
        self._post_gcode("M400")
    # 回零
    # ------------------------------------------------------------------ #
    def wait_until_idle(self):
        """
        等待机器空闲 (M400 + G4 P100)
        """
        if not self.connected:
            return
        # M400 等待移动结束
        self._post_gcode("M400")
        # G4 P100 强制停留 100ms，确保状态同步
        self._post_gcode("G4 P100")

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

    # ------------------------------------------------------------------ #
    # 阀门控制 (新增)
    # ------------------------------------------------------------------ #
    def valve_on(self):
        """打开点胶阀 (发送 VALVE_ON 宏)"""
        if not self.connected: return
        self._post_gcode("VALVE_ON")
        print("[MachineClient] VALVE_ON")

    def valve_off(self):
        """关闭点胶阀 (发送 VALVE_OFF 宏)"""
        if not self.connected: return
        self._post_gcode("VALVE_OFF")
        print("[MachineClient] VALVE_OFF")
