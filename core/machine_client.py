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
        # query timestamp for motion check
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
                return True

            # --- 方案 B: 尝试读取 gcode_move (备用) ---
            gcode_move = all_status.get("gcode_move")
            if gcode_move:
                pos = gcode_move.get("gcode_position", [0.0, 0.0, 0.0, 0.0])
                self._x, self._y, self._z = map(float, pos[:3])
                return True

            # --- 两个都失败 ---
            self._log("[MachineClient] 读取状态失败：toolhead 和 gcode_move 均为空。status =", all_status)
            return False

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
            r = self.session.get(url, params=params, headers=self._headers(), timeout=2)
            data = r.json()
            status = data.get("result", {}).get("status", {}).get("probe", {})
            # last_query: 1 (triggered) / 0 (open)
        except:
            pass
        return 'unknown'

    def send_and_get_probe_status(self) -> str:
        """
        发送 QUERY_PROBE 并返回状态。
        """
        if not self.base_url: return 'unknown'
        self._post_gcode("QUERY_PROBE")
        
        url = f"{self.base_url}/printer/objects/query"
        params = {"probe": ""}
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=1)
            data = r.json()
            val = data.get("result", {}).get("status", {}).get("probe", {}).get("last_query")
            if val is None: return 'unknown'
            return 'triggered' if val else 'open'
        except:
            return 'unknown'

    def get_output_pin_status(self, pin_name: str) -> bool:
        """
        查询 output_pin 的状态 (value)。
        返回 True (On/1) 或 False (Off/0)。
        """
        if not self.base_url or not self.connected:
            return False
        
        # 名字中有空格需分别处理，但一般 output_pin xx 没有空格
        # Klipper object name: output_pin pin_name
        obj_name = f"output_pin {pin_name}"
        
        url = f"{self.base_url}/printer/objects/query"
        params = {obj_name: ""}
        
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=1)
            data = r.json()
            val = data.get("result", {}).get("status", {}).get(obj_name, {}).get("value")
            # value 通常是 0.0 或 1.0
            return bool(val and val > 0.001)
        except:
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
        相对运动（mm）
        """
        if not self.connected:
            print("[MachineClient] 未连接，忽略 move_rel。")
            return

        f_xy = float(feedrate_xy)
        f_z = float(feedrate_z)

        cmd_list = ["G91"]
        if dx != 0 or dy != 0:
            cmd_list.append(f"G0 X{dx:.3f} Y{dy:.3f} F{f_xy:.0f}")
        if dz != 0:
            cmd_list.append(f"G0 Z{dz:.3f} F{f_z:.0f}")
        cmd_list.append("G90")

        script = "\n".join(cmd_list)
        ok = self._post_gcode(script)
        if ok:
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

    def wait_until_idle(self, timeout: float = 30.0):
        """
        阻塞等待机器空闲。
        策略：发送 M400，然后监控坐标变化。
        如果坐标在约 0.3 秒内（3次检查）变化极小，则认为已静止。
        """
        if not self.connected:
            return

        # 1. 发送 M400 确保指令入队
        self._post_gcode("M400")
        
        # 2. 坐标稳定性检查
        start_t = time.time()
        last_log_t = start_t
        stable_count = 0
        last_pos = (self._x, self._y, self._z)
        
        while time.time() - start_t < timeout:
            # 同步最新坐标
            if not self._sync_toolhead_status():
                time.sleep(0.1)
                continue
                
            curr_pos = (self._x, self._y, self._z)
            
            # 计算位移距离
            dist = sum(abs(c - l) for c, l in zip(curr_pos, last_pos))
            
            if dist < 0.05: # 放宽阈值到 0.05mm (抗抖动)
                stable_count += 1
            else:
                stable_count = 0 # 重置
                last_pos = curr_pos
            
            # 连续 3 次 (0.3s) 稳定则退出
            if stable_count >= 3:
                return
            
            # 每 2 秒打印一次状态，防止用户以为死机
            if time.time() - last_log_t > 2.0:
                self._log(f"[MachineClient] 等待停稳... 偏差: {dist:.4f}mm")
                last_log_t = time.time()
            
            time.sleep(0.1)
        
        self._log(f"[MachineClient] wait_until_idle 超时 ({timeout}s)")


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
    # 运动控制 (新增 move_to)
    # ------------------------------------------------------------------ #
    def move_to(self, x: float = None, y: float = None, z: float = None, speed: float = 3000):
        """
        绝对移动
        """
        parts = []
        if x is not None: parts.append(f"X{x:.3f}")
        if y is not None: parts.append(f"Y{y:.3f}")
        if z is not None: parts.append(f"Z{z:.3f}")
        
        if not parts:
            return True

        # G90: 绝对定位
        cmd = "G1 " + " ".join(parts) + f" F{int(speed)}"
        script = f"G90\n{cmd}"
        return self._post_gcode(script)

    # ------------------------------------------------------------------ #
    # 阀门控制 (新增)
    # ------------------------------------------------------------------ #
    def test_valve_macro(self, pin_name: str, duration: float = 0.5):
        """
        调用 Klipper 宏 TEST_VALVE PIN=... T=...
        """
        if not self.connected: return
        cmd = f"TEST_VALVE PIN={pin_name} T={duration:.2f}"
        self._post_gcode(cmd)
        print(f"[MachineClient] {cmd}")
    
    def set_pin(self, pin_name: str, value: float):
        """
        直接设置 output_pin 的值
        value: 0.0 (关) 或 1.0 (开)
        """
        if not self.connected: return
        cmd = f"SET_PIN PIN={pin_name} VALUE={value:.1f}"
        self._post_gcode(cmd)
        print(f"[MachineClient] {cmd}")

    def all_valves_off(self):
        """调用 HAL 宏 ALL_VALVES_OFF"""
        if not self.connected: return
        self._post_gcode("ALL_VALVES_OFF")
        print("[MachineClient] ALL_VALVES_OFF")

    # Keep compatibility if needed, or deprecate old valve_on/off
    # But user specifically asked for "Test Valve" dialog capability.


    # ------------------------------------------------------------------ #
    # 温度控制 (新增)
    # ------------------------------------------------------------------ #
    def get_temp_status(self):
        """
        获取温度信息
        返回 dict: {
            'syringe': {'current': 0.0, 'target': 0.0, 'power': 0.0},
            'mcu': 0.0,   # Fly-Super8
            'host': 0.0   # FLY-pi
        }
        """
        if not self.base_url or not self.connected:
            return None

        # Klipper 对象名 (根据用户截图和配置推断)
        # 注意：如果有空格，requests 会自动 URL 编码
        params = {
            "heater_generic syringe_heater": "",
            "temperature_sensor Fly-Super8": "",
            "temperature_sensor FLY-pi": ""
        }
        
        url = f"{self.base_url}/printer/objects/query"
        try:
            r = self.session.get(url, params=params, headers=self._headers(), timeout=2)
            data = r.json()
            status = data.get("result", {}).get("status", {})
            
            res = {}
            
            # 1. Syringe Heater
            sh = status.get("heater_generic syringe_heater", {})
            # 如果没读到数据，默认为 0
            if sh:
                res['syringe'] = {
                    'current': sh.get('temperature', 0.0),
                    'target': sh.get('target', 0.0),
                    'power': sh.get('power', 0.0)
                }
            else:
                 res['syringe'] = {'current': 0.0, 'target': 0.0, 'power': 0.0}

            # 2. MCU Temp (Fly-Super8)
            # 注意：如果名字不对，这里 get 出来是 None，再 .get 会报错，所以要小心
            sensor_mcu = status.get("temperature_sensor Fly-Super8", {})
            res['mcu'] = sensor_mcu.get("temperature", 0.0) if sensor_mcu else 0.0
            
            # 3. Host Temp (FLY-pi)
            # 常见名称: 'temperature_sensor FLY-pi', 'temperature_sensor raspberry_pi', 'temperature_host'
            # 如果找不到，打印一下所有 status keys 方便调试
            host_keys = [k for k in status.keys() if 'temperature_sensor' in k or 'temperature_host' in k]
            
            sensor_host = status.get("temperature_sensor FLY-pi", {})
            if not sensor_host:
                 # Backup try
                 if "temperature_host" in status:
                     sensor_host = status["temperature_host"]
                 elif "temperature_sensor raspberry_pi" in status:
                     sensor_host = status["temperature_sensor raspberry_pi"]
            
            # Debug Log (Only once or occasionally)
            # print(f">>> DEBUG CHECK HOST KEYS: {host_keys}", flush=True)
            # raw_val = status.get("temperature_sensor FLY-pi")
            
            res['host'] = 0.0
            
            # Helper logic to extract temperature from diverse return types
            def extract_temp(val):
                if isinstance(val, dict):
                    return val.get("temperature", 0.0)
                elif hasattr(val, 'temperature'):
                    return val.temperature
                return 0.0

            sensor_host_val = extract_temp(sensor_host)
            
            if sensor_host_val > 0:
                res['host'] = sensor_host_val
            else:
                # Fallback: Try Moonraker /machine/proc_stats
                # This is more reliable for Host CPU temp
                try:
                    url_stats = f"{self.base_url}/machine/proc_stats"
                    r_stats = self.session.get(url_stats, headers=self._headers(), timeout=1)
                    if r_stats.ok:
                        s_data = r_stats.json().get("result", {})
                        # usually in 'cpu_temp'
                        res['host'] = s_data.get("cpu_temp", 0.0)
                        # print(f"[DEBUG] Fallback to proc_stats: {res['host']}")
                except:
                    pass
            
            return res
            
        except Exception as e:
            # 避免刷屏，只打印一次或调试打印
            # self._log(f"[MachineClient] get_temp_status error: {e}") 
            return None

    def set_heater_temp(self, temp: float):
        """设置针筒加热温度"""
        if not self.connected: return
        # SET_HEATER_TEMPERATURE HEATER=syringe_heater TARGET=xxx
        self._post_gcode(f"SET_HEATER_TEMPERATURE HEATER=syringe_heater TARGET={temp:.1f}")

