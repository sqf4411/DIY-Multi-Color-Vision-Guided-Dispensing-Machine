#!/usr/bin/env python3
"""
Raspberry Pi Xbox Controller Script for Klipper/Moonraker
---------------------------------------------------------
此脚本设计用于运行在连接了 Xbox 手柄的树莓派上 (Upper Computer)。
它通过 HTTP 请求直接向本地的 Moonraker (Klipper) 发送 G-code 指令。

功能升级:
1. **线性速度控制**: 摇杆推得越远，移动速度越快 (Velocity Mode)。
2. **快/慢速切换**: 使用 Start 键或 X 键切换 "精细慢速" 和 "快速移动" 模式。
3. **电磁阀控制**:
   - 按住 **右肩键 (RB)** 或 **右扳机 (RT)** 打开阀门 (点胶)。
   - 松开即关闭。
   - 默认控制 PIN: 'VALVE_A_CTRL' (请根据实际情况修改下方配置)
4. **安全特性**: 掉线自动重连，急停 (B键)。

安装依赖 (在树莓派上):
  sudo apt-get install python3-pip
  pip3 install evdev requests

使用方法:
  python3 rpi_xbox_control.py
"""

import os
import sys
import time
import requests
import threading
from evdev import InputDevice, categorize, ecodes, list_devices

# ================= 用户配置区域 =================

# Moonraker 地址
MOONRAKER_URL = "http://localhost:7125"

# 移动控制参数
LOOP_INTERVAL = 0.1      # 控制循环周期 (秒)，越小越顺滑，但负载越高 (推荐 0.05 - 0.1)
DEADZONE = 0.15          # 摇杆死区 (0.0 - 1.0)

# "快速模式" 参数 (用于长距离移动)
MAX_SPEED_FAST_XY = 6000 # mm/min
MAX_SPEED_FAST_Z  = 1000 # mm/min

# "慢速/精细模式" 参数 (用于对刀、微调)
MAX_SPEED_SLOW_XY = 500  # mm/min
MAX_SPEED_SLOW_Z  = 200  # mm/min

# 默认模式 (True=快, False=慢)
DEFAULT_FAST_MODE = False 

# 电磁阀配置
# 填入您 Klipper 配置文件 [output_pin] 或 [gcode_macro] 中定义的 PIN 名称
# 常见: 'VALVE_A_CTRL', 'A1_VALVE', 'PA1' 等
VALVE_PIN_NAME = "VALVE_A_CTRL" 

# ===============================================

# 键值映射
AXIS_LX = ecodes.ABS_X
AXIS_LY = ecodes.ABS_Y
AXIS_RX = ecodes.ABS_RX
AXIS_RY = ecodes.ABS_RY

# Xbox 手柄常见按键 (根据 evdev dump 可能有所不同)
BTN_A = ecodes.BTN_SOUTH
BTN_B = ecodes.BTN_EAST
BTN_X = ecodes.BTN_NORTH
BTN_Y = ecodes.BTN_WEST
BTN_START = ecodes.BTN_START
BTN_SELECT = ecodes.BTN_SELECT
BTN_TL = ecodes.BTN_TL # Left Shoulder (LB)
BTN_TR = ecodes.BTN_TR # Right Shoulder (RB)

class KlipperController:
    def __init__(self, host_url):
        self.url = host_url
        self.session = requests.Session()
        self.connected = False
        
    def check_connection(self):
        try:
            r = self.session.get(f"{self.url}/server/info", timeout=2)
            if r.status_code == 200:
                if not self.connected:
                    print(f"[INFO] 已连接到 Moonraker: {self.url}")
                self.connected = True
                return True
        except Exception as e:
            if self.connected: 
                 print(f"[ERROR] Moonraker 连接断开: {e}")
            self.connected = False
        return False

    def send_gcode(self, gcode):
        if not self.connected:
            return
        try:
            payload = {"script": gcode}
            self.session.post(f"{self.url}/printer/gcode/script", json=payload, timeout=0.2)
        except Exception:
            pass # 忽略高频发送时的偶发错误

    def set_pin(self, pin_name, value):
        """控制输出引脚 (电磁阀)"""
        # value: 1.0 (On) or 0.0 (Off)
        cmd = f"SET_PIN PIN={pin_name} VALUE={value}"
        print(f"[VALVE] {cmd}")
        self.send_gcode(cmd)

    def move_velocity(self, v_x, v_y, v_z, dt):
        """
        基于速度的移动 (Velocity Control)
        原理: 计算 dt 时间内应该移动的距离 d = v * dt
        """
        if v_x == 0 and v_y == 0 and v_z == 0:
            return

        # 计算这一小段时间内需要移动的距离 (mm)
        # 速度单位是 mm/min，需要除以 60 换算成 mm/sec
        d_x = (v_x / 60.0) * dt
        d_y = (v_y / 60.0) * dt
        d_z = (v_z / 60.0) * dt

        # 计算合成移动速度 F (Klipper 需要知道这条指令的移动速度)
        # G1 X... Y... F...
        # 这里直接取轴向最大分量作为合成速度指令即可，或者取矢量合
        # 简单起见，取当前控制指令要求的最大速度
        feedrate = max(abs(v_x), abs(v_y), abs(v_z))
        
        # 忽略极小移动
        if abs(d_x) < 0.001 and abs(d_y) < 0.001 and abs(d_z) < 0.001:
            return

        # 发送 G-code
        # G91 相对模式
        cmd = f"G91\nG1 X{d_x:.4f} Y{d_y:.4f} Z{d_z:.4f} F{feedrate:.0f}\nG90"
        self.send_gcode(cmd)

class XboxJogger:
    def __init__(self, controller_path=None):
        self.dev = None
        self.klipper = KlipperController(MOONRAKER_URL)
        
        # 输入状态
        self.lx = 0.0
        self.ly = 0.0
        self.ry = 0.0
        
        # 模式状态
        self.is_fast_mode = DEFAULT_FAST_MODE
        self.valve_active = False
        
        self.running = True
        self.connect_gamepad(controller_path)

    def connect_gamepad(self, path=None):
        devices = [InputDevice(path) for path in list_devices()]
        gamepad = None
        if not devices:
            print("[ERROR] 未找到任何输入设备。")
            return

        for device in devices:
            if "Xbox" in device.name or "Microsoft" in device.name or "Gamepad" in device.name:
                gamepad = device
                break
        
        if gamepad:
            self.dev = gamepad
            print(f"[SUCCESS] 已连接: {self.dev.name}")
        elif devices:
            self.dev = devices[0]
            print(f"[WARN] 默认使用: {self.dev.name}")

    def normalize(self, value, min_val, max_val):
        range_val = max_val - min_val
        norm = (value - min_val) / range_val 
        return (norm * 2) - 1.0

    def apply_deadzone(self, val):
        if abs(val) < DEADZONE: return 0.0
        return val

    def loop(self):
        if not self.dev: return

        # 后台线程：定时发送运动指令
        jog_thread = threading.Thread(target=self.jog_task, daemon=True)
        jog_thread.start()

        print("================ 操作说明 ================")
        print(" [左摇杆]     XY 平面移动 (线性速度)")
        print(" [右摇杆]     Z 轴移动")
        print(" [Start/X键]  切换 快/慢速 模式 (默认: 慢速)")
        print(" [RB/RT键]    按住出胶 (控制阀门)")
        print(" [B键]        急停 (M112)")
        print("==========================================")
        
        for event in self.dev.read_loop():
            # 摇杆处理
            if event.type == ecodes.EV_ABS:
                # 获取范围
                absinfo = self.dev.absinfo(event.code)
                min_v, max_v = absinfo.min, absinfo.max

                if event.code == AXIS_LX:
                    self.lx = self.apply_deadzone(self.normalize(event.value, min_v, max_v))
                elif event.code == AXIS_LY:
                    self.ly = -self.apply_deadzone(self.normalize(event.value, min_v, max_v))
                elif event.code == AXIS_RY: 
                    self.ry = -self.apply_deadzone(self.normalize(event.value, min_v, max_v))
                
                # 某些手柄 RT/LT 也是 ABS (ABS_Z / ABS_RZ)，如果需要线性扳机控制阀门可以在这里加
                # 这里暂只用按钮模式触发阀门 (EV_KEY)

            # 按键处理
            elif event.type == ecodes.EV_KEY:
                # 切换速度模式 (Start 或 X)
                if (event.code == BTN_START or event.code == BTN_X) and event.value == 1:
                    self.is_fast_mode = not self.is_fast_mode
                    mode_str = "FAST (快速)" if self.is_fast_mode else "SLOW (精细)"
                    print(f"[MODE] 切换模式 -> {mode_str}")

                # 阀门控制 (RB = Right Bumper) - 按下开，松开合
                elif event.code == BTN_TR:
                    if event.value == 1: # Defined Pressed
                        self.klipper.set_pin(VALVE_PIN_NAME, 1)
                        print("[VALVE] ON")
                    else:                # Released
                        self.klipper.set_pin(VALVE_PIN_NAME, 0)
                        print("[VALVE] OFF")

                # 急停
                elif event.code == BTN_B and event.value == 1:
                    print("[STOP] !!! 急停触发 !!!")
                    self.klipper.send_gcode("M112")

    def jog_task(self):
        """以固定频率发送速度移动指令"""
        self.klipper.check_connection()
        
        while self.running:
            if not self.klipper.connected:
                if self.klipper.check_connection(): time.sleep(1)
                else: time.sleep(2)
                continue

            # 确定当前最大速度
            max_xy = MAX_SPEED_FAST_XY if self.is_fast_mode else MAX_SPEED_SLOW_XY
            max_z  = MAX_SPEED_FAST_Z  if self.is_fast_mode else MAX_SPEED_SLOW_Z

            # 计算各个轴的目标速度 (Speed = Max * Ratio)
            # lx, ly, ry 范围是 -1.0 ~ 1.0
            vx = self.lx * max_xy
            vy = self.ly * max_xy
            vz = self.ry * max_z

            # 如果有速度请求，则发送移动
            if abs(vx) > 0 or abs(vy) > 0 or abs(vz) > 0:
                self.klipper.move_velocity(vx, vy, vz, LOOP_INTERVAL)
            
            time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    try:
        XboxJogger().loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
