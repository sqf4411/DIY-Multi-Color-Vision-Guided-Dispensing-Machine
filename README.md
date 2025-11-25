# DIY 多色视觉引导点胶机 – 相机 & Klipper 一体化 UI

本项目是一个围绕 **工业相机 + Klipper（Moonraker）** 的桌面端工具，用于：
- 实时预览工业相机画面，调节曝光 / 增益 / Gamma
- 自动识别硅胶模具、提取“大框 + 4 个腔体位置”
- 通过「黄金示教」「路径示教」生成点胶模板
- 通过自定义 `MachineClient` 控制 Klipper（或本地模拟）实现 XYZ 点动、回零
- 后续扩展：像素坐标 → 机械坐标标定、虚拟跑、空跑、真实点胶等

适用场景：  
DIY 6 轴点胶机 / 工匠键帽多色点胶 / 硅胶负模点胶生产线等。

---

## 目录结构

项目主要文件：

- `main_camera_ui.py`  
  主程序入口，PyQt5 UI：
  - 左侧：工业相机实时画面
  - 右侧：相机参数调节、Klipper 连接、XYZ 点动控制、日志窗口
  - 按钮：启动 / 停止相机、保存图片、识别模具、黄金示教、路径示教、标定坐标系、虚拟跑 / 空跑 / 真跑、测试出胶等

- `machine_client.py`  
  运动控制抽象层：
  - 当前版本：**本地模拟坐标版**，不真正发网络请求，方便开发 UI 和流程
  - 以后可以替换为：调用 Klipper / Moonraker HTTP API、串口、TCP socket 等

- `mold_detector.py`  
  模具识别与黄金模板缓存：
  - 负责从图像中检测：
    - 大模具外框（1 个）
    - 4 个小腔体的位置
  - 管理黄金示教产生的模板（`slot_templates.npy`、`warp_size.npy` 等）
  - 提供统一接口给 UI 进行「识别模具」显示

- `golden_teach.py`  
  「黄金示教」工具脚本：
  - 打开一张基准图（例如 `mold2.png`）
  - 用户在图像上手动画 4 个黄色框（4 个产品）
  - 保存到：
    - `slot_templates.npy`：4×4×2 的黄框坐标（4 个产品 × 每个框四个角）
    - `warp_size.npy`：透视矫正后的宽高

- `path_teach.py`  
  「点胶路径示教」工具脚本：
  - 基于黄金示教中的某个小格（cell），手动画点胶折线 / 点集
  - 保存路径到：
    - `cell_0_polylines.npy` / `cell_0_dots.npy` 等
  - 后续由 UI 调用，用于虚拟跑 / 空跑 / 真实点胶

- `golden_corners_adjust.py`（建议名，单独脚本）  
  「心理更踏实版」四角精调工具：
  - 读取基准图 `mold2.png`
  - 通过 `detect_mold_and_cells()` 自动获取大框四角
  - 提供放大视图 + 方向键 / WASD 精调像素
  - 保存最终 4 个角点到 `golden_corners.npy`，作为更精准的基准角点

- 典型数据文件（运行过程中产生）：
  - `mold2.png`：黄金示教用的基准图
  - `slot_templates.npy`：4 个产品的黄框模板
  - `warp_size.npy`：透视后的图像尺寸
  - `golden_corners.npy`：精调后的大框四角坐标（可选）
  - `cell_0_polylines.npy` / `cell_0_dots.npy`：某个腔体的点胶路径
  - `camera_params.json`：曝光、增益、Gamma 等相机参数

---

## 环境依赖

### 1. Python 环境

- 建议使用 **Python 3.8+，Windows 10/11**
- 建议在虚拟环境中安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
