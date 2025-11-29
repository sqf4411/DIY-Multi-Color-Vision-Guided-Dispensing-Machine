# DIY 多色视觉引导点胶机 (DIY Multi-Color Vision Guided Dispensing Machine)

本项目使用 Python、OpenCV 和 PyQt5 实现了一个视觉引导的点胶系统。它通过 HTTP/WebSocket 接口与运动控制器 (Klipper) 通信，根据视觉反馈执行精确的点胶任务。

## 📂 1. Python 源代码文件

| 文件名 | 描述 |
| :--- | :--- |
| `main_camera_ui.py` | **主程序入口**。负责主 GUI 界面、相机显示、事件循环，并协调所有其他模块。 |
| `mold_detector.py` | **核心视觉逻辑**。处理模具识别、模板匹配以及路径计算（包含点和线）。 |
| `golden_teach.py` | **黄金示教**。允许用户定义标准模具模板（4个槽位），支持实时绘制和撤销功能。 |
| `path_teach.py` | **路径编辑**。允许用户在模板上绘制点胶路径（左键画线，右键画点）。 |
| `dry_run_dialog.py` | **G代码生成**。空跑/点胶设置对话框。根据路径和设置（安全高度、工作高度、延时、速度）生成 G 代码。 |
| `machine_client.py` | **机器控制**。通过 HTTP/WebSocket 处理与运动控制器 (Klipper/Moonraker) 的通信。 |
| `golden_corners_adjust.py` | **标定辅助**。用于在坐标标定过程中微调模具的 4 个角点。 |
| `run_production.py` | **生产脚本**。用于生产模式运行的脚本（无头模式或简化界面）。 |
| `testUi.py` | **UI 测试**。用于单独测试 UI 组件的简单脚本。 |

## 🛠️ 2. 重要函数说明

| 函数名 | 所在文件 | 描述 |
| :--- | :--- | :--- |
| `detect_mold_and_cells` | `mold_detector.py` | **视觉识别**。在当前帧中检测大模具框架和单个产品槽位。 |
| `get_calculated_paths` | `mold_detector.py` | **路径计算**。计算所有检测到的产品的最终点胶路径（像素坐标），应用偏移和透视校正。 |
| `run_golden_teach` | `golden_teach.py` | **示教**。启动黄金示教 UI 循环。 |
| `run_path_teach` | `path_teach.py` | **示教**。启动路径编辑 UI 循环。 |
| `generate_gcode` | `dry_run_dialog.py` | **G代码**。将像素路径转换为机器 G 代码，处理抬刀、速度和延时。 |
| `render_current_frame` | `main_camera_ui.py` | **渲染**。主渲染循环。绘制相机画面、覆盖层（框、路径、点）并处理 UI 状态显示。 |
| `on_dry_run` | `main_camera_ui.py` | **逻辑**。点击“空跑/点胶”按钮时触发。准备路径并打开设置对话框。 |
| `on_golden_teach` | `main_camera_ui.py` | **逻辑**。点击“黄金示教”按钮时触发。管理图片选择并调用 `run_golden_teach`。 |
| `on_edit_path` | `main_camera_ui.py` | **逻辑**。点击“编辑路径”按钮时触发。管理图片选择并调用 `run_path_teach`。 |

## 📄 3. 生成的数据文件

| 文件模式 | 描述 |
| :--- | :--- |
| `mold2.png` | 用于参考的“黄金模板”图片。 |
| `slot_templates.npy` | 存储在黄金示教中定义的 4 个槽位位置（框）。 |
| `warp_size.npy` | 存储透视校正后模板的尺寸 `[宽, 高]`。 |
| `cell_0_polylines.npy` | 存储点胶路径的原始**线段**数据（局部坐标）。 |
| `cell_0_dots.npy` | 存储点胶路径的原始**点**数据（局部坐标）。 |
| `disp_poly_cell0.npy` | 存储用于显示的**线段**路径（黄金坐标系）。 |
| `disp_dots_cell0.npy` | 存储用于显示的**点**路径（黄金坐标系）。 |
| `golden_corners.npy` | 存储标定好的模具视觉角点。 |
| `cam2mach_matrix.npy` | 存储相机到机器的坐标变换矩阵。 |
| `app_settings.json` | 存储应用程序设置（速度、Z轴高度、延时等）。 |
| `camera_params.json` | 存储相机参数（曝光、增益、Gamma）。 |
| `captures/` | 包含从相机保存的截图目录。 |
