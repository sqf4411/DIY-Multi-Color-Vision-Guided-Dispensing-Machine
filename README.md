# DIY 多色视觉引导点胶机 (DIY Multi-Color Vision Guided Dispensing Machine)

本项目使用 Python、OpenCV 和 PyQt5 实现了一个视觉引导的点胶系统。它通过 HTTP/WebSocket 接口与运动控制器 (Klipper) 通信，根据视觉反馈执行精确的点胶任务。

## 🚀 优化重构版本 (v1)

本分支 `optimization/refactor-v1` 对项目进行了核心重构，旨在提高代码的模块化程度、稳定性和可维护性。

## 📂 1. 项目结构与源代码

### 核心模块 (`core/`)
| 文件名 | 描述 |
| :--- | :--- |
| `config_manager.py` | **【新增】配置管理器**。统一管理应用设置 (`app_settings.json`) 和相机参数 (`params.json`)，提供安全的读写接口和默认值回退。 |
| `camera_manager.py` | **【新增】相机管理器**。封装了海康相机 SDK (`MvImport`) 的底层调用，管理图像采集线程，并通过信号 (`pyqtSignal`) 发送图像数据。 |
| `machine_client.py` | **机器控制**。通过 HTTP/WebSocket 处理与运动控制器 (Klipper/Moonraker) 的通信。 |

### 界面与逻辑
| 文件名 | 描述 |
| :--- | :--- |
| `main_camera_ui.py` | **主程序入口**。负责主 GUI 界面和用户交互，现已通过 `ConfigManager` 和 `CameraManager` 解耦了底层逻辑。 |
| `mold_detector.py` | **视觉算法**。处理模具识别、模板匹配以及路径计算。 |
| `dry_run_dialog.py` | **G代码生成**。根据路径和设置生成 G 代码，处理抬刀、速度和延时。 |

### 辅助工具
| 文件名 | 描述 |
| :--- | :--- |
| `golden_teach.py` | **黄金示教**。定义标准模具模板和槽位。 |
| `path_teach.py` | **路径编辑**。绘制点胶路径。 |

## ✨ 2. 优化修改摘要 (Optimization Summary)

### 2.1 架构重构 (Refactoring)
- **解耦上帝类**: 原 `main_camera_ui.py` 中的相机控制和配置读写逻辑被剥离，分别移至 `core/camera_manager.py` 和 `core/config_manager.py`。
- **配置中心化**: 所有的参数读写现在都由 `ConfigManager` 统一处理，修复了之前散落在各处的 `json` 读写可能导致的文件损坏问题。
- **相机封装**: 移除了主 UI 中直接对 `MvCamera` 的调用，改为使用 `CameraManager` 的高层接口 (`open_camera`, `start_grabbing` 等)。

### 2.2 稳定性提升 (Stability)
- **异常处理**: 修复了旧代码中多处 `try...except: pass` 掩盖错误的问题。
- **线程安全**: 相机取流线程现在由管理器内部维护，确保了开启/关闭时的资源正确释放。

### 2.3 功能改进
- **参数持久化**: 现在的参数保存更加可靠，并且在启动时会自动校验配置文件。

## 📄 3. 生成的数据文件 (Config)

配置现在统一存储在 `config/` 目录下：
- `config/app_settings.json`: 应用程序设置（速度、Z轴高度、延时等）。
- `config/params.json`: 相机参数（曝光、增益、Gamma）。
- `config/cam2mach_matrix.npy`: 坐标变换矩阵。
