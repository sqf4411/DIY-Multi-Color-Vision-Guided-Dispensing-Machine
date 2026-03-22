# DIY Multi-Color Vision-Guided Dispensing Machine (V3.1)

工业点胶控制项目，支持多颜色路径、视觉定位（Halcon/OpenCV）、Klipper 运动控制、配方管理与批量生产流程。

## 1. 项目目标

本项目用于“视觉引导 + 多阀点胶 + 运动平台”一体化控制，核心能力：

- 多颜色路径编辑与执行（点/线混合）
- 视觉识别后自动映射点胶路径
- 通过 Moonraker 接口控制 Klipper 运动平台
- 配方化管理（模型、参数、路径、标定数据）
- 支持无硬件环境下 UI 与逻辑联调

## 2. V3.1 版本重点

V3.1 在 V3 基础上新增了批量颜色切换工艺检查点：

- 每种颜色完成后可自动停靠到“换胶停靠 XY”
- 自动弹出确认对话框：`测试出胶 / 继续执行 / 停止任务`
- 支持配置是否启用该流程（相机批量默认不触发）

V3 的稳定性与可维护性优化继续保留：

- 清理关键代码中的裸 `except:`（改为显式异常类型）
- 强化 `mold_detector` 的配置加载与数据形状校验
- 修复潜在静默失败路径，补充更明确的错误输出
- 完成完整项目 README 重写

详细修复记录见：

- `FIX_README.md`（初次审计）
- `FIX_README_AFTER.md`（首次修复后复检）
- `FIX_README_V2.md`（V2 修复）
- `FIX_README_V3.md`（V3 修复与建议）
- `FIX_README_V3_1.md`（V3.1 工艺流程增强）

## 3. 技术栈

- Python 3.10+（建议）
- PyQt5（桌面 UI）
- OpenCV + NumPy（视觉与几何计算）
- Halcon（可选视觉引擎）
- Moonraker/Klipper（运动控制）

## 4. 快速开始

### 4.1 克隆项目

```bash
git clone https://github.com/sqf4411/DIY-Multi-Color-Vision-Guided-Dispensing-Machine.git
cd DIY-Multi-Color-Vision-Guided-Dispensing-Machine
```

### 4.2 创建虚拟环境

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.3 安装依赖

```bash
pip install -r requirements.txt
```

开发测试依赖（可选）：

```bash
pip install -r requirements-dev.txt
```

### 4.4 启动主程序

```bash
python main_camera_ui.py
```

## 5. 无硬件环境调试（推荐先走这一条）

当电脑未连接相机/运动平台时，可以先验证 UI 和业务逻辑：

1. 启动 `main_camera_ui.py`
2. 使用“纯模拟模式”（UI 顶部按钮）
3. 不执行真实运动，先验证：
   - 配方创建/加载
   - 路径编辑与保存
   - 参数加载和日志流程

说明：

- 未安装 Halcon 时会有告警，但不影响 OpenCV 模式和大多数 UI 调试。
- 未连接相机或 Klipper 时，相关状态将显示不可用或空帧。

## 6. 硬件接入说明（生产环境）

### 6.1 相机

- 工业相机依赖 Hikrobot MVS SDK（`MvImport`）
- 若 SDK 不存在，系统会拒绝打开工业相机并安全返回，不会直接崩溃

### 6.2 运动控制

- 通过 Moonraker HTTP API 与 Klipper 通信
- 建议先在 Fluidd/Mainsail 中验证：
  - 回零（G28）
  - 基础移动（G0/G1）
  - 探针对象状态（`probe` 或宏变量）

### 6.3 视觉引擎

- Halcon：高精度工业方案（可选）
- OpenCV：默认开源方案，可用于大多数联调场景

## 7. 项目结构

```text
.
├─ main_camera_ui.py
├─ core/
│  ├─ machine_client.py
│  ├─ camera_manager.py
│  ├─ mold_detector.py
│  ├─ product_manager.py
│  ├─ halcon_manager.py
│  ├─ halcon_product_manager.py
│  ├─ batch_run_dialog.py
│  └─ ...
├─ mobile_ui/
├─ scripts/
│  └─ utils/
├─ tests/
├─ config/
├─ produces/
└─ docs/
```

## 8. 测试与质量检查

### 8.1 编译检查

```bash
python -m compileall -q .
```

### 8.2 运行测试

```bash
python -m pytest -q
```

当前测试基线为无硬件 smoke tests，重点覆盖：

- 产品路径读写安全
- 探针状态解析逻辑
- 关键格式兼容逻辑

## 9. 关键设计约束

- 不信任外部文件输入：`.npy` 加载禁用 pickle
- 产品名与路径严格校验，避免路径穿越
- UI/设备异常尽量显式处理，减少静默失败
- 默认以“无硬件可调试”作为开发体验基线

## 10. 常见问题

### Q1: `ModuleNotFoundError: No module named 'OpenGL'`

安装依赖：

```bash
pip install PyOpenGL PyOpenGL_accelerate
```

并确认你正在虚拟环境中运行程序。

### Q2: PowerShell 已显示 `(.venv)`，但包仍找不到

请确认实际执行器是虚拟环境解释器：

```powershell
python -c "import sys; print(sys.executable)"
```

应指向项目内 `.venv\Scripts\python.exe`。

### Q3: UI 能打开但没有图像

无硬件时属于正常现象。可先验证 UI 和配置流程，不影响代码调试。

## 11. 开发协作建议

- 分支建议：`codex/*`、`feature/*`、`fix/*`
- 提交建议：单一主题、附带测试结果
- 合并前建议至少执行：
  - `compileall`
  - `pytest -q`

## 12. 许可证

MIT License。
