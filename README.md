# 🎯 DIY多色视觉引导点胶机

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

**一个基于Python的工业级自动化点胶解决方案**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [文档](#-文档) • [演示](#-演示) • [贡献](#-贡献)

</div>

---

## 📖 项目简介

这是一个完整的**视觉引导多色点胶自动化系统**，集成了机器视觉、运动控制和智能路径规划，可用于PCB涂胶、电子封装、精密点胶等工业场景。

### 核心优势

- 🎨 **多色支持**：独立控制4路颜色通道（A1-A4），实现复杂配色需求
- 🎯 **视觉定位**：支持Halcon/OpenCV双引擎，亚像素级精度识别
- 🤖 **智能控制**：基于Klipper固件，高精度三轴运动控制
- 🖼️ **可视化路径编辑**：图形化界面设计点胶路径，所见即所得
- ⚡ **批量生产**：自动识别多产品位置，一键完成阵列化点胶

---

## ✨ 功能特性

<table>
<tr>
<td width="50%">

### 🎨 视觉识别
- Halcon形状匹配（亚像素精度）
- OpenCV模板匹配（开源方案）
- 自动多目标定位
- 实时畸变校正

</td>
<td width="50%">

### 🤖 运动控制
- Klipper固件集成
- WebSocket实时通信
- 8点坐标标定
- Z轴自动对刀

</td>
</tr>
<tr>
<td width="50%">

### 🌈 多色点胶
- 4路独立颜色控制
- 多阀联动逻辑
- 可视化路径编辑
- 点/线混合模式

</td>
<td width="50%">

### 📊 生产管理
- 配方管理系统
- 批量运行模式
- G代码预览
- 实时状态监控

</td>
</tr>
</table>

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.8 | 推荐3.9+ |
| PyQt5 | ≥ 5.15 | UI框架 |
| OpenCV | ≥ 4.5 | 视觉库 |
| NumPy | ≥ 1.20 | 数值计算 |
| Klipper | 最新版 | 运动控制固件 |

### 安装步骤

**1. 克隆仓库**
```bash
git clone https://github.com/sqf4411/DIY-Multi-Color-Vision-Guided-Dispensing-Machine.git
cd DIY-Multi-Color-Vision-Guided-Dispensing-Machine
```

**2. 创建虚拟环境**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

**3. 安装依赖**
```bash
pip install -r requirements.txt
```

**4. 配置相机（可选）**
```bash
# 安装Hikrobot MVS SDK
# 设置环境变量 MVS_DLL_DIR
```

**5. 启动程序**
```bash
python main_camera_ui.py
```

---

## 📁 项目结构

```
DIY-Multi-Color-Vision-Guided-Dispensing-Machine/
├── 📄 main_camera_ui.py          # 主程序入口
├── 📂 core/                       # 核心模块
│   ├── camera_manager.py         # 相机管理
│   ├── machine_client.py         # 运动控制
│   ├── halcon_manager.py         # Halcon识别
│   ├── mold_detector.py          # OpenCV检测
│   ├── path_editor.py            # 路径编辑
│   ├── multicolor_path_editor.py # 多色编辑器
│   ├── dry_run_dialog.py         # 空跑模拟
│   ├── batch_run_dialog.py       # 批量运行
│   └── config_manager.py         # 配置管理
├── 📂 produces/                   # 产品配方
│   └── {配方名}/
│       ├── reference.png         # 训练图片
│       ├── {配方}.shm            # Halcon模型
│       ├── multicolor_path.json  # 多色路径
│       └── config.json           # 配方配置
├── 📂 config/                     # 全局配置
│   ├── params.json               # 系统参数
│   └── app_settings.json         # 应用设置
├── 📂 scripts/                    # 辅助脚本
│   └── detect_mold.hdev          # Halcon脚本
└── 📂 docs/                       # 文档
    ├── complete_guide.md         # 完整指南
    └── coordinate_system_manual.md # 坐标系统
```

### 核心模块说明

| 模块 | 功能 | 关键类/函数 |
|------|------|------------|
| `camera_manager.py` | 相机连接与取流 | `CameraManager` |
| `machine_client.py` | Klipper通信 | `MachineClient` |
| `halcon_manager.py` | Halcon识别引擎 | `HalconManager` |
| `path_editor.py` | 路径可视化编辑 | `PathEditor` |
| `multicolor_path_editor.py` | 多色路径规划 | `MultiColorPathEditor` |
| `dry_run_dialog.py` | G代码生成与模拟 | `DryRunDialog` |
| `batch_run_dialog.py` | 批量生产控制 | `BatchRunDialog` |

---

## 📚 文档

### 新手入门
- 📖 [完整项目指南](docs/complete_guide.md) - 从零开始学习整个项目
- 🗺️ [坐标系统详解](docs/coordinate_system_manual.md) - 理解4种坐标系转换

### 进阶开发
- 🔧 API参考手册（见complete_guide.md第7章）
- 🎨 自定义路径算法（见complete_guide.md第9.1节）
- ⚡ 性能优化技巧（见complete_guide.md第9.3节）

### 常见问题
- ❓ [FAQ](docs/complete_guide.md#8-常见问题faq)
- 🐛 [问题排查指南](docs/complete_guide.md#q1-相机无法连接怎么办)

---

## 🎬 演示

### 主界面
*实时视频流 + 识别结果叠加显示*

### 多色路径编辑器
*拖拽式路径设计，支持点和线两种模式*

### 批量运行
*自动识别4x4阵列，一键完成16个产品点胶*

> 📹 视频演示：[待上传]

---

## 🛠️ 技术栈

**前端界面**
- PyQt5 - 跨平台GUI框架
- QThread - 多线程管理

**视觉算法**
- OpenCV - 图像处理
- Halcon (可选) - 工业视觉库
- NumPy - 矩阵运算

**运动控制**
- Klipper - 固件
- Moonraker - WebSocket API
- WebSocket - 实时通信

**数据管理**
- JSON - 配置存储
- NumPy (.npy) - 数组序列化

---

## 🔧 硬件要求

### 必需组件
- **工业相机**：Hikrobot MV系列（或USB摄像头用于测试）
- **运动平台**：Klipper兼容主板（如SKR系列）
- **步进电机**：NEMA17 x 3（XYZ三轴）
- **气动阀**：24V电磁阀 x 4（多色切换）

### 可选组件
- Z轴探针（BLTouch/机械探针）
- 工业光源
- 专用点胶针头

### 推荐配置
- CPU: i5或更高
- RAM: 8GB+
- 操作系统: Windows 10/11

---

## 🌟 特色功能

### 1. 智能多阀控制
采用"主路+分路"气路设计，确保无气压倒灌：
```
开启序列: 颜色阀 → 主路A → 主路B
关闭序列: 主路B → 主路A → 颜色阀（逆序安全）
```

### 2. 4种坐标系统
- **Raw**: 相机原始图像
- **Warped**: 透视校正后
- **Cell**: 单产品区域
- **Machine**: 机器物理坐标

完整的坐标转换链，保证微米级精度。

### 3. 双视觉引擎
- **Halcon模式**：商业级精度，适合工业生产
- **OpenCV模式**：开源免费，适合学习开发

### 4. 配方管理
每个产品独立配方，包含：
- 训练图片
- 识别模型
- 点胶路径
- 工艺参数

一键切换，互不干扰。

---

## 📈 开发路线图

- [x] 基础视觉引导功能
- [x] 多色点胶支持
- [x] 批量生产模式
- [x] 配方管理系统
- [x] 完整文档
- [ ] 云端配方库
- [ ] 机器学习优化
- [ ] Web远程监控
- [ ] 移动端APP

---

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. **Fork** 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add: AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 **Pull Request**

### 代码规范
- 遵循PEP8
- 添加类型注解
- 编写文档字符串
- 包含单元测试

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

您可以自由地：
- ✅ 商业使用
- ✅ 修改
- ✅ 分发
- ✅ 私人使用

但需要：
- 📝 保留版权声明
- 📝 包含许可证副本

---

## 🙏 致谢

感谢以下开源项目和团队：

- [Klipper](https://github.com/Klipper3d/klipper) - 高性能3D打印固件
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - Python GUI框架
- [NumPy](https://numpy.org/) - 科学计算库
- MVTec公司 - Halcon工业视觉库

---

## 📮 联系方式

- **作者**: sqf4411
- **仓库**: [GitHub](https://github.com/sqf4411/DIY-Multi-Color-Vision-Guided-Dispensing-Machine)
- **问题反馈**: [Issues](https://github.com/sqf4411/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个Star！⭐**

Made with ❤️ by sqf4411

</div>
