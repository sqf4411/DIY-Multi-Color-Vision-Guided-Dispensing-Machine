# Halcon 识别鲁棒性优化记录

## 1. 问题现象与原因分析
**现象**：
在调整相机高度（Z轴）后，原有的视觉识别系统失效，无法识别出模具，或者只能识别出部分目标。系统对高度和焦距的变化极其敏感，稍微调整一点点就无法识别。

**根本原因**：
原代码使用的是 **“定比例形状匹配” (Standard Shape Matching)** 算法。
*   **特性**：这种算法假设目标物体在图像中的大小（像素尺寸）是完全固定的。
*   **局限性**：当相机高度改变（例如从 100mm 变到 166mm 或 180mm），物体在画面中的成像大小会发生缩放。原算法无法容忍这种缩放，导致匹配分数（Score）急剧下降，低于阈值从而识别失败。

## 2. 代码位置
相关的核心 Halcon 脚本文件位于项目目录：
`H:\CodeForGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\scripts\detect_mold.hdev`

此文件被 `core/halcon_manager.py` 调用，用于执行具体的图像处理和识别逻辑。

## 3. 修改内容详解

为了提高系统的**鲁棒性 (Robustness)**，使其能适应一定范围内的相机高度和焦距误差，我们对 `detect_mold.hdev` 进行了关键修改。

### 修改前 (Old Code)
使用的是 `find_shape_model` 算子：
```halcon
* 仅支持固定大小，无缩放容忍度
find_shape_model (Image, ModelID, rad(0), rad(360), 0.2, 0, 0.5, 'least_squares', 0, 0.9, ResultRow, ResultColumn, ResultAngle, ResultScore)
```

### 修改后 (New Code)
改为了 `find_scaled_shape_model` 算子，并增加了缩放参数：
```halcon
* 启用缩放匹配，容忍通过
* MinScale = 0.9 (允许缩小到 90%)
* MaxScale = 1.1 (允许放大到 110%)
find_scaled_shape_model (Image, ModelID, rad(0), rad(360), 0.9, 1.1, 0.2, 0, 0.5, 'least_squares', 0, 0.9, ResultRow, ResultColumn, ResultAngle, ResultScale, ResultScore)
```

## 4. 优化效果
*   **容差提升**：现在系统允许物体在画面中的大小有 **±10%** 的变化。
*   **实际意义**：这意味着您在调节 Z 轴高度时，不需要像以前那样必须精准对齐到毫米级。只要画面清晰度足够，稍微高一点或低一点，系统都能稳定地抓住所有模具。
