# Halcon ROI 调整指南 - 解决路径偏移问题

## 📋 问题总结

**现象**：Halcon识别后的点胶路径不精确

**原因**：
- 🔴 **Halcon ROI**：只框选产品本体（不含腔壁）
- 🟡 **OpenCV黄框**：框选整个产品（含腔壁）
- ⚠️ 两者中心点不重合，导致路径映射偏移

**解决方案**：修改Halcon ROI，使其与OpenCV黄框一致（包含腔壁）

---

## 🛠️ 操作步骤

### 步骤1️⃣：测量OpenCV黄框尺寸（参考）

**可选步骤**，用于了解目标ROI大小

```python
# 在Python中查看
import numpy as np
templates = np.load('config/slot_templates.npy')
box0 = templates[0]  # 第0个黄框
print(f"OpenCV黄框坐标: {box0}")
# 输出示例：[[col1, row1], [col2, row1], [col2, row2], [col1, row2]]
```

或者直接在UI中：
1. 点击"黄金示教"
2. 查看第0个框的位置和大小
3. 记录大致范围

---

### 步骤2️⃣：在Halcon中重新绘制ROI

#### 打开Halcon HDevelop

```halcon
* 1. 读取训练图像
read_image (Image, 'D:/CodeFromGithub/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/config/mold2.png')

* 2. 显示图像
dev_display (Image)

* 3. 使用ROI工具
* 菜单：Visualization → ROI → Rectangle1
* 或者直接在代码中调整坐标
```

#### 绘制新ROI（包含腔壁）

**关键**：框选范围要和OpenCV黄框一样大！

```halcon
* 原来的ROI（只含产品本体）
gen_rectangle1 (OldROI, 448.829, 936.949, 854.524, 1289.7)

* 新ROI（包含腔壁）- 坐标需要实际测量
* 参考：Row1, Col1, Row2, Col2
gen_rectangle1 (NewROI, NewRow1, NewCol1, NewRow2, NewCol2)

* 显示对比
dev_set_color ('red')
dev_display (OldROI)
dev_set_color ('green')
dev_display (NewROI)
```

**测量技巧**：
- 使用鼠标悬停查看坐标
- 确保ROI包含完整的腔壁边框
- 上下左右都要留足够边距

---

### 步骤3️⃣：更新detect_mold.hdev文件

**文件位置**：
```
D:\CodeFromGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\scripts\detect_mold.hdev
```

**修改内容**：

找到这一行（约第46行）：
```halcon
gen_rectangle1 (ModelRegion, 448.829, 936.949, 854.524, 1289.7)
```

替换为新坐标：
```halcon
gen_rectangle1 (ModelRegion, 新Row1, 新Col1, 新Row2, 新Col2)
```

**示例**（实际坐标以你测量为准）：
```halcon
* 假设测量到的新坐标
gen_rectangle1 (ModelRegion, 400.0, 880.0, 900.0, 1340.0)
```

**保存文件** ✅

---

### 步骤4️⃣：删除旧模型文件

**必须步骤**，否则系统会继续使用旧模型

```bash
# 删除这个文件
D:\CodeFromGithub\DIY-Multi-Color-Vision-Guided-Dispensing-Machine\config\mold_model.shm
```

或者：
- 重命名为 `mold_model.shm.old`（作为备份）

---

### 步骤5️⃣：重新训练Halcon模型

#### 在Python UI中：

1. **启动UI**
   ```bash
   python main_camera_ui.py
   ```

2. **勾选"启用 Halcon 识别"**

3. **触发训练**（任选一种方式）：
   - 点击"识别模具"按钮
   - 点击"黄金示教"后运行识别
   
4. **观察日志**：
   ```
   [LOG] ✅ Halcon: 模型不存在，正在训练...
   [LOG] ✅ Halcon: 模型已保存为 mold_model.shm
   ```

5. **确认文件生成**：
   ```
   config/mold_model.shm  ← 新模型文件
   ```

---

### 步骤6️⃣：验证精度

#### 测试1：对比两种模式的中心点

```python
# 1. 关闭Halcon模式
取消勾选"启用 Halcon 识别"

# 2. OpenCV识别
点击"识别模具"
记录第0个产品中心：OpenCV_X, OpenCV_Y

# 3. 开启Halcon模式  
勾选"启用 Halcon 识别"

# 4. Halcon识别
点击"识别模具"
记录第0个产品中心：Halcon_X, Halcon_Y

# 5. 计算偏差
偏差 = sqrt((Halcon_X - OpenCV_X)² + (Halcon_Y - OpenCV_Y)²)
```

**期望结果**：
- ✅ 偏差 < 5像素：完美！
- ⚠️ 偏差 5-10像素：可接受，可能需要微调
- ❌ 偏差 > 10像素：ROI还不够匹配，需要重新调整

#### 测试2：实际点胶路径

1. 使用Halcon识别
2. 执行空跑（虚拟点胶）
3. 观察路径是否精确覆盖产品轮廓
4. 检查4个产品的路径是否一致

---

## ⚠️ 注意事项

### 1. ROI不要太大
- ❌ 不要包含相邻产品
- ❌ 不要包含过多背景
- ✅ 刚好包含单个产品+腔壁即可

### 2. 保持纵横比
- 如果OpenCV黄框是正方形，Halcon ROI也应该接近正方形
- 避免拉伸变形

### 3. 训练图像质量
- 确保`config/mold2.png`清晰无模糊
- 光照均匀
- 产品完整可见

### 4. 备份重要文件
在修改前备份：
```
config/mold_model.shm        → mold_model.shm.backup
scripts/detect_mold.hdev     → detect_mold.hdev.backup
```

---

## 🐛 常见问题

### Q1: 训练后识别不到产品？

**原因**：ROI太大或太小

**解决**：
1. 检查ROI是否包含完整产品
2. 确保产品特征明显（不要全是平整区域）
3. 调整匹配分数阈值：
   ```halcon
   find_shape_model (..., 0.2, ...)  # 降低阈值到0.2
   ```

### Q2: 识别到错误位置？

**原因**：产品特征不够独特

**解决**：
1. 提高匹配分数阈值：
   ```halcon
   find_shape_model (..., 0.5, ...)  # 提高到0.5
   ```
2. 缩小搜索角度范围：
   ```halcon
   find_shape_model (..., rad(-10), rad(10), ...)  # 只允许±10度
   ```

### Q3: 4个产品只识别到2个？

**原因**：重叠阈值设置过高

**解决**：
```halcon
find_shape_model (..., 0.3, ...)  # 降低重叠阈值
```

---

## 📊 成功标准

- ✅ Halcon和OpenCV识别的中心点偏差 < 5像素
- ✅ 点胶路径准确覆盖产品轮廓
- ✅ 4个产品的路径一致性良好
- ✅ 不同光照条件下识别稳定

---

## 🎯 如果还是不精确？

如果按照此方案调整后，路径仍有偏移，可能需要：

### 方案A：微调路径偏移补偿
在代码中添加固定偏移量：
```python
# main_camera_ui.py
HALCON_OFFSET_X = 5  # 根据实际测量调整
HALCON_OFFSET_Y = -3

# 应用路径时
adjusted_center = (center[0] + HALCON_OFFSET_X, 
                   center[1] + HALCON_OFFSET_Y)
```

### 方案B：使用Halcon专用路径
参考之前讨论的方案2，为Halcon单独编辑路径

---

## 📞 需要帮助？

测试过程中遇到任何问题，随时联系！

**记录以下信息方便诊断**：
1. 调整后的ROI坐标
2. OpenCV和Halcon中心点坐标对比
3. 实际偏差数值
4. 识别结果截图

祝测试顺利！🚀

---

## 📝 附录：坐标系对比示意图

```
图示：OpenCV vs Halcon ROI

OpenCV黄框（包含腔壁）
┌─────────────────────────────────┐
│  ████ 腔壁 ████████████████████  │
│  █                           █  │
│  █   Halcon旧ROI（仅本体）   █  │
│  █   ┌───────────────────┐   █  │
│  █   │                   │   █  │
│  █   │   产品本体        │   █  │
│  █   │                   │   █  │
│  █   │   🔴 Halcon中心   │   █  │
│  █   │                   │   █  │
│  █   └───────────────────┘   █  │
│  █                           █  │
│  █   🟡 OpenCV中心 ← 期望目标 █  │
│  ████████████████████████████  │
└─────────────────────────────────┘

调整后：Halcon新ROI = OpenCV黄框
两个中心点重合 → 路径映射精确！
```

---

**文档创建时间**：2025-12-15  
**版本**：v1.0
