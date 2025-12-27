# Halcon 脚本逐行解析 (detect_mold.hdev)

这个文档对最新的 `detect_mold.hdev` 脚本进行了逐行解释。该脚本的核心功能是**加载/训练模型**并进行**带缩放的形状匹配**。

---

### 1-12: HDevelop 文件头与 Main 函数
```xml
1: <?xml version="1.0" encoding="UTF-8"?>
2: <hdevelop file_version="1.2" halcon_version="20.11">
...
7: <l>exit ()</l>
...
12: </procedure>
```
*   **解释**：这是 HDevelop 文件的标准 XML 结构。`main` 过程是程序的入口，但我们这里只用它的 Local Procedure `detect_mold`，所以 `main` 里只有一个 `exit()`。

---

### 13-27: detect_mold 接口定义
```xml
13: <procedure name="detect_mold">
16:     <par name="Image" base_type="iconic" dimension="0"/>
19:     <par name="ModelPath" base_type="ctrl" dimension="0"/>
22:     <par name="ResultRow" base_type="ctrl" dimension="0"/>
...
27: </interface>
```
*   **解释**：定义了 `detect_mold` 子过程的输入输出接口。
    *   **输入 (Input)**:
        *   `Image`: 待检测的图像（从相机获取）。
        *   `ModelPath` (未使用，见下文): 代码中本意是传入路径，但实际逻辑中写死了路径以保证稳定性。
    *   **输出 (Output)**:
        *   `ResultRow`, `ResultColumn`: 识别到的目标的行列坐标。
        *   `ResultAngle`: 目标的旋转角度。
        *   `ResultScore`: 匹配的分数（相似度）。

---

### 28-34: 变量初始化
```halcon
27: ResultRow := []
28: ResultColumn := []
29: ResultAngle := []
30: ResultScore := []
```
*   **解释**：先清空用于存储结果的数组，防止残留旧数据。`[]` 表示空数组。

---

### 35-42: 定义模型路径与检查
```halcon
33: ModelFile := 'H:/CodeForGithub/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/config/mold_model.shm'
36: file_exists (ModelFile, FileExists)
38: if (FileExists == 0)
```
*   **解释**：
    *   **Line 33**: 这里**强制指定**了模型文件的绝对路径。虽然接口里有 `ModelPath`，但为了防止路径传错，脚本内部优先使用这个硬编码的路径。
    *   **Line 36**: 检查这个 `.shm` 模型文件是否已经存在。
    *   **Line 38**: 如果文件**不存在** (`FileExists == 0`)，则进入**自动训练**流程。

---

### 43-56: 自动训练流程 (如果模型不存在)
```halcon
40:    GoldenImagePath := 'H:/CodeForGithub/DIY-Multi-Color-Vision-Guided-Dispensing-Machine/config/mold2.png'
43:    read_image (TrainImage, GoldenImagePath)
47:    gen_rectangle1 (ModelRegion, 448.829, 936.949, 854.524, 1289.7)
48:    reduce_domain (TrainImage, ModelRegion, TemplateImage)
52:    create_shape_model (TemplateImage, 'auto', rad(0), rad(360), 'auto', 'auto', 'use_polarity', 'auto', 'auto', ModelID)
53:    write_shape_model (ModelID, ModelFile)
```
*   **解释**：
    *   **Line 40**: 指定“黄金模板图片”的路径。
    *   **Line 47**: **关键**。`gen_rectangle1` 生成一个矩形区域。这个坐标 (448, 936...) 是之前训练时自动计算或手动框选确定的“单个模具”的位置。
    *   **Line 48**: `reduce_domain` 把图片裁剪，只保留这个矩形框里的内容作为训练模板。
    *   **Line 52**: `create_shape_model` 训练模型。参数 `rad(0), rad(360)` 表示允许 0-360 度旋转。
    *   **Line 53**: 训练好后，立刻保存到硬盘，下次就不用再训练了。

---

### 57-62: 加载已有模型
```halcon
58:    read_shape_model (ModelFile, ModelID)
```
*   **解释**：如果模型文件已经存在（平时都是这种情况），直接从硬盘加载 `ModelID`，速度很快。

---

### 67-71: 核心匹配算法 (最新优化版)
```halcon
67: * Find Scaled Shape Model (Robust to Scale/Height variations)
71: find_scaled_shape_model (Image, ModelID, rad(0), rad(360), 0.9, 1.1, 0.2, 0, 0.5, 'least_squares', 0, 0.5, ResultRow, ResultColumn, ResultAngle, ResultScale, ResultScore)
```
*   **解释**：这是**最关键**的一行，也是刚才提高鲁棒性的全部秘密。
    *   `Image`: 当前相机拍到的实时画面。
    *   `ModelID`: 加载进来的模具模板。
    *   `rad(0), rad(360)`: 允许模具在画面中任意旋转。
    *   **`0.9, 1.1` (MinScale, MaxScale)**: **核心改动**。允许物体大小在 90% 到 110% 之间变化。这就是为什么您调整相机高度（物体变大变小）后依然能识别的原因。
    *   `0.2` (MinScore): 最小相似度阈值，低于这个分数的不要。
    *   `0` (NumMatches): 也就是“找到所有能找到的”，为了找齐 4 个。
    *   `0.5` (MaxOverlap): 允许两个目标之间最多重叠 50%（防止同一个目标被识别两次）。
    *   `'least_squares'`: 亚像素精度的优化方法。
    *   **`0.5` (Greediness)**: **核心改动**。贪婪度。
        *   默认通常是 0.9（很快，但不仔细）。
        *   改为 **0.5** 后，算法会更“细心”地搜索，虽然慢几毫秒，但能大幅提高在模糊或边缘情况下的识别率。

---

### 74: 清理内存
```halcon
74: clear_shape_model (ModelID)
```
*   **解释**：识别完后，释放模型占用的内存。

---

### End
```halcon
76: return ()
```
*   **解释**：函数结束，将 `ResultRow` 等结果返回给 Python 程序。
