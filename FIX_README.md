# FIX README（代码与逻辑审计报告）

生成时间：2026-03-22  
审计方式：静态代码审查 + 基础可运行检查（无硬件）

## 1. 检查范围

- 扫描了项目内 66 个 Python 文件（排除 `.venv`、`.git`、`__pycache__`）。
- 执行了 `python -m compileall -q .`（发现 1 个语法/缩进问题文件）。
- 尝试执行 `pytest`（当前虚拟环境未安装 `pytest`，无法直接跑自动化用例）。

## 2. 严重级别定义

- `P0`：高概率造成设备/数据严重损害，需立即修复。
- `P1`：高风险逻辑/安全问题，建议优先修复。
- `P2`：中风险功能正确性/稳定性问题，应尽快修复。
- `P3`：低风险质量问题或工程化缺陷，可排期修复。

## 3. 发现清单（按严重级别）

### P1-01：产品名未校验，存在路径穿越与越界删除风险

- 位置：
  - `core/product_manager.py:35`
  - `main_camera_ui.py:6286-6289`
  - `main_camera_ui.py:6419-6421`
- 现象：
  - 产品名直接参与路径拼接并 `abspath`，可用 `..\\` 越界到 `produces` 目录之外。
  - 删除产品时直接对拼接路径执行 `shutil.rmtree(...)`。
- 风险：
  - 可将训练/保存文件写到项目外部目录，甚至误删项目外目录。
  - 在生产场景属于高风险文件系统操作漏洞。
- 修复建议：
  - 只允许白名单命名（如 `^[A-Za-z0-9_-]{1,64}$`）。
  - 路径拼接后做 `resolved_path.startswith(products_root)` 强校验。
  - 删除前二次校验目标路径必须位于 `products_root` 下。

### P1-02：配方加载失败时仍继续执行“成功后流程”

- 位置：
  - `main_camera_ui.py:6457`
  - `main_camera_ui.py:6459-6468`
  - `main_camera_ui.py:6470-6628`
- 现象：
  - `success, msg = load_product(name)` 后，仅部分赋值在 `if success` 内；
  - 但后续 UI 参数同步、路径覆盖、`"产品加载成功"` 日志在 `if success` 外，加载失败也会执行。
- 风险：
  - 界面显示“加载成功”，实际可能是旧参数/旧路径，造成误操作。
  - 在点胶业务中属于高风险逻辑一致性缺陷。
- 修复建议：
  - 将 `6470-6628` 全部纳入 `if success` 分支。
  - `if not success` 立即 `QMessageBox.critical(...)` 并 `return`。
  - 增加失败回滚（清空缓存、禁止运行按钮）。

### P1-03：多处 `np.load(..., allow_pickle=True)` 读取可变更文件，存在反序列化执行风险

- 位置（典型）：
  - `core/product_manager.py:174`
  - `core/product_manager.py:219`
  - `main_camera_ui.py:4233`
  - `main_camera_ui.py:4562`
  - `main_camera_ui.py:6085`
- 现象：
  - 对产品目录/配置目录中的 `.npy` 使用 `allow_pickle=True`。
- 风险：
  - 若文件被替换为恶意 pickle 载荷，加载时可执行任意代码（本地供应链风险）。
- 修复建议：
  - 优先改为 JSON/NPZ（纯数值）格式。
  - 必须用 pickle 时，加签名校验（hash/签名）与来源校验。
  - 对外部导入文件做隔离与审核。

---

### P2-01：`get_probe_status` 逻辑不完整，函数恒返回 `'unknown'`

- 位置：
  - `core/machine_client.py:251-258`
- 现象：
  - 请求后仅取出 `status`，未解析 `last_query` 并返回；
  - 函数尾部固定 `return 'unknown'`。
- 风险：
  - 探针状态不可用，影响 UI 状态可信度和自动流程判断。
- 修复建议：
  - 解析 `last_query`：`1 -> triggered`，`0 -> open`，异常时才 `unknown`。

### P2-02：批量流程内重复视觉识别，造成性能浪费与状态不一致风险

- 位置：
  - `core/batch_run_dialog.py:170`
  - `core/batch_run_dialog.py:207`
- 现象：
  - 同一帧 `gray_img` 被 `get_vision_result(...)` 调用两次；
  - 第一次结果先写缓存，后又被第二次覆盖。
- 风险：
  - 识别开销翻倍，批量节拍下降；
  - 混合模式下产品切换前后结果可能不一致。
- 修复建议：
  - 统一为“产品确认后只识别一次并缓存”。
  - 若需预览，明确分离 preview 结果与生产结果缓存。

### P2-03：路径处理仍有 CWD 依赖，跨目录启动易异常

- 位置（典型）：
  - `core/product_manager.py:57`
  - `core/product_manager.py:147`
  - `main_camera_ui.py:3312`
- 现象：
  - 部分配置路径使用相对路径（如 `Path("config")`、`np.load("config/...")`）。
- 风险：
  - 从非项目根目录启动时，读写到错误目录或直接报文件不存在。
- 修复建议：
  - 统一使用“项目根绝对路径 + 相对资源路径”的路径构建策略。
  - 抽出 `get_project_root()` 统一管理。

---

### P3-01：存在不可编译的 Python 文件，影响质量门禁

- 位置：
  - `scripts/utils/halcon_methods_snippet.py:2`
- 现象：
  - `compileall` 报错：`IndentationError: unexpected indent`。
- 风险：
  - 自动检查流程失败；后续工具链（静态分析/打包）受阻。
- 修复建议：
  - 若仅为片段，改名为 `.md/.txt`；
  - 若要保留为 `.py`，补齐为合法模块结构。

### P3-02：测试工程化不完整（当前环境无法直接执行测试）

- 位置：
  - 项目依赖缺少 `pytest`（当前 `.venv` 中无该模块）
  - `tests/` 中多为硬件/环境依赖脚本
- 现象：
  - 无法“一键回归”验证本次修复是否引入回归。
- 风险：
  - 变更质量依赖人工回归，成本高且易漏测。
- 修复建议：
  - 增加 `requirements-dev.txt`（至少 `pytest`）。
  - 将硬件依赖测试打 `@pytest.mark.hardware`，默认跳过。
  - 增加最小无硬件 smoke tests（配置加载、路径处理、产品读写）。

## 4. 建议修复顺序

1. 先修 `P1-01`、`P1-02`、`P1-03`（安全与错误状态传播）。
2. 再修 `P2-01`、`P2-02`、`P2-03`（稳定性与可维护性）。
3. 最后处理 `P3`（工程化与质量门禁）。

## 5. 说明

- 本报告基于当前无硬件环境，重点覆盖代码层面的逻辑/安全风险。
- 若进入实施阶段，建议每修复一项都补对应最小回归用例。
