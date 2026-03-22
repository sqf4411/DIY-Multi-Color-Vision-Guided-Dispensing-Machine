# FIX README V2（复检 + 修复报告）

生成时间：2026-03-22  
分支：`codex/fix-audit-issues`

## 1. 本轮复检范围

- 全项目 Python 编译检查：`python -m compileall -q .`
- 自动化测试（无硬件模式）：`.venv\Scripts\python.exe -m pytest -q`
- 风险扫描：
  - `allow_pickle=True`
  - 硬编码绝对路径（如 `D:\...`）
  - 裸 `except:`
- 核心模块抽查：`core/machine_client.py`、`core/camera_manager.py`、`scripts/utils/repair_ui.py`

## 2. 本轮新增修复（FixV2）

### P1-01 探针状态字符串误判（已修复）

- 文件：`core/machine_client.py`
- 问题：`last_query` 若返回字符串 `"0"`，旧逻辑 `bool("0") == True`，会被误判为 `triggered`。
- 修复：
  - 新增 `_parse_switch_state` 统一解析 `0/1`、`true/false`、`on/off`、`open/triggered`。
  - 应用于 `query_probe_status` / `get_probe_status` / `send_and_get_probe_status`。
- 风险收益：避免误触发状态导致自动流程分支错误。

### P2-01 output_pin 字符串数值兼容（已修复）

- 文件：`core/machine_client.py`
- 问题：若 Moonraker 返回 `"1.0"`（字符串），旧逻辑比较浮点时可能异常并误判。
- 修复：统一转 `float(val)` 再比较，异常时安全返回 `False`。

### P2-02 相机 SDK 缺失时的稳定性（已修复）

- 文件：`core/camera_manager.py`
- 问题：`MvImport` 导入失败时，后续调用可能触发 `NameError` 风险。
- 修复：
  - 增加 `MV_IMPORT_AVAILABLE` 标志位。
  - `open_camera()` 在 SDK 不可用时直接返回 `False`。
  - 将模块内裸 `except:` 收敛为 `except Exception:`。

### P3-01 修复脚本可移植性（已修复）

- 文件：`scripts/utils/repair_ui.py`
- 问题：存在硬编码绝对路径，跨机器不可用。
- 修复：
  - 改为基于项目相对路径自动定位。
  - 支持 `--target` / `--snippet` 参数。
  - 增加文件存在性校验与清晰错误返回码。

### P3-02 测试回归增强（已修复）

- 文件：
  - `tests/test_machine_client_probe_parsing.py`（新增）
  - `tests/test_product_manager_safety.py`（更新）
- 修复：
  - 新增 3 条无硬件单测覆盖探针/输出状态解析。
  - 现有 `product_manager` 测试改为 `tmp_path`，避免污染仓库目录。

## 3. 复检结果

- `compileall`：通过
- `pytest`：`5 passed`
- `allow_pickle=True`：0 命中
- Python 文件硬编码绝对路径：0 命中（排除 `.gemini` 备份目录）

## 4. 当前仍建议优化（未在本轮全部改动）

### P2 建议：继续清理裸 `except:`

当前仍有少量裸 `except:`（例如 `core/mold_detector.py`、`main_camera_ui.py`、`mobile_ui/*`）。  
建议后续分批替换为明确异常类型，并补上下文日志，减少静默失败。

### P3 建议：补充更多“无硬件回归”测试

建议增加：
- 关键配置加载/保存的 roundtrip 测试
- 视觉路径数据兼容性（legacy -> JSON）测试
- 关键 UI 逻辑的服务层单测（脱离 PyQt 实例）

## 5. FixV2 版本结论

FixV2 已完成并通过自动化回归。当前高风险问题已收敛，项目在“无硬件环境”下的可运行稳定性与可维护性均有提升。下一阶段建议聚焦异常处理标准化与测试覆盖扩展。
