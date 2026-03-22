# FIX README（修复后复检报告）

分支：`codex/fix-audit-issues`  
复检时间：2026-03-22

## 1. 修复结论（按原报告分级）

### P1

1. `P1-01 路径穿越/越界删除`：已修复  
   - 关键修改：
   - `core/product_manager.py:24, 50` 增加产品名白名单校验 + `commonpath` 路径边界校验
   - `main_camera_ui.py:6376` 新建产品前做名称合法性校验
   - `main_camera_ui.py:4142` 编辑多色路径前对产品路径异常做拦截

2. `P1-02 加载失败继续走成功流程`：已修复  
   - 关键修改：
   - `main_camera_ui.py:6552` 在 `load_product` 失败后立即弹错并 `return`，阻断后续“成功路径”UI更新。

3. `P1-03 allow_pickle 反序列化风险`：已修复（运行链路）  
   - 关键修改：
   - `core/product_manager.py:85, 180` 路径段改为 `path_poly.json`，读取 `.npy` 时禁用 pickle。
   - `main_camera_ui.py:1140, 1158` 新增安全加载器，替换原 `allow_pickle=True` 调用。
   - `core/mold_detector.py:128, 168` 优先读 JSON，legacy `.npy` 仅读数值格式（`allow_pickle=False`）。
   - `core/path_teach.py:159, 164, 429, 457` 迁移为 JSON 存储，并删除 legacy object `.npy`。
   - 仓库扫描结果：`allow_pickle=True` 已为 0 处。

### P2

1. `P2-01 get_probe_status 恒 unknown`：已修复  
   - `core/machine_client.py:259` 正确返回 `open/triggered/unknown`。

2. `P2-02 批量流程重复识别`：已修复  
   - `core/batch_run_dialog.py:190` 保留单次主识别，去掉冗余识别缓存分支。

3. `P2-03 CWD 依赖路径`：已修复（核心运行路径）  
   - `main_camera_ui.py:914, 3393` 配置路径统一为项目绝对路径。
   - `core/product_manager.py:30, 123` `config/produces` 读写统一基于项目根目录。

### P3

1. `P3-01 compileall 被坏文件阻塞`：已修复  
   - `scripts/utils/halcon_methods_snippet.py` 改为合法 Python 占位说明文件。

2. `P3-02 测试基线缺失`：已修复（最小可执行基线）  
   - 新增：
   - `requirements-dev.txt`（含 `pytest`）
   - `pytest.ini`
   - `tests/conftest.py`（默认忽略硬件/Halcon脚本测试）
   - `tests/test_product_manager_safety.py`（路径安全+JSON路径读写单测）

## 2. 复检结果

1. 语法检查  
   - 命令：`python -m compileall -q .`  
   - 结果：通过（无编译错误）

2. 自动化测试  
   - 命令：`python -m pytest -q`  
   - 结果：`2 passed`

3. 高风险加载扫描  
   - 扫描条件：`allow_pickle=True`  
   - 结果：0 命中

## 3. 备注（剩余非阻断项）

1. 项目中仍有部分裸 `except:`（历史代码），建议后续分批替换为显式异常类型并补日志上下文。  
2. 本次回归在“无硬件环境”执行，硬件联调（相机/Moonraker/Halcon）用例仍需在现场环境补测。
