# FIX README V3.1（批量颜色切换流程增强）

生成时间：2026-03-22  
分支：`codex/fix-v3`

## 1. 变更目标

根据现场工艺要求，补齐“按颜色批次执行时的中间确认流程”：

1. 一种颜色全部完成后，自动移动到固定换胶停靠位（XY）。
2. 弹出小对话框，支持：
   - 测试出胶
   - 继续执行下一颜色
   - 停止任务
3. 用户确认后继续后续颜色，不再重复视觉扫描已缓存路径。

## 2. 代码修改

核心文件：`core/batch_run_dialog.py`

### 2.1 Worker 新增颜色切换检查点

- 新增信号：`color_change_request_signal`
- 新增状态与接口：
  - `confirm_color_change_continue()`
  - `confirm_color_change_stop()`
  - `_await_color_change_confirmation(...)`
- 运行逻辑：
  - 颜色外循环结束后（非最后一色）触发检查点
  - 自动抬安全高度并移动到 `color_change_park_x/y`
  - 等待 UI 对话框返回继续/停止决策

### 2.2 批量界面新增可配置项

- `颜色切换停靠确认（测试出胶后继续）` 开关（默认开启）
- `换胶停靠 X/Y` 坐标输入
- 相关配置持久化：
  - `batch_color_change_pause`
  - `batch_color_change_x`
  - `batch_color_change_y`

### 2.3 新增颜色切换确认弹窗

- 新增槽函数：`on_color_change_request(...)`
- 弹窗按钮：
  - `测试出胶`（可重复点击，不退出弹窗）
  - `继续执行`
  - `停止任务`

### 2.4 顺手修复（阀门收尾）

- 修正批量末尾颜色阀关闭命令，统一为：
  - `SET_PIN PIN=<valve> VALUE=0`
- 避免发送裸字符串阀名导致固件端不执行的风险。

## 3. 与相机批量关系

- `camera_dry` 仍保持连续流程，不触发颜色切换确认弹窗。
- 该流程主要作用于 `nozzle_dry` / `real`，符合换胶工艺。

## 4. 验证结果

- `python -m compileall -q .`：通过
- `.venv\Scripts\python.exe -m pytest -q`：`5 passed`

## 5. V3.1 结论

V3.1 已实现“颜色批次中间停靠 + 人机确认 + 测试出胶”流程，且保持原有路径缓存与批量执行机制。适合你描述的多颜色人工换胶工艺。
