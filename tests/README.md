# Tests Directory

这个目录包含所有的测试文件。

## 测试文件列表

- `test_halcon_call.py` - Halcon调用测试
- `test_halcon_genimage.py` - Halcon图像生成测试  
- `test_halcon_insp.py` - Halcon检查测试
- `test_machine_client.py` - 机器客户端测试
- `test_touch_numpad.py` - 触摸数字键盘测试

## 运行测试

```bash
# 运行单个测试
python tests/test_machine_client.py

# 或者从项目根目录运行
python -m tests.test_machine_client
```
