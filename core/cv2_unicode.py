"""
cv2 中文路径安全读写工具
OpenCV 在 Windows 上不支持非 ASCII 路径，
此模块提供 fallback 方案。
"""
import os
import cv2
import numpy as np


def _is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread 的中文路径安全版本"""
    # 先尝试原生方法 (仅当路径为纯 ASCII 时，避免 Windows 上的 C++ imread 报错输出)
    if _is_ascii(path):
        try:
            img = cv2.imread(path, flags)
            if img is not None:
                return img
        except Exception:
            pass
    # Fallback: 使用 numpy 中转
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite 的中文路径安全版本"""
    # 先尝试原生方法 (仅当路径为纯 ASCII 时)
    if _is_ascii(path):
        try:
            if cv2.imwrite(path, img, params):
                return True
        except Exception:
            pass
    # Fallback: 使用 numpy 中转
    try:
        ext = os.path.splitext(path)[1]
        if params:
            _, buf = cv2.imencode(ext, img, params)
        else:
            _, buf = cv2.imencode(ext, img)
        buf.tofile(path)
        return True
    except Exception:
        return False
