# -*- coding: utf-8 -*-
"""多模态视觉分析测试：生成一张测试图，直接调用 backend.vision.analyze_image 验证链路。

用法：python scripts/test_vision.py
前置：项目根 .env 已配置 DASHSCOPE_API_KEY（+ 可选 DASHSCOPE_VL_MODEL）。
"""
import sys
import os
import io as _io
import base64

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw

from backend.vision import analyze_image, VISION_DISCLAIMER


def make_test_image() -> str:
    """生成一张模拟舌象的测试图：浅粉背景 + 红色椭圆（无真实医学含义，仅验证链路）"""
    img = Image.new("RGB", (400, 300), (255, 225, 210))
    draw = ImageDraw.Draw(img)
    draw.ellipse([120, 60, 280, 240], fill=(210, 90, 90))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


if __name__ == "__main__":
    data_url = make_test_image()
    print("=== 多模态 · 舌诊测试 ===")
    ans = analyze_image(data_url, task="tongue")
    print(ans)
    print("\n" + VISION_DISCLAIMER)
    print("\n✅ 多模态链路测试完成")
