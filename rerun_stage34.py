"""
rerun_stage34.py
7-1・48-1の段階1・2（パケット抽出・Biflow生成）は完了済みのため、
修正済みの段階3（ラベル照合）だけをやり直し、続けて段階4を
全9キャプチャ分まとめて実行する。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b"))
import stage3_label
import stage4_features

for name in ["7-1", "48-1"]:
    stage3_label.label_one(name)

ALL_CAPTURES = [
    "hue-4-1", "echo-5-1", "3-1", "20-1", "21-1", "8-1", "34-1", "7-1", "48-1",
]
stage4_features.main(ALL_CAPTURES)