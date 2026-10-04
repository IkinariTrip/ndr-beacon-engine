"""
build_dataset_engine_b_v25.py
仕様書v2.5準拠 エンジンB データ構築パイプライン 実行スクリプト。
旧 build_dataset_engine_b.py（Geminiが作成した初期版）とは別物。
旧ファイルは今後使用しない（README.md 更新履歴 v2.5 参照）。
使い方:
    # まず小さいキャプチャで動作確認する（推奨）
    python build_dataset_engine_b_v25.py hue-4-1 echo-5-1 3-1 20-1 21-1 8-1 34-1
    # 動作確認後、大きいキャプチャ(7-1, 48-1)も含めてすべて実行する
    python build_dataset_engine_b_v25.py --all
"""
import os
import sys

# src/engine_b を import パスに追加する
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "src", "engine_b"))

from config import CAPTURES  # noqa: E402
import stage1_extract  # noqa: E402
import stage2_biflow  # noqa: E402
import stage3_label  # noqa: E402
import stage4_features  # noqa: E402


def run(targets):
    print("#" * 70)
    print("# 段階1: パケット抽出 (tshark)")
    print("#" * 70)
    for name in targets:
        stage1_extract.extract_one(name)

    print("\n" + "#" * 70)
    print("# 段階2: Biflow生成")
    print("#" * 70)
    for name in targets:
        stage2_biflow.build_biflows(name)

    print("\n" + "#" * 70)
    print("# 段階3: ラベル照合")
    print("#" * 70)
    for name in targets:
        stage3_label.label_one(name)

    print("\n" + "#" * 70)
    print("# 段階4: ブロック生成・特徴量計算")
    print("#" * 70)
    stage4_features.main(targets)


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = list(CAPTURES) if "--all" in args else [a for a in args if not a.startswith("--")]
    if not targets:
        print("使い方: python build_dataset_engine_b_v25.py <capture名> [<capture名> ...] | --all")
        print("利用可能なキャプチャ名:", ", ".join(CAPTURES))
        sys.exit(1)
    run(targets)