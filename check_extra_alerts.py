"""
check_extra_alerts.py
推論でC2サーバー以外に警告が出たペアが、Zeekの正解ラベル上で何だったかを確認する。
（推論には正解ラベルを使っていないため、答え合わせとして見る）
リポジトリ直下で実行:  python check_extra_alerts.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b"))
import pandas as pd
from config import INTERMEDIATE_DIR

PAIRS = [("123.59.209.185", 80), ("71.61.66.148", 80), ("50.50.50.53", 53), ("66.67.61.168", 63798),
         ("185.244.25.235", 80)]
lab = pd.read_parquet(os.path.join(INTERMEDIATE_DIR, "labeled_34-1.parquet"))
for ip, port in PAIRS:
    g = lab[(lab["peer_ip"] == ip) & (lab["peer_port"] == port)]
    print(f"\n=== {ip}:{port}  フロー数 {len(g)} ===")
    print("  Zeekの詳細ラベル:", g["zeek_detailed"].fillna("-").value_counts().to_dict())
    print("  学習時の扱い    :", g["exclude_reason"].fillna(
        g["label_final"].map({1.0: "Label=1(C2)", 0.0: "Label=0(正常)"})).value_counts().to_dict())
    print("  接続状態        :", g["conn_state"].fillna("-").value_counts().head(4).to_dict())
