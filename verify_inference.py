"""
verify_inference.py
推論処理（src/engine_b/inference.py）が、学習時と同じ特徴量を作れているかを確認する。
リポジトリ直下に配置して実行する。

確認1【計算の一致】
  学習時のラベル付きBiflow（labeled_34-1.parquet）を推論側のブロック化関数に通し、
  学習データ（dataset_engine_b_features.csv）の34-1と、8特徴量が完全に一致するかを見る。
  → 一致すれば「推論と学習で特徴量の計算・窓の切り方が同じ」と言える。

確認2【PCAPからの通し実行】
  34-1の生PCAPを、正解ラベルを一切使わずに推論にかける。
  既知のC2サーバー（config の c2_server_ips）とのペアが検知されるか、
  それ以外にどれだけ警告が出るかを見る。

実行:
    python verify_inference.py            # 確認1と確認2
    python verify_inference.py --skip-pcap # 確認1のみ（数秒で終わる）
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b"))

import numpy as np
import pandas as pd

from config import CAPTURES, INTERMEDIATE_DIR, OUTPUT_DIR
from inference import build_blocks, analyze_pcap

FEATURES = ["IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio",
            "Burst_Active_Ratio", "Payload_Bytes_MAD", "Byte_Ratio_Median", "Connection_Count"]
KEY = ["peer_ip", "peer_port", "proto", "window_index"]


def check_feature_match(name):
    print("=" * 90)
    print(f"確認1：特徴量計算の一致（{name}）")
    print("=" * 90)
    lab = pd.read_parquet(os.path.join(INTERMEDIATE_DIR, f"labeled_{name}.parquet"))
    kept = lab[lab["label_final"].isin([0, 1])].copy()
    kept["host_ip"] = CAPTURES[name]["host_ip"]
    inf_blocks = build_blocks(kept)

    train = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_engine_b_features.csv"))
    train = train[train["capture"] == name].copy()
    parts = train["block_id"].str[len(name) + 1:].str.split("_", expand=True)
    train["peer_ip"] = parts[0]
    train["peer_port"] = parts[1].astype(int)
    train["proto"] = parts[2]
    train["window_index"] = parts[3].astype(int)

    m = train.merge(inf_blocks, on=KEY, how="left", suffixes=("_train", "_infer"),
                    indicator=True)
    n_train = len(train)
    n_found = int((m["_merge"] == "both").sum())
    print(f"  学習データのブロック数        : {n_train:,}")
    print(f"  推論側で同じブロックが見つかった: {n_found:,}")
    print(f"  （推論側の総ブロック数 {len(inf_blocks):,}。学習時に除外した混在ブロックも含むため多い）")

    both = m[m["_merge"] == "both"]
    all_ok = n_found == n_train
    print("\n  特徴量ごとの一致状況:")
    for c in FEATURES:
        a = both[f"{c}_train"].values.astype(float)
        b = both[f"{c}_infer"].values.astype(float)
        ok = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        all_ok &= bool(ok.all())
        print(f"    {c:22s}: 不一致 {int((~ok).sum()):>5} 件")
    print("\n  判定:", "✅ 完全に一致（学習と推論の計算は同じ）" if all_ok
          else "❌ 不一致あり（上の件数を確認してください）")
    return all_ok


def check_pcap(name):
    from common import find_pcap

    print("\n" + "=" * 90)
    print(f"確認2：PCAPからの通し実行（{name}、正解ラベルは使わない）")
    print("=" * 90)
    pcap = find_pcap(CAPTURES[name]["dir"])
    summary, blocks, stats = analyze_pcap(pcap)
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")
    if len(summary) == 0:
        print("  ブロックが作られませんでした")
        return

    c2_ips = set(CAPTURES[name]["c2_server_ips"])
    is_c2 = summary["peer_ip"].isin(c2_ips)
    cols = ["host_ip", "peer_ip", "peer_port", "proto", "n_blocks", "n_alert_blocks",
            "alert_ratio", "max_c2_proba"]
    print("\n--- 既知のC2サーバーとのペア ---")
    print(summary.loc[is_c2, cols].round(3).to_string(index=False) if is_c2.any()
          else "  （ブロックが作られたペアなし）")
    others = summary[~is_c2]
    print(f"\n--- それ以外のペア：{len(others)} ペア中 {int(others['is_alert'].sum())} ペアで警告 ---")
    if others["is_alert"].any():
        print(others.loc[others["is_alert"], cols].head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="34-1")
    ap.add_argument("--skip-pcap", action="store_true")
    args = ap.parse_args()
    check_feature_match(args.capture)
    if not args.skip_pcap:
        check_pcap(args.capture)
