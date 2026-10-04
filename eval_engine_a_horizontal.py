"""
eval_engine_a_horizontal.py
エンジンAの水平スキャン検知（ルール）を、IoT-23 の正解ラベル（Zeek）で評価する。
リポジトリ直下に配置して実行する。

【やること】
  1. 各キャプチャのPCAPを、アプリと同じ処理（extract_flows_from_pcap → build_blocks）でブロック化
  2. 正解：エンジンBの段階3で作った labeled_<cap>.parquet から、Zeekラベルが
     「水平ポートスキャン」の宛先（IP・ポート）を集め、感染端末発のフローと突き合わせる
     → ブロック内の過半数が水平スキャンのフローなら「水平スキャンのブロック」
  3. 既存の縦スキャンモデル（CatBoost v4）と、新しい水平スキャンルールの両方で判定し、
     キャプチャごとに検知数・誤検知数を出す
  4. しきい値を少しずつ変えたときの適合率・再現率（どの値が妥当かの確認用）

【注意】
  - extract_flows_from_pcap はPCAP全体を読み込むため、大きいPCAPは時間・メモリを使う。
    既定では200MBを超えるPCAPは飛ばす（--allow-large で実行）。
  - 正解の突き合わせは（宛先IP・宛先ポート）の一致で行う近似。水平スキャンの宛先は
    ほぼ1回しか使われないランダムなIPのため、正常通信と取り違える可能性は小さい。

実行:
    python eval_engine_a_horizontal.py                      # 全キャプチャ（200MB以下）
    python eval_engine_a_horizontal.py --captures 34-1 8-1  # 指定のみ
"""
import os
import sys
import argparse
import itertools

import joblib
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src", "engine_b"))
from src.flow_generator import extract_flows_from_pcap  # noqa: E402
from src.engine_a.features import (flows_to_dataframe, build_blocks, horizontal_scan_flags,  # noqa: E402
                                   VERTICAL_FEATURES, HORIZONTAL_FEATURES, H_RULE)
from config import CAPTURES, INTERMEDIATE_DIR, OUTPUT_DIR  # noqa: E402
from common import find_pcap  # noqa: E402

MODEL_PATH = os.path.join(REPO, "models", "model_catboost_v4.joblib")
WINDOW = 25
VERTICAL_TH = 0.5
OUT_DIR = os.path.join(OUTPUT_DIR, "engine_a_horizontal")


def load_vertical_model():
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_names") or VERTICAL_FEATURES
    return payload, VERTICAL_FEATURES


def scan_targets(cap):
    """Zeekラベルが水平ポートスキャンの（宛先IP, 宛先ポート）の集合。"""
    path = os.path.join(INTERMEDIATE_DIR, f"labeled_{cap}.parquet")
    if not os.path.exists(path):
        return None
    lab = pd.read_parquet(path)
    det = lab["zeek_detailed"].astype(str).str.lower() if "zeek_detailed" in lab else pd.Series("", index=lab.index)
    hs = lab[det.str.contains("horizontalportscan", na=False)]
    return set(zip(hs["peer_ip"].astype(str), hs["peer_port"].astype(int)))


def evaluate_capture(cap, model, feats, allow_large):
    info = CAPTURES[cap]
    pcap = find_pcap(info["dir"])
    size_mb = os.path.getsize(pcap) / 1e6
    if size_mb > 200 and not allow_large:
        print(f"  [{cap}] {size_mb:,.0f}MB のためスキップ（--allow-large で実行）")
        return None
    targets = scan_targets(cap)
    tgt_txt = "ラベルなし→全ブロックを正常として扱う" if targets is None else f"{len(targets):,}件"
    print(f"  [{cap}] {size_mb:,.0f}MB 読み込み中…（Zeek上の水平スキャン宛先 {tgt_txt}）", flush=True)

    flows, _ = extract_flows_from_pcap(pcap)
    df = flows_to_dataframe(flows)
    host = info.get("host_ip")
    if targets:
        key = list(zip(df["Dst_IP"].astype(str), df["Dst_Port"].astype(int)))
        df["is_hscan"] = [(df.at[i, "Src_IP"] == host) and (k in targets) for i, k in zip(df.index, key)]
    else:
        df["is_hscan"] = False

    blocks = build_blocks(df, WINDOW)
    # ブロックの正解：build_blocks と同じ区切りで、水平スキャンのフローの割合を求める
    truth = []
    for src_ip, g in df.groupby("Src_IP"):
        for i in range(0, len(g), WINDOW):
            truth.append(g["is_hscan"].iloc[i:i + WINDOW].mean())
    blocks["hscan_share"] = truth
    blocks["truth_h"] = blocks["hscan_share"] >= 0.5
    blocks["capture"] = cap
    blocks["v_score"] = model.predict_proba(blocks[feats])[:, 1]
    blocks["pred_v"] = blocks["v_score"] >= VERTICAL_TH
    blocks["pred_h"] = horizontal_scan_flags(blocks)
    return blocks


def summarize(b):
    rows = []
    for cap, g in b.groupby("capture", sort=False):
        t = g["truth_h"]
        rows.append(dict(
            capture=cap, blocks=len(g), 水平スキャン正解=int(t.sum()),
            縦モデル検知=int((g["pred_v"] & t).sum()),
            水平ルール検知=int((g["pred_h"] & t).sum()),
            どちらか検知=int(((g["pred_v"] | g["pred_h"]) & t).sum()),
            水平ルール誤検知=int((g["pred_h"] & ~t).sum()),
            縦モデル警告計=int(g["pred_v"].sum()),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="+", default=list(CAPTURES.keys()))
    ap.add_argument("--allow-large", action="store_true")
    args = ap.parse_args()

    model, feats = load_vertical_model()
    print("=" * 96)
    print("エンジンA：水平スキャン検知の評価（正解＝Zeekの PartOfAHorizontalPortScan）")
    print("=" * 96)
    parts = [evaluate_capture(c, model, feats, args.allow_large) for c in args.captures if c in CAPTURES]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        print("評価できるキャプチャがありませんでした。")
        return
    b = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    b.to_csv(os.path.join(OUT_DIR, "blocks.csv"), index=False)

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\n--- キャプチャ別の結果（初期ルール：%s） ---" % H_RULE)
        print(summarize(b).to_string(index=False))

        t = b["truth_h"]
        print("\n--- 水平スキャン正解ブロック vs それ以外：追加特徴量の中央値 ---")
        print(b.groupby(t.map({True: "水平スキャン", False: "それ以外"}))[HORIZONTAL_FEATURES +
              ["Unique_Dst_Ports", "SYN_Flag_Ratio"]].median().round(3).to_string())

        print("\n--- しきい値の感度（全キャプチャ合計） ---")
        grid = []
        for ipr, ps, nr in itertools.product([0.6, 0.7, 0.8, 0.9], [0.6, 0.8, 0.9], [0.6, 0.8, 0.9]):
            pred = horizontal_scan_flags(b, ip_ratio=ipr, port_share=ps, no_return=nr)
            tp, fp, fn = int((pred & t).sum()), int((pred & ~t).sum()), int((~pred & t).sum())
            grid.append(dict(ip_ratio=ipr, port_share=ps, no_return=nr, TP=tp, FP=fp, FN=fn,
                             適合率=tp / (tp + fp) if tp + fp else np.nan,
                             再現率=tp / (tp + fn) if tp + fn else np.nan))
        g = pd.DataFrame(grid).sort_values(["FP", "再現率"], ascending=[True, False])
        print(g.head(12).round(3).to_string(index=False))
        fp_blocks = b[b["pred_h"] & ~t]
        if len(fp_blocks):
            print("\n--- 初期ルールの誤検知ブロック（上位10件） ---")
            print(fp_blocks[["capture", "Src_IP", "Top_Dst_Port"] + HORIZONTAL_FEATURES]
                  .head(10).round(3).to_string(index=False))
    print(f"\n[+] 保存しました: {OUT_DIR}/blocks.csv")


if __name__ == "__main__":
    main()
