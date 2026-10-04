"""
inference.py
エンジンBの推論処理。正解ラベル（Zeek）を使わずに、PCAP 1本から
「C2ビーコンの疑いがある通信ペア」を判定する。src/engine_b/ に配置する。

【学習時との対応（ずれを防ぐため、特徴量計算は学習時の関数をそのまま呼ぶ）】
  段階1 パケット抽出   : stage1_extract.run_tshark / normalize_chunk を使用
  段階2 Biflow生成     : stage2_biflow.build_biflows を使用（作業フォルダだけ差し替え）
  段階3 ラベル照合     : 行わない。社外通信（North-South）への絞り込みのみ行う
  段階4 ブロック化     : 学習時と同じ窓（20フロー・5フローずらし・10フロー未満のペアは除外）
                          特徴量は stage4_features.compute_features を使用
  段階5 判定           : 保存済みCatBoost（モデルB）で判定し、Wireshark用フィルタを作る

【学習時との違い（仕様上の違い。結果を読むときの注意）】
  - 学習時は感染端末を1台に固定していたが、推論では社内側のIPごとにペアを分ける
    （通信ペア = 社内IP・社外IP・社外側ポート・プロトコル）
  - 学習時はスキャン等のラベルが付いたフローを除外してからブロック化したが、
    推論ではラベルがないため、同じペアの全フローを使う

使い方（コマンドライン）:
    python src/engine_b/inference.py <PCAPファイル> [--threshold 0.5] [--out-dir 出力先]
アプリからの使い方:
    from inference import analyze_pcap
    summary, blocks, stats = analyze_pcap("xxx.pcap")
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json
import shutil
import argparse
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import pandas as pd

import stage1_extract
import stage2_biflow
from stage4_features import compute_features
from config import WINDOW_SIZE, WINDOW_STRIDE, MIN_PAIR_FLOWS
from common import is_external

REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "engine_b", "catboost_engine_b_v1.cbm")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "inference_results")
PAIR_KEYS = ["host_ip", "peer_ip", "peer_port", "proto"]


# ---------------------------------------------------------------- 段階1・2
def extract_packets(pcap_path, work_dir):
    """段階1と同じ処理（tshark → 正規化 → Parquet）。パケット数を返す。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tsv_path = os.path.join(work_dir, "packets_infer.tsv")
    out_path = os.path.join(work_dir, "packets_infer.parquet")
    stage1_extract.run_tshark(pcap_path, tsv_path)

    writer, n_rows = None, 0
    reader = pd.read_csv(tsv_path, sep="\t", dtype=str, chunksize=500_000, low_memory=False)
    for chunk in reader:
        chunk = stage1_extract.normalize_chunk(chunk)
        if chunk.empty:
            continue
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_rows += len(chunk)
    if writer is not None:
        writer.close()
    os.remove(tsv_path)
    return n_rows


def build_biflows(work_dir):
    """段階2の関数をそのまま使う。読み書き先だけ作業フォルダに一時的に差し替える。"""
    original = stage2_biflow.INTERMEDIATE_DIR
    stage2_biflow.INTERMEDIATE_DIR = work_dir
    try:
        return stage2_biflow.build_biflows("infer")
    finally:
        stage2_biflow.INTERMEDIATE_DIR = original


# ---------------------------------------------------------------- 段階3の代わり
def to_north_south(biflows):
    """片側が社内・もう片側が社外のフローだけ残し、社内側をhost、社外側をpeerにする。"""
    if biflows is None or len(biflows) == 0:
        return pd.DataFrame(columns=PAIR_KEYS)
    ips = pd.unique(pd.concat([biflows["src_ip"], biflows["dst_ip"]]))
    ext = {ip: bool(is_external(ip)) for ip in ips}
    src_ext = biflows["src_ip"].map(ext).values
    dst_ext = biflows["dst_ip"].map(ext).values
    keep = src_ext != dst_ext
    b = biflows[keep].copy()
    se = src_ext[keep]
    b["host_ip"] = np.where(se, b["dst_ip"], b["src_ip"])
    b["peer_ip"] = np.where(se, b["src_ip"], b["dst_ip"])
    b["peer_port"] = np.where(se, b["src_port"], b["dst_port"]).astype(int)
    return b.reset_index(drop=True)


# ---------------------------------------------------------------- 段階4
def build_blocks(flows):
    """学習時（stage4）と同じ窓の切り方で、通信ペアごとにブロックを作り特徴量を計算する。"""
    rows = []
    if len(flows) == 0:
        return pd.DataFrame()
    for key, g in flows.groupby(PAIR_KEYS, sort=True):
        g = g.sort_values("start_ts").reset_index(drop=True)
        if len(g) < MIN_PAIR_FLOWS:
            continue
        n_windows = max(0, (len(g) - WINDOW_SIZE) // WINDOW_STRIDE + 1)
        for w in range(n_windows):
            i0 = w * WINDOW_STRIDE
            win = g.iloc[i0:i0 + WINDOW_SIZE]
            if len(win) < WINDOW_SIZE:
                continue
            feats = compute_features(win)
            feats.update(
                host_ip=key[0], peer_ip=key[1], peer_port=int(key[2]), proto=key[3],
                window_index=w,
                first_seen=float(win["start_ts"].min()), last_seen=float(win["end_ts"].max()),
            )
            rows.append(feats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 段階5
def load_model(model_path=DEFAULT_MODEL):
    from catboost import CatBoostClassifier

    meta_path = model_path.replace(".cbm", "_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if "fill_medians" not in meta:
        raise RuntimeError(
            "メタ情報に fill_medians（欠損値の補完値）がありません。"
            "更新版の train_final_model.py を実行してモデルを保存し直してください。")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model, meta


def predict_blocks(blocks, model, meta, threshold=None):
    """学習時（prepare_xy）と同じ前処理（無限大→欠損→Trainの中央値で補完）をして判定する。"""
    thr = meta.get("threshold", 0.5) if threshold is None else threshold
    out = blocks.copy()
    if len(out) == 0:
        out["c2_proba"] = []
        out["is_alert"] = []
        return out
    cols = meta["features"]
    X = out[cols].replace([np.inf, -np.inf], np.nan).fillna(pd.Series(meta["fill_medians"]))
    out["c2_proba"] = model.predict_proba(X[cols].values)[:, 1]
    out["is_alert"] = out["c2_proba"] >= thr
    return out


def wireshark_filter(host_ip, peer_ip, peer_port, proto):
    return f"ip.addr=={host_ip} && ip.addr=={peer_ip} && {proto}.port=={peer_port}"


def summarize(blocks):
    """ブロック単位の判定を、通信ペア単位にまとめる（アプリ表示用）。"""
    if len(blocks) == 0:
        return pd.DataFrame()
    g = blocks.groupby(PAIR_KEYS, sort=False)
    s = g.agg(
        n_blocks=("c2_proba", "size"),
        n_alert_blocks=("is_alert", "sum"),
        max_c2_proba=("c2_proba", "max"),
        mean_c2_proba=("c2_proba", "mean"),
        iat_median_sec=("IAT_Median", "median"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
    ).reset_index()
    s["alert_ratio"] = s["n_alert_blocks"] / s["n_blocks"]
    s["is_alert"] = s["n_alert_blocks"] > 0
    s["first_seen"] = pd.to_datetime(s["first_seen"], unit="s")
    s["last_seen"] = pd.to_datetime(s["last_seen"], unit="s")
    s["wireshark_filter"] = [
        wireshark_filter(r.host_ip, r.peer_ip, r.peer_port, r.proto) for r in s.itertuples()]
    return s.sort_values(["is_alert", "max_c2_proba"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- まとめて実行
def analyze_pcap(pcap_path, model_path=DEFAULT_MODEL, threshold=None, keep_work_dir=False):
    """PCAP 1本を判定する。戻り値: (ペア単位の要約, ブロック単位の結果, 件数の記録)"""
    model, meta = load_model(model_path)
    work_dir = tempfile.mkdtemp(prefix="engine_b_infer_")
    stats = dict(pcap=os.path.basename(pcap_path))
    try:
        stats["n_packets"] = extract_packets(pcap_path, work_dir)
        if stats["n_packets"] == 0:
            return pd.DataFrame(), pd.DataFrame(), stats
        biflows = build_biflows(work_dir)
        stats["n_biflows"] = 0 if biflows is None else len(biflows)
        flows = to_north_south(biflows)
        stats["n_north_south_flows"] = len(flows)
        sizes = flows.groupby(PAIR_KEYS).size() if len(flows) else []
        stats["n_filtered_pairs"] = int((sizes < MIN_PAIR_FLOWS).sum()) if len(flows) else 0
        blocks = build_blocks(flows)
        stats["n_pairs"] = 0 if len(blocks) == 0 else int(blocks.groupby(PAIR_KEYS).ngroups)
        stats["n_blocks"] = len(blocks)
        blocks = predict_blocks(blocks, model, meta, threshold)
        summary = summarize(blocks)
        stats["n_alert_pairs"] = 0 if len(summary) == 0 else int(summary["is_alert"].sum())
        stats["threshold"] = meta.get("threshold", 0.5) if threshold is None else threshold
        return summary, blocks, stats
    finally:
        if keep_work_dir:
            stats["work_dir"] = work_dir
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="エンジンB：PCAPからC2ビーコンの疑いがある通信を判定する")
    ap.add_argument("pcap")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    summary, blocks, stats = analyze_pcap(args.pcap, args.model, args.threshold)

    print("\n" + "=" * 90)
    print("エンジンB 推論結果")
    print("=" * 90)
    for k, v in stats.items():
        print(f"  {k:22s}: {v:,}" if isinstance(v, int) else f"  {k:22s}: {v}")
    if len(summary):
        cols = ["host_ip", "peer_ip", "peer_port", "proto", "n_blocks", "n_alert_blocks",
                "max_c2_proba", "iat_median_sec", "wireshark_filter"]
        print("\n--- 判定結果（疑いの強い順・上位20ペア） ---")
        with pd.option_context("display.width", 220, "display.max_columns", None,
                               "display.max_colwidth", 80):
            print(summary[cols].head(20).round(3).to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pcap))[0]
    summary.to_csv(os.path.join(args.out_dir, f"{base}_engine_b_pairs.csv"), index=False)
    blocks.to_csv(os.path.join(args.out_dir, f"{base}_engine_b_blocks.csv"), index=False)
    print(f"\n[+] 結果を保存しました: {args.out_dir}")


if __name__ == "__main__":
    main()
