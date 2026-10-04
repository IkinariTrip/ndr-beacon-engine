"""
stage4_features.py
ラベル付きBiflowから、通信ペア単位の時系列ブロックを作り、
以下の3種類の出力を作る（仕様書v2.5 3.6節・第4章準拠）。

  1. dataset_engine_b_features.csv   … 8特徴量テーブル（A系統モデル用）
  2. dataset_engine_b_metadata.csv   … 調査用メタデータ（block_idで結合可能）
  3. dataset_engine_b_timeseries.npz … 生の時系列データ（B系統DLモデル用）

【生の時系列データについて（2026-09-30 合意事項）】
  ブロック内20フロー分の「IAT」「送信バイト」「受信バイト」の3系列のみを書き出す。
  宛先ポート・サービス名などの識別子は、近道学習（ポート番号の暗記）につながる
  ため含めない。IATは先頭を0とし、全系列を長さ20に揃える。

出力: build_log.csv にも、ペア数・ブロック数・除外件数を記録する（3.7節）。
"""
import os
import sys

import numpy as np
import pandas as pd

from config import CAPTURES, INTERMEDIATE_DIR, OUTPUT_DIR, WINDOW_SIZE, WINDOW_STRIDE, MIN_PAIR_FLOWS


def autocorr_peak(deltas):
    x = np.asarray(deltas, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        # 間隔が完全に一定（分散ゼロ）の場合。Jitter 0%の低速ビーコン等で起こりうる。
        # 自己相関は定義上 0/0 になるが、これは「最も強い周期性」を意味するため、
        # NaNではなく最大値(1.0)として扱う。
        return 1.0
    best = 0.0
    max_lag = min(n - 1, 10)
    for lag in range(1, max_lag + 1):
        num = np.dot(x[:-lag], x[lag:])
        best = max(best, num / denom)
    return float(best)


def compute_features(win):
    start = win["start_ts"].values
    end = win["end_ts"].values
    dur = win["duration"].values
    fwd_b = win["fwd_bytes"].values
    bwd_b = win["bwd_bytes"].values

    deltas = np.diff(start)
    if len(deltas) == 0:
        iat_median = iat_mad = iat_cv = np.nan
    else:
        iat_median = float(np.median(deltas))
        iat_mad = float(np.median(np.abs(deltas - iat_median)))
        m = deltas.mean()
        iat_cv = float(deltas.std() / m) if m > 0 else 0.0

    peak_ratio = autocorr_peak(deltas)

    span = end.max() - start.min()
    burst_ratio = float(min(1.0, dur.sum() / span)) if span > 0 else np.nan

    total = fwd_b + bwd_b
    payload_mad = float(np.median(np.abs(fwd_b - np.median(fwd_b))))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, fwd_b / total, 1.0)
    byte_ratio_median = float(np.median(ratio))

    span_hours = span / 3600.0
    conn_count = float(len(win) / span_hours) if span_hours > 0 else np.nan

    return dict(
        IAT_Median=iat_median, IAT_MAD=iat_mad, IAT_CV=iat_cv,
        Periodic_Peak_Ratio=peak_ratio, Burst_Active_Ratio=burst_ratio,
        Payload_Bytes_MAD=payload_mad, Byte_Ratio_Median=byte_ratio_median,
        Connection_Count=conn_count,
    )


def compute_raw_series(win, window_size):
    """B系統DLモデル用の生の時系列（IAT・送信バイト・受信バイト）を作る。
    IP・ポート・サービス名などの識別子は含めない（近道学習対策）。
    """
    start = win["start_ts"].values
    iat = np.zeros(window_size, dtype=np.float32)
    if len(start) > 1:
        iat[1:len(start)] = np.diff(start)
    fwd = np.zeros(window_size, dtype=np.float32)
    bwd = np.zeros(window_size, dtype=np.float32)
    fwd[:len(win)] = win["fwd_bytes"].values
    bwd[:len(win)] = win["bwd_bytes"].values
    return iat, fwd, bwd


def build_blocks_for_capture(name, log_rows):
    cap = CAPTURES[name]
    path = os.path.join(INTERMEDIATE_DIR, f"labeled_{name}.parquet")
    df = pd.read_parquet(path)
    n_total = len(df)
    if n_total == 0:
        log_rows.append(dict(capture=name, split=cap["split"], n_biflows_total=0))
        return pd.DataFrame(), pd.DataFrame(), []

    kept = df[df["label_final"].isin([0, 1])].copy()
    kept["label_final"] = kept["label_final"].astype(int)
    kept["pair_key"] = list(zip(kept["peer_ip"], kept["peer_port"], kept["proto"]))

    feat_rows, meta_rows, raw_rows = [], [], []
    n_pairs_total = 0
    n_pairs_used = 0
    n_blocks_mixed_dropped = 0

    for pair_key, g in kept.groupby("pair_key"):
        n_pairs_total += 1
        g = g.sort_values("start_ts").reset_index(drop=True)
        if len(g) < MIN_PAIR_FLOWS:
            continue
        n_pairs_used += 1

        n_windows = max(0, (len(g) - WINDOW_SIZE) // WINDOW_STRIDE + 1)
        for w in range(n_windows):
            i0 = w * WINDOW_STRIDE
            win = g.iloc[i0:i0 + WINDOW_SIZE]
            if len(win) < WINDOW_SIZE:
                continue

            c2_ratio = float(win["label_final"].mean())
            if c2_ratio == 0:
                block_label = 0
            elif c2_ratio >= 0.5:
                block_label = 1
            else:
                n_blocks_mixed_dropped += 1
                continue

            block_id = f"{name}_{pair_key[0]}_{pair_key[1]}_{pair_key[2]}_{w}"
            feats = compute_features(win)
            feats.update(
                block_id=block_id, label=block_label, capture=name,
                split=cap["split"], window_type="w20_s5",
            )
            feat_rows.append(feats)

            iat, fwd, bwd = compute_raw_series(win, WINDOW_SIZE)
            raw_rows.append(dict(
                block_id=block_id, label=block_label, split=cap["split"], capture=name,
                iat=iat, fwd_bytes=fwd, bwd_bytes=bwd,
            ))

            if win["conn_state"].notna().any():
                conn_state_major = win["conn_state"].mode().iloc[0]
            else:
                conn_state_major = None
            meta_rows.append(dict(
                block_id=block_id, Src_IP=cap["host_ip"], Dst_IP=pair_key[0],
                Dst_Port=pair_key[1], Protocol=pair_key[2],
                First_Seen=win["start_ts"].min(), Last_Seen=win["end_ts"].max(),
                conn_state_major=conn_state_major, c2_ratio=c2_ratio, capture=name,
            ))

    log_rows.append(dict(
        capture=name, split=cap["split"], n_biflows_total=n_total,
        n_matched=int(df["label_final"].notna().sum()) + int(df["exclude_reason"].notna().sum()),
        n_label1=int((df["label_final"] == 1).sum()),
        n_label0=int((df["label_final"] == 0).sum()),
        n_pairs_total=n_pairs_total, n_pairs_used=n_pairs_used,
        n_blocks=len(feat_rows), n_blocks_mixed_dropped=n_blocks_mixed_dropped,
    ))
    for reason, cnt in df["exclude_reason"].value_counts(dropna=True).items():
        log_rows.append(dict(capture=name, split=cap["split"], exclude_reason=str(reason), count=int(cnt)))

    return pd.DataFrame(feat_rows), pd.DataFrame(meta_rows), raw_rows


def save_timeseries_npz(all_raw_rows, out_path):
    if not all_raw_rows:
        print("[!] 生の時系列データが0件のため、npzは作成しません")
        return
    block_id = np.array([r["block_id"] for r in all_raw_rows], dtype=object)
    label = np.array([r["label"] for r in all_raw_rows], dtype=np.int64)
    split = np.array([r["split"] for r in all_raw_rows], dtype=object)
    capture = np.array([r["capture"] for r in all_raw_rows], dtype=object)
    iat = np.stack([r["iat"] for r in all_raw_rows])
    fwd_bytes = np.stack([r["fwd_bytes"] for r in all_raw_rows])
    bwd_bytes = np.stack([r["bwd_bytes"] for r in all_raw_rows])

    np.savez_compressed(
        out_path, block_id=block_id, label=label, split=split, capture=capture,
        iat=iat, fwd_bytes=fwd_bytes, bwd_bytes=bwd_bytes,
    )
    print(f"[+] 完了: {out_path}  形状={iat.shape}（ブロック数 × 窓長）")


def main(targets):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_feat, all_meta, all_raw, log_rows = [], [], [], []
    for name in targets:
        print(f"\n===== 段階4（ブロック生成）: {name} =====")
        f, m, r = build_blocks_for_capture(name, log_rows)
        if len(f):
            print(f"  ブロック数: {len(f):,}  "
                  f"(Label=1: {int((f['label']==1).sum())}, Label=0: {int((f['label']==0).sum())})")
        else:
            print("  ブロック数: 0")
        all_feat.append(f)
        all_meta.append(m)
        all_raw.extend(r)

    feat_df = pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()
    meta_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    log_df = pd.DataFrame(log_rows)

    feat_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_engine_b_features.csv"), index=False)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_engine_b_metadata.csv"), index=False)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "build_log.csv"), index=False)
    save_timeseries_npz(all_raw, os.path.join(OUTPUT_DIR, "dataset_engine_b_timeseries.npz"))

    print("\n===== 区分×ラベル別ブロック数 =====")
    if len(feat_df):
        print(feat_df.groupby(["split", "label"]).size())
    else:
        print("（ブロックは1件も作成されませんでした）")
    print(f"\n[+] 出力完了: {OUTPUT_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = list(CAPTURES) if "--all" in args else [a for a in args if not a.startswith("--")]
    if not targets:
        print("使い方: python stage4_features.py <capture名> [<capture名> ...] | --all")
        sys.exit(1)
    main(targets)
