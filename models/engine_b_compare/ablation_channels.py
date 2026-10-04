"""
ablation_channels.py
B系統（時系列DL：CNN / GRU / TCN）の入力チャンネルを変えて性能を比べる
アブレーション検証。models/engine_b_compare/ に配置して実行する。

【目的】CNN/GRU/TCN が約99.7%の性能を出したが、何を根拠に判定しているかを確かめる。
  - 送信サイズが毎回一定、という「近道」で当てていないか
  - 「間隔が短い＝C2」という近道（Mirai再接続は数秒、NTPは数十秒〜数分）で当てていないか

【比較する4条件】
  all       : IAT・送信バイト・受信バイト（現行。基準）
  iat       : IATのみ（通信間隔だけで判定できるか）
  bytes     : 送信バイト・受信バイトのみ（サイズだけで判定できてしまうか）
  iat_rel   : IATをブロックごとに自身の中央値で割ったもの（間隔の「長さ」を消し、
              「規則正しさ」だけを残す。これで高性能なら、間隔の長さではなく
              周期性で判定している根拠になる）

各条件 × 3モデル × 3シード（42,43,44）で学習し、平均と標準偏差を出す。
dl_models.py は変更しない（入力チャンネル数だけ、この中で差し替える）。

実行:
    python models/engine_b_compare/ablation_channels.py
    python models/engine_b_compare/ablation_channels.py --epochs 50 --seeds 42 43 44 45 46
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import functools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import OUTPUT_DIR, load_timeseries, load_metadata, conn_state_recall  # noqa: E402
import dl_models  # noqa: E402

CH_INDEX = {"iat": 0, "fwd_bytes": 1, "bwd_bytes": 2}


def make_input(X, condition):
    """X: (N, 3, 20)  -> 条件に応じた (N, C, 20)"""
    if condition == "all":
        return X
    if condition == "iat":
        return X[:, [CH_INDEX["iat"]], :]
    if condition == "bytes":
        return X[:, [CH_INDEX["fwd_bytes"], CH_INDEX["bwd_bytes"]], :]
    if condition == "iat_rel":
        iat = X[:, CH_INDEX["iat"], :].astype(np.float64)
        # 先頭は常に0（前のフローがない）ため、中央値は2番目以降で計算する
        med = np.median(iat[:, 1:], axis=1, keepdims=True)
        rel = iat / (med + 1e-3)
        rel = np.clip(rel, 0.0, 50.0)   # 中央値が極小のブロックで値が発散するのを防ぐ
        return rel[:, None, :]
    raise ValueError(condition)


def run(condition, model_name, ts, meta_df, seed, epochs, threshold=0.5):
    tr = ts["split"] == "train"
    te = ts["split"] == "test"
    Xall = make_input(ts["X"], condition)
    Xtr, ytr = Xall[tr], ts["y"][tr]
    Xte, yte = Xall[te], ts["y"][te]

    # dl_models の MODEL_CLASSES を、入力チャンネル数を合わせたものに一時的に差し替える
    orig = dl_models.MODEL_CLASSES[model_name]
    dl_models.MODEL_CLASSES[model_name] = functools.partial(orig, in_channels=Xtr.shape[1])
    try:
        _, proba, train_time, _ = dl_models.train_torch_model(
            model_name, Xtr, ytr, Xte, epochs=epochs, seed=seed,
        )
    finally:
        dl_models.MODEL_CLASSES[model_name] = orig

    y_pred = (proba >= threshold).astype(int)
    neg = yte == 0
    row = dict(
        condition=condition, model=model_name, seed=seed,
        n_channels=int(Xtr.shape[1]),
        F1=f1_score(yte, y_pred),
        PR_AUC=average_precision_score(yte, proba),
        FPR=float((y_pred[neg] == 1).mean()) if neg.any() else None,  # 正常を誤ってC2と判定した割合
        train_time_sec=train_time,
    )
    row.update(conn_state_recall(ts["block_id"][te], yte, y_pred, meta_df))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--conditions", nargs="+", default=["all", "iat", "bytes", "iat_rel"])
    ap.add_argument("--models", nargs="+", default=["cnn1d", "gru", "tcn"])
    args = ap.parse_args()

    if not dl_models.TORCH_AVAILABLE:
        print("torch が見つかりません。pip install torch を実行してください。")
        sys.exit(1)

    ts = load_timeseries()
    meta_df = load_metadata()

    rows = []
    total = len(args.conditions) * len(args.models) * len(args.seeds)
    k = 0
    for cond in args.conditions:
        for m in args.models:
            for s in args.seeds:
                k += 1
                print(f"[{k}/{total}] 条件={cond:8s} モデル={m:6s} seed={s}", flush=True)
                rows.append(run(cond, m, ts, meta_df, s, args.epochs))

    df = pd.DataFrame(rows)
    raw_path = os.path.join(OUTPUT_DIR, "ablation_channels_raw.csv")
    df.to_csv(raw_path, index=False)

    metrics = ["F1", "PR_AUC", "FPR", "Recall_34-1_S0", "Recall_34-1_S3", "Recall_8-1"]
    agg = df.groupby(["condition", "model"], sort=False)[metrics].agg(["mean", "std"]).round(3)
    sum_path = os.path.join(OUTPUT_DIR, "ablation_channels_summary.csv")
    agg.to_csv(sum_path)

    mean_tbl = df.groupby(["condition", "model"], sort=False)[metrics].mean().round(3)
    print("\n" + "=" * 100)
    print(f"チャンネル・アブレーション結果（{len(args.seeds)}シードの平均）")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(mean_tbl.to_string())
    print("\n[読み方]")
    print("  bytes が all と同等に高い  → サイズだけで判定できてしまう（近道の疑い）")
    print("  iat が高く bytes が低い    → 通信間隔が判定の主な根拠")
    print("  iat_rel も高い            → 間隔の『長さ』でなく『規則正しさ』で判定している根拠")
    print("  iat_rel だけ大きく下がる   → 『間隔が短い＝C2』という近道の疑い")
    print(f"\n[+] 生データ : {raw_path}")
    print(f"[+] 集計表   : {sum_path}（平均と標準偏差）")


if __name__ == "__main__":
    main()
