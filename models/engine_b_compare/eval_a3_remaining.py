"""
eval_a3_remaining.py
仕様書v2.5 8.3節 A3-2 のうち、未出力の評価をまとめて出す。
models/engine_b_compare/ に配置して実行する。

  評価1：ブロック単位（stride 5 と、重ならない stride 20 の両方）
         適合率・再現率・F1・PR-AUC・FPR・混同行列
  評価2：通信ペア単位（C2ペア○本中○本を検知、正常ペア○本中○本を誤検知）
         Hakai 8-1 の2ペアは1回のビーコンで両方に接続しているため、1事例としても数える（1.6節）
  評価5：Torii（参考）… C2通信がペア内10フロー以上に届かず、判定対象外になることを確認

対象モデル：採用モデル（CatBoost・モデルB）。保存済みの models/engine_b/catboost_engine_b_v1.cbm を使う。

検算：stride 5 の結果が、これまでの比較結果
      （F1 0.792、PR-AUC 0.989、FPR 2.3%、34-1 S3 再現率 1.00、Hakai 再現率 0.00）と一致すること。

実行:
    python models/engine_b_compare/eval_a3_remaining.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (OUTPUT_DIR, FEATURE_COLS, load_features, load_metadata,  # noqa: E402
                        prepare_xy, conn_state_recall)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO, "models", "engine_b", "catboost_engine_b_v1.cbm")
EXPECTED_STRIDE5 = {"F1": 0.792, "PR_AUC": 0.989, "FPR": 0.023,
                    "Recall_34-1_S3": 1.00, "Recall_8-1": 0.00}
HAKAI_CAPTURE = "8-1"
TORII_CAPTURES = ["20-1", "21-1"]
TORII_C2_LABEL = "c&c-torii"
TORII_C2_NET = "66.85.157."          # 付録A.2：20-1／21-1 のC2サーバ 66.85.157.90:443
MIN_PAIR_FLOWS = 10                  # 3.6節：ペア内10フロー未満は Filtered


def load_model():
    from catboost import CatBoostClassifier
    with open(MODEL_PATH.replace(".cbm", "_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model, meta


def block_metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        n_blocks=len(y), n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        PR_AUC=average_precision_score(y, proba) if 0 < y.sum() < len(y) else np.nan,
        FPR=fp / (fp + tn) if (fp + tn) else np.nan,
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    ), pred


def main():
    model, meta = load_model()
    thr = float(meta.get("threshold", 0.5))
    feat_df = load_features()
    meta_df = load_metadata()

    # 学習時と同じ前処理（無限大→欠損→Trainの中央値で補完）
    _, _, Xte_all, yte, _, test = prepare_xy(feat_df, scale=False)
    idx = [FEATURE_COLS.index(c) for c in meta["features"]]
    proba = model.predict_proba(Xte_all[:, idx])[:, 1]

    test = test.reset_index(drop=True).copy()
    test["proba"] = proba
    test["y"] = yte
    parts = test["block_id"].str.rsplit("_", n=1)
    test["pair_id"] = parts.str[0]
    test["window_index"] = parts.str[1].astype(int)

    # ---------------- 評価1：ブロック単位 ----------------
    print("=" * 96)
    print("評価1：ブロック単位（Test）  ※stride 20 = 窓番号が4の倍数のブロックのみ（重なりなし）")
    print("=" * 96)
    rows = []
    for name, sub in [("stride 5（全ブロック）", test),
                      ("stride 20（重なりなし）", test[test["window_index"] % 4 == 0])]:
        m, pred = block_metrics(sub["y"].values, sub["proba"].values, thr)
        m.update(conn_state_recall(sub["block_id"].values, sub["y"].values, pred, meta_df))
        rows.append(dict(条件=name, **m))
    res1 = pd.DataFrame(rows)
    cols = ["条件", "n_blocks", "n_pos", "n_neg", "Precision", "Recall", "F1", "PR_AUC", "FPR",
            "TN", "FP", "FN", "TP", "Recall_34-1_S0", "Recall_34-1_S3", "Recall_8-1"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(res1[[c for c in cols if c in res1.columns]].round(3).to_string(index=False))

    print("\n--- 検算（stride 5 とこれまでの比較結果） ---")
    s5 = res1.iloc[0]
    all_ok = True
    for k, exp in EXPECTED_STRIDE5.items():
        got = s5.get(k)
        ok = got is not None and not pd.isna(got) and abs(float(got) - exp) < 0.0015
        all_ok &= ok
        print(f"  {k:16s}: 期待 {exp:.3f} / 今回 {float(got):.3f}  {'✅' if ok else '❌'}")
    print("  判定:", "✅ 一致（保存モデル・評価方法とも従来と同じ）" if all_ok
          else "❌ 不一致あり（保存モデルか評価条件が従来と違う可能性）")

    # ---------------- 評価2：通信ペア単位 ----------------
    print("\n" + "=" * 96)
    print("評価2：通信ペア単位（Test、stride 5）  ペアのどれか1ブロックでも警告なら「検知」")
    print("=" * 96)
    test["pred"] = (test["proba"] >= thr).astype(int)
    pair = test.groupby("pair_id").agg(
        capture=("capture", "first"), n_blocks=("y", "size"),
        c2_ratio=("y", "mean"), detected=("pred", "max"), max_proba=("proba", "max"),
    ).reset_index()
    pair["is_c2_pair"] = pair["c2_ratio"] >= 0.5

    c2p = pair[pair["is_c2_pair"]]
    bnp = pair[~pair["is_c2_pair"]]
    print(f"  C2ペア  : {len(c2p)} 本中 {int(c2p['detected'].sum())} 本を検知")
    print(f"  正常ペア: {len(bnp)} 本中 {int(bnp['detected'].sum())} 本を誤検知")

    hakai = c2p[c2p["capture"] == HAKAI_CAPTURE]
    non_hakai = c2p[c2p["capture"] != HAKAI_CAPTURE]
    n_events = len(non_hakai) + (1 if len(hakai) else 0)
    n_det = int(non_hakai["detected"].sum()) + (int(hakai["detected"].max()) if len(hakai) else 0)
    print(f"  事例単位（Hakaiの{len(hakai)}ペアを1事例として数える）: {n_events} 事例中 {n_det} 事例を検知")

    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print("\n--- C2ペアの内訳 ---")
        print(c2p[["pair_id", "capture", "n_blocks", "detected", "max_proba"]]
              .round(3).to_string(index=False))
        fp_pairs = bnp[bnp["detected"] == 1]
        print("\n--- 誤検知した正常ペア ---")
        print(fp_pairs[["pair_id", "capture", "n_blocks", "max_proba"]].round(3).to_string(index=False)
              if len(fp_pairs) else "  なし")
        print("\n--- 正常ペアのキャプチャ別内訳 ---")
        print(bnp.groupby("capture").agg(ペア数=("pair_id", "size"), 誤検知=("detected", "sum"))
              .to_string())

    # ---------------- 評価5：Torii（参考） ----------------
    print("\n" + "=" * 96)
    print("評価5：Torii（参考）  C2通信が判定対象（ペア内10フロー以上）に届くか")
    print("=" * 96)
    sys.path.insert(0, os.path.join(REPO, "src", "engine_b"))
    from config import INTERMEDIATE_DIR
    torii_rows = []
    for cap in TORII_CAPTURES:
        path = os.path.join(INTERMEDIATE_DIR, f"labeled_{cap}.parquet")
        if not os.path.exists(path):
            print(f"  {cap}: ファイルなし（{path}）")
            continue
        lab = pd.read_parquet(path)
        det = lab["zeek_detailed"].astype(str).str.lower() if "zeek_detailed" in lab else ""
        is_c2 = (det == TORII_C2_LABEL) | lab["peer_ip"].astype(str).str.startswith(TORII_C2_NET)
        c2 = lab[is_c2]
        keys = [k for k in ["peer_ip", "peer_port", "proto"] if k in c2.columns]
        if len(c2) == 0:
            print(f"  {cap}: C2フロー 0件")
            continue
        for key, g in c2.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            n = len(g)
            span_h = (g["start_ts"].max() - g["start_ts"].min()) / 3600 if "start_ts" in g else np.nan
            torii_rows.append(dict(capture=cap, pair=":".join(map(str, key)), c2_flows=n,
                                   span_hours=round(span_h, 1),
                                   判定=("判定対象外（Filtered）" if n < MIN_PAIR_FLOWS else "判定対象")))
    if torii_rows:
        tr = pd.DataFrame(torii_rows)
        print(tr.to_string(index=False))
        ok = (tr["c2_flows"] < MIN_PAIR_FLOWS).all()
        print("\n  判定:", f"✅ すべてのToriiのC2ペアが{MIN_PAIR_FLOWS}フロー未満で判定対象外" if ok
              else "⚠️ 判定対象に入るペアがある（上表を確認）")

    out = os.path.join(OUTPUT_DIR, "eval_a3_remaining")
    os.makedirs(out, exist_ok=True)
    res1.to_csv(os.path.join(out, "eval1_block_stride5_stride20.csv"), index=False)
    pair.to_csv(os.path.join(out, "eval2_pair_level.csv"), index=False)
    if torii_rows:
        pd.DataFrame(torii_rows).to_csv(os.path.join(out, "eval5_torii.csv"), index=False)
    print(f"\n[+] 結果を保存しました: {out}")


if __name__ == "__main__":
    main()
