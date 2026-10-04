"""
waterfall_s3.py
仕様書v2.5 8.3節 A3-4：34-1 の接続確立（S3）ブロックの SHAP Waterfall Plot を出力する。
対比のため、検知できなかった Hakai（8-1）のブロックと、誤検知した正常ブロックも同じ形式で出す。
models/engine_b_compare/ に配置して実行する。

対象モデル：採用モデル（CatBoost・モデルB）。保存済みの models/engine_b/catboost_engine_b_v1.cbm を使う。
代表ブロックの選び方：各グループで C2確率が中央値に最も近いブロック（極端な1件を選ばないため）。

出力（raw_pcap/engine_b_iot23/_output/waterfall_s3/）:
  waterfall_34-1_S3.png        … 本題。接続確立したC2を、何を根拠にC2と判定したか
  waterfall_8-1_hakai.png      … 対比。Hakaiを、何を根拠に正常と判定したか
  waterfall_false_positive.png … 対比。誤検知した正常ブロックで、何がC2寄りに押したか
  shap_s3_mean.csv             … S3ブロック全体での平均SHAP（代表1件が偏っていないかの確認用）

実行:
    python models/engine_b_compare/waterfall_s3.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import OUTPUT_DIR, FEATURE_COLS, load_features, load_metadata, prepare_xy  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO, "models", "engine_b", "catboost_engine_b_v1.cbm")
OUT_DIR = os.path.join(OUTPUT_DIR, "waterfall_s3")


def load_model():
    from catboost import CatBoostClassifier
    with open(MODEL_PATH.replace(".cbm", "_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model, meta


def pick_median(df):
    """C2確率が、そのグループの中央値に最も近いブロックを1件選ぶ。"""
    med = df["proba"].median()
    return df.iloc[(df["proba"] - med).abs().argsort().iloc[0]]


def save_waterfall(shap_mod, explanation, title, path):
    plt.figure()
    shap_mod.plots.waterfall(explanation, max_display=len(explanation.values), show=False)
    plt.title(title, fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    import shap

    os.makedirs(OUT_DIR, exist_ok=True)
    model, meta = load_model()
    cols = meta["features"]
    thr = float(meta.get("threshold", 0.5))

    feat_df = load_features()
    meta_df = load_metadata()[["block_id", "conn_state_major"]].drop_duplicates("block_id")
    _, _, Xte_all, yte, _, test = prepare_xy(feat_df, scale=False)
    idx = [FEATURE_COLS.index(c) for c in cols]
    X = Xte_all[:, idx]

    test = test.reset_index(drop=True).copy()
    test["row"] = np.arange(len(test))
    test["y"] = yte
    test["proba"] = model.predict_proba(X)[:, 1]
    test = test.merge(meta_df, on="block_id", how="left")

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])

    def explanation(row):
        return shap.Explanation(values=sv[row], base_values=base, data=X[row], feature_names=cols)

    groups = {
        "34-1_S3": test[(test["capture"] == "34-1") & (test["y"] == 1) & (test["conn_state_major"] == "S3")],
        "8-1_hakai": test[(test["capture"] == "8-1") & (test["y"] == 1)],
        "false_positive": test[(test["y"] == 0) & (test["proba"] >= thr)],
    }
    titles = {
        "34-1_S3": "34-1 Mirai, connected C2 (S3): why judged C2",
        "8-1_hakai": "8-1 Hakai (unseen family): why judged normal",
        "false_positive": "False positive (normal block): what pushed toward C2",
    }
    names = {"34-1_S3": "waterfall_34-1_S3.png", "8-1_hakai": "waterfall_8-1_hakai.png",
             "false_positive": "waterfall_false_positive.png"}

    print("=" * 90)
    print("A3-4：SHAP Waterfall Plot（採用モデル CatBoost・モデルB）")
    print("  ※SHAP値の単位は対数オッズ（log-odds）。基準値＋各特徴量の寄与＝ブロックの判定スコア")
    print(f"  基準値（全体平均の出力）: {base:.3f}")
    print("=" * 90)
    for key, g in groups.items():
        if len(g) == 0:
            print(f"\n[{key}] 該当ブロックなし")
            continue
        r = pick_median(g)
        row = int(r["row"])
        save_waterfall(shap, explanation(row), titles[key], os.path.join(OUT_DIR, names[key]))
        print(f"\n[{key}] 対象 {len(g)} ブロック／代表: {r['block_id']}  C2確率 {r['proba']:.3f}")
        contrib = pd.DataFrame({"feature": cols, "value": X[row], "shap": sv[row]})
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
        print(contrib.round(3).to_string(index=False))

    s3 = groups["34-1_S3"]
    if len(s3):
        rows = s3["row"].values
        mean = pd.DataFrame({
            "feature": cols,
            "mean_shap": sv[rows].mean(axis=0),
            "mean_abs_shap": np.abs(sv[rows]).mean(axis=0),
            "share_of_pushes_to_c2": (sv[rows] > 0).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        mean.to_csv(os.path.join(OUT_DIR, "shap_s3_mean.csv"), index=False)
        print(f"\n--- 34-1 S3 全{len(s3)}ブロックの平均（代表1件が偏っていないかの確認） ---")
        print(mean.round(3).to_string(index=False))

    print(f"\n[+] 出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
