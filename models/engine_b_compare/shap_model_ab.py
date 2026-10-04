"""
shap_model_ab.py
採用候補（CatBoost / Random Forest）と、比較対象として近道型のLightGBMについて、仕様書v2.5 4.1節の
「モデルA（8特徴量すべて）」と「モデルB（タイミング系5特徴量のみ）」を比較し、
SHAPで判定根拠を可視化する。models/engine_b_compare/ に配置して実行する。

【確かめたいこと】
  1. バイト系の特徴量（Burst_Active_Ratio / Payload_Bytes_MAD / Byte_Ratio_Median）を
     外しても、34-1のS3（接続確立したC2）の再現率が維持されるか
  2. SHAPの上位がIAT系（通信間隔）の特徴量になっているか
  → 両方を満たせば「採用モデルは通信間隔で判定している」と示せる

【出力（raw_pcap/engine_b_iot23/_output/shap_model_ab/ 以下）】
  model_ab_results.csv       : 4通り（2モデル×A/B）の評価指標
  shap_importance.csv        : 特徴量ごとの平均|SHAP|（Test全体 / 34-1のS3ブロック）
  shap_<model>_<A|B>.png     : SHAP summary plot（スライド貼り付け用）
  shap_bar_<model>_<A|B>.png : 平均|SHAP|の棒グラフ

実行:
    python models/engine_b_compare/shap_model_ab.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (  # noqa: E402
    OUTPUT_DIR, FEATURE_COLS, load_features, load_metadata, conn_state_recall, prepare_xy,
)

TIMING_COLS = ["IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio", "Connection_Count"]
FEATURE_SETS = {"A": FEATURE_COLS, "B": TIMING_COLS}
# SHAPはTest全件（約2,100ブロック）で計算する（偏りのないグラフにするため）
MODELS = ["catboost", "random_forest", "lightgbm"]   # lightgbm は「近道型」の比較用
OUT_DIR = os.path.join(OUTPUT_DIR, "shap_model_ab")


def build_model(name):
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                                  loss_function="Logloss", random_seed=42,
                                  verbose=0, thread_count=1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                  random_state=42, n_jobs=1, verbosity=-1)
    raise ValueError(name)


def positive_class_shap(sv):
    """SHAPの戻り値の形がライブラリ・バージョンで異なるため、陽性クラス(C2)分に揃える"""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def main():
    import shap

    os.makedirs(OUT_DIR, exist_ok=True)
    feat_df = load_features()
    meta_df = load_metadata()
    Xtr_all, ytr, Xte_all, yte, train, test = prepare_xy(feat_df, scale=False)

    test_meta = test[["block_id"]].merge(
        meta_df[["block_id", "capture", "conn_state_major"]], on="block_id", how="left")
    s3_mask = ((yte == 1) & (test_meta["capture"].values == "34-1")
               & (test_meta["conn_state_major"].values == "S3"))

    rows, imp_rows = [], []
    for mname in MODELS:
        for set_name, cols in FEATURE_SETS.items():
            idx = [FEATURE_COLS.index(c) for c in cols]
            Xtr, Xte = Xtr_all[:, idx], Xte_all[:, idx]
            tag = f"{mname}_{set_name}"
            print(f"[*] 学習中: {mname} / モデル{set_name}（{len(cols)}特徴量）", flush=True)
            try:
                model = build_model(mname)
            except ImportError as e:
                print(f"   -> スキップ（ライブラリ未導入: {e.name}）")
                continue
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            neg = yte == 0
            row = dict(model=mname, feature_set=set_name, n_features=len(cols),
                       F1=f1_score(yte, y_pred), PR_AUC=average_precision_score(yte, proba),
                       FPR=float((y_pred[neg] == 1).mean()))
            row.update(conn_state_recall(test["block_id"].values, yte, y_pred, meta_df))
            rows.append(row)

            # ---- SHAP（Test）----
            print("   -> SHAP計算中...", flush=True)
            pick = np.arange(len(Xte))
            X_shap = pd.DataFrame(Xte[pick], columns=cols)
            sv = positive_class_shap(shap.TreeExplainer(model).shap_values(X_shap))

            plt.figure()
            shap.summary_plot(sv, X_shap, show=False, max_display=len(cols))
            plt.title(f"{mname} / Model {set_name} ({len(cols)} features)", fontsize=12)
            plt.savefig(os.path.join(OUT_DIR, f"shap_{tag}.png"), dpi=150, bbox_inches="tight")
            plt.close("all")

            plt.figure()
            shap.summary_plot(sv, X_shap, plot_type="bar", show=False, max_display=len(cols))
            plt.title(f"{mname} / Model {set_name}: mean |SHAP|", fontsize=12)
            plt.savefig(os.path.join(OUT_DIR, f"shap_bar_{tag}.png"), dpi=150, bbox_inches="tight")
            plt.close("all")

            s3_in_pick = s3_mask[pick]
            for j, c in enumerate(cols):
                imp_rows.append(dict(
                    model=mname, feature_set=set_name, feature=c,
                    mean_abs_shap_test=float(np.abs(sv[:, j]).mean()),
                    mean_abs_shap_341_S3=float(np.abs(sv[s3_in_pick, j]).mean()) if s3_in_pick.any() else None,
                ))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "model_ab_results.csv"), index=False)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False)

    print("\n" + "=" * 90)
    print("モデルA（8特徴量） vs モデルB（タイミング系5特徴量）")
    print("=" * 90)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res.round(3).to_string(index=False))
        print("\n--- 平均|SHAP|の順位（Test全体 / 34-1のS3ブロックのみ） ---")
        for (m, s), g in imp.groupby(["model", "feature_set"], sort=False):
            g = g.sort_values("mean_abs_shap_test", ascending=False)
            print(f"\n[{m} / モデル{s}]")
            print(g[["feature", "mean_abs_shap_test", "mean_abs_shap_341_S3"]].round(4).to_string(index=False))
    print("\n[読み方]")
    print("  モデルBのS3がモデルAと同等 → バイト系を外しても通信間隔だけで判定できている")
    print("  SHAP上位がIAT系          → 判定根拠が通信間隔（周期の長さ・ばらつき）であることの裏付け")
    print("  モデルAのSHAP上位がByte_Ratio_Median等 → 応答の有無に頼っている疑い")
    print("  LightGBM（近道型）とCatBoost（タイミング型）でSHAP上位が違えば、判定の型の違いを根拠付きで説明できる")
    print(f"\n[+] 出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
