"""
data_utils.py
compare_models.py / tabnet_runner.py で共有するデータ読込・前処理関数。
このファイルは models/engine_b_compare/data_utils.py に配置する前提。

【2026-10-02 修正】
- conn_state_recall: 旧版は Test の Label=1 全体（34-1 と 8-1 の両方）を
  S0/S3 で分けていたため、列名は「34-1」なのに 8-1（Hakai、全件S0）の
  404ブロックが S0 側に混入していた。capture で 34-1 に絞るよう修正し、
  8-1（未学習ファミリ）の再現率を別列 Recall_8-1 として出力する。
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../models/engine_b_compare
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))        # .../pcap-anomaly-detector
_SRC_ENGINE_B = os.path.join(_REPO_ROOT, "src", "engine_b")
for _p in (_SRC_ENGINE_B, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import OUTPUT_DIR  # noqa: E402

FEATURE_COLS = [
    "IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio",
    "Burst_Active_Ratio", "Payload_Bytes_MAD", "Byte_Ratio_Median", "Connection_Count",
]

EXPLAINABILITY = {
    "random_forest": "SHAP(TreeExplainer) 対応",
    "xgboost": "SHAP(TreeExplainer) 対応",
    "lightgbm": "SHAP(TreeExplainer) 対応",
    "catboost": "SHAP(TreeExplainer) 対応",
    "mlp": "SHAP(KernelExplainer) 対応・計算コスト高",
    "tabnet": "Attention重み+SHAP(Kernel) 対応",
    "cnn1d": "勾配ベース手法を別途検討（未実施）",
    "gru": "勾配ベース手法を別途検討（未実施）",
    "tcn": "勾配ベース手法を別途検討（未実施）",
}

ALGORITHM_LABEL = {
    "random_forest": "決定木（バギング）",
    "xgboost": "決定木（勾配ブースティング）",
    "lightgbm": "決定木（勾配ブースティング）",
    "catboost": "決定木（勾配ブースティング）",
    "mlp": "ディープラーニング（全結合）",
    "tabnet": "ディープラーニング（表形式注目機構）",
    "cnn1d": "ディープラーニング（1次元畳み込み）",
    "gru": "ディープラーニング（再帰結合）",
    "tcn": "ディープラーニング（時系列畳み込み）",
}


def load_features():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_engine_b_features.csv"))


def load_timeseries():
    d = np.load(os.path.join(OUTPUT_DIR, "dataset_engine_b_timeseries.npz"), allow_pickle=True)
    X = np.stack([d["iat"], d["fwd_bytes"], d["bwd_bytes"]], axis=1)  # (N, 3, 20)
    return dict(X=X, y=d["label"], split=d["split"], capture=d["capture"], block_id=d["block_id"])


def load_metadata():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_engine_b_metadata.csv"))


def model_size_mb(save_fn):
    """save_fn(path) でモデルを保存させ、ファイルサイズ(MB)を返す（単一ファイルを書くモデル用）"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        save_fn(path)
        size = os.path.getsize(path) / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.remove(path)
    return size


def conn_state_recall(block_ids, y_true, y_pred, meta_df):
    """
    Test の C2(Label=1) ブロックについて、以下の再現率を返す。
      Recall_34-1_S0 : 34-1（既知ファミリMirai）のうち、応答なし(S0)中心のブロック
      Recall_34-1_S3 : 34-1 のうち、接続確立(S3)中心のブロック  ← 近道学習の判定に使う
      Recall_8-1     : 8-1（未学習ファミリHakai。全件S0）のブロック ← 未知ファミリへの汎化
    """
    df = pd.DataFrame({"block_id": block_ids, "y_true": y_true, "y_pred": y_pred})
    df = df.merge(meta_df[["block_id", "conn_state_major", "capture"]], on="block_id", how="left")
    pos = df[df["y_true"] == 1]

    def _recall(sub):
        return float((sub["y_pred"] == 1).mean()) if len(sub) else None

    s341 = pos[pos["capture"] == "34-1"]
    return {
        "Recall_34-1_S0": _recall(s341[s341["conn_state_major"] == "S0"]),
        "Recall_34-1_S3": _recall(s341[s341["conn_state_major"] == "S3"]),
        "Recall_8-1": _recall(pos[pos["capture"] == "8-1"]),
        "n_34-1_S0": int((s341["conn_state_major"] == "S0").sum()),
        "n_34-1_S3": int((s341["conn_state_major"] == "S3").sum()),
        "n_8-1": int((pos["capture"] == "8-1").sum()),
    }


def prepare_xy(feat_df, scale=False, clip_q=(0.01, 0.99)):
    """
    1. inf→NaN  2. Trainの中央値で欠損補完（Testにも同じ値）
    3. scale=True（MLP・TabNet向け）: Trainの1/99パーセンタイルでクリップ→StandardScaler
    """
    train = feat_df[feat_df["split"] == "train"]
    test = feat_df[feat_df["split"] == "test"]

    Xtr_df = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).copy()
    Xte_df = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).copy()

    medians = Xtr_df.median()
    if medians.isna().any():
        medians = medians.fillna(0.0)
    Xtr_df = Xtr_df.fillna(medians)
    Xte_df = Xte_df.fillna(medians)

    if scale:
        lo_q, hi_q = clip_q
        for c in Xtr_df.columns:
            lo, hi = Xtr_df[c].quantile(lo_q), Xtr_df[c].quantile(hi_q)
            Xtr_df[c] = Xtr_df[c].clip(lower=lo, upper=hi)
            Xte_df[c] = Xte_df[c].clip(lower=lo, upper=hi)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_df.values)
        Xte = scaler.transform(Xte_df.values)
    else:
        Xtr, Xte = Xtr_df.values, Xte_df.values

    return Xtr, train["label"].values, Xte, test["label"].values, train, test
