"""
train_final_model.py（更新版）
エンジンBの採用モデル（CatBoost・モデルB：タイミング系5特徴量）を学習し、
アプリで使うためにファイルへ保存する。学習データはTrainのみ。

【更新点】推論時に学習時と同じ前処理をするため、欠損値の補完値
（Trainの中央値。compare_models.py の prepare_xy と同じ計算）を
メタ情報 fill_medians として保存するようにした。
乱数シードは固定なので、モデル自体は前回保存したものと同じになる。
"""
import os, sys, json, datetime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data_utils import FEATURE_COLS, load_features, prepare_xy
from catboost import CatBoostClassifier

TIMING_COLS = ["IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio", "Connection_Count"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(REPO, "models", "engine_b")
os.makedirs(SAVE_DIR, exist_ok=True)

feat_df = load_features()
Xtr_all, ytr, _, _, _, _ = prepare_xy(feat_df, scale=False)

# prepare_xy と同じ補完値（Trainの8特徴量の中央値。全件欠損の列は0）
tr = feat_df[feat_df["split"] == "train"][FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
medians = tr.median().fillna(0.0)

idx = [FEATURE_COLS.index(c) for c in TIMING_COLS]
model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                           loss_function="Logloss", random_seed=42, verbose=0, thread_count=1)
model.fit(Xtr_all[:, idx], ytr)

model_path = os.path.join(SAVE_DIR, "catboost_engine_b_v1.cbm")
model.save_model(model_path)
meta = {
    "model": "CatBoost", "version": "v1", "feature_set": "B (timing 5)",
    "features": TIMING_COLS, "threshold": 0.5,
    "window_size": 20, "window_stride": 5, "min_pair_flows": 10,
    "fill_medians": {c: float(medians[c]) for c in TIMING_COLS},
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "train_data": "IoT-23 v2 (train split: 7-1, 48-1, 20-1, 3-1, hue-4-1)",
}
with open(os.path.join(SAVE_DIR, "catboost_engine_b_v1_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("[+] 保存しました:", model_path)
print("[+] 補完値:", meta["fill_medians"])
