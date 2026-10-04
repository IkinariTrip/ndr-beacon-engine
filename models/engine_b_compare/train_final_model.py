"""
エンジンBの採用モデル（CatBoost・モデルB：タイミング系5特徴量）を学習し、
アプリで使うためにファイルへ保存する。
学習データはTrainのみ（スライドに載せた評価結果と同じ条件のモデルにするため）。
"""
import os, sys, json, datetime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import FEATURE_COLS, load_features, prepare_xy
from catboost import CatBoostClassifier

TIMING_COLS = ["IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio", "Connection_Count"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(REPO, "models", "engine_b")
os.makedirs(SAVE_DIR, exist_ok=True)

Xtr_all, ytr, _, _, _, _ = prepare_xy(load_features(), scale=False)
idx = [FEATURE_COLS.index(c) for c in TIMING_COLS]
model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                           loss_function="Logloss", random_seed=42, verbose=0, thread_count=1)
model.fit(Xtr_all[:, idx], ytr)

model_path = os.path.join(SAVE_DIR, "catboost_engine_b_v1.cbm")
model.save_model(model_path)

meta = {
    "model": "CatBoost", "version": "v1", "feature_set": "B (timing 5)",
    "features": TIMING_COLS, "threshold": 0.5,
    "window_size": 20, "window_stride": 5,
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "train_data": "IoT-23 v2 (train split: 7-1, 48-1, 20-1, 3-1, hue-4-1)",
}

with open(os.path.join(SAVE_DIR, "catboost_engine_b_v1_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("[+] 保存しました:", model_path)