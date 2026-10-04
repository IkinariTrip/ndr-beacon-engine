"""
tabnet_runner.py
TabNetだけを独立したサブプロセスで実行する（macOSのOpenMP二重初期化によるフリーズ対策）。

【2026-10-02 修正】FileExistsError の解消
  旧版はモデル容量の計測に NamedTemporaryFile（先に空ファイルを作る）を使っていたが、
  TabNet の save_model(path) は path に「フォルダ」を作ってから zip 化する仕様のため、
  同名のファイルが既にあると FileExistsError になっていた（学習自体は完了していた）。
  → 空の一時フォルダの中に、まだ存在しない名前を渡す方式に変更。
  さらに、評価指標を先に計算し、容量計測が万一失敗しても結果は失わないようにした。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (  # noqa: E402
    load_features, load_metadata, conn_state_recall,
    prepare_xy, EXPLAINABILITY, ALGORITHM_LABEL,
)


def tabnet_size_mb(model):
    """TabNet専用の容量計測。save_model は '<path>.zip' を作り、そのパスを返す。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "tabnet_model")   # まだ存在しない名前を渡す
        saved = model.save_model(target)
        if not (saved and os.path.exists(saved)):
            saved = target + ".zip"
        return os.path.getsize(saved) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = dict(model="tabnet", status="未実行（原因不明のエラー）")
    try:
        feat_df = load_features()
        meta_df = load_metadata()
        Xtr, ytr, Xte, yte, train, test = prepare_xy(feat_df, scale=True)

        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier(seed=42, verbose=0, device_name="cpu")

        t0 = time.perf_counter()
        model.fit(Xtr, ytr, max_epochs=100, patience=20,
                  batch_size=256, virtual_batch_size=128,
                  num_workers=0, drop_last=False)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = model.predict_proba(Xte)[:, 1]
        infer_time = time.perf_counter() - t0

        from sklearn.metrics import f1_score, average_precision_score, accuracy_score
        y_pred = (proba >= args.threshold).astype(int)

        # 先に評価指標を確定させる（容量計測が失敗しても結果は残す）
        result = dict(
            model="tabnet", status="実行済み", algorithm=ALGORITHM_LABEL["tabnet"],
            n_train=int(len(Xtr)), n_test=int(len(Xte)),
            Accuracy=float(accuracy_score(yte, y_pred)),
            F1=float(f1_score(yte, y_pred)),
            PR_AUC=float(average_precision_score(yte, proba)),
            train_time_sec=train_time,
            infer_time_us_per_block=infer_time / max(len(Xte), 1) * 1e6,
            model_size_MB=None, explainability=EXPLAINABILITY["tabnet"],
        )
        result.update(conn_state_recall(test["block_id"].values, yte, y_pred, meta_df))

        try:
            result["model_size_MB"] = tabnet_size_mb(model)
        except Exception as e:
            result["status"] = f"実行済み（容量計測のみ失敗: {type(e).__name__}）"

    except ImportError as e:
        result = dict(model="tabnet", status=f"未実行（ライブラリ未導入: {getattr(e, 'name', str(e))}）")
    except Exception as e:
        result = dict(model="tabnet", status=f"未実行（エラー: {type(e).__name__}: {e}）")

    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
