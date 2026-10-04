"""
compare_models.py
エンジンBの9モデル比較（A系統6種＋B系統3種）を同一条件で実行し、
エンジンAの「モデル総合比較実験テーブル」と同じ構成の表を出力する。

対象モデル（2026-09-30 合意事項。Transformerは今回対象外＝フェーズB/C以降の発展項目）:
    A系統（8特徴量＝dataset_engine_b_features.csv）:
        random_forest, xgboost, lightgbm, catboost, mlp, tabnet
    B系統（生の時系列＝dataset_engine_b_timeseries.npz。IAT/送信バイト/受信バイト）:
        cnn1d, gru, tcn

評価項目（エンジンAより追加。仕様書v2.5 第5章準拠）:
    - PR-AUC（Label=1が少ないため、正解率だけでは実態を見誤るため追加）
    - 34-1の conn_state_major 別 Recall（S0=無応答 / S3=接続確立）
      ※ 接続確立(S3)側も検知できて初めて「周期性で判定している」証拠になる
    - 学習時間・推論時間（1ブロックあたり）・モデル容量
    - 説明性の確認状況（SHAPが使えるか／別手法が必要か）

【2026-10-02 修正（3点）】
    1. フリーズ対策：macOSでXGBoost/LightGBM/CatBoost（libomp同梱）実行後に
       同一プロセスでPyTorch（TabNet）を初期化すると、OpenMPランタイムの
       二重初期化によりCPU使用率が0%のまま無反応になることがある既知の問題。
       対策として、
         a) KMP_DUPLICATE_LIB_OK等の環境変数を他importより前に設定
         b) 各モデルの n_jobs/thread_count を 1 に固定し、並列処理による
            プロセス・スレッドの競合を減らす
         c) TabNetだけは tabnet_runner.py という別プロセスで実行し、
            タイムアウト（既定300秒）を設けることで、万一フリーズしても
            Mac本体が固まったままにならないようにした
    2. 正規化（StandardScaler）に加えて、Trainの1/99パーセンタイルで
       外れ値をクリップ(ウィンザライズ)してから標準化する処理を追加。
       Connection_Count等が桁違いに大きくなるケースがあり、これが
       MLP学習時のオーバーフロー警告の原因だった（data_utils.prepare_xy参照）。
    3. ディレクトリ解決のバグ修正：旧版はrepo rootのパス計算が1階層
       足りておらず、config.pyを正しく見つけられなかった
       （data_utils.py側で修正済み）。

実行例:
    python compare_models.py
    python compare_models.py --epochs 50              # DLモデルの学習エポック数を変更
    python compare_models.py --tabnet-timeout 600      # TabNetのタイムアウトを延長
    python compare_models.py --skip-tabnet             # TabNetを飛ばして他を先に確認
"""
import os

# 【最重要】他のimportより前に設定すること。
# macOSでXGBoost/LightGBM/CatBoost（libomp同梱）とPyTorch（libomp同梱）を
# 同一プロセス内で使うと、OpenMPランタイムが二重に初期化され、
# クラッシュまたは無反応（フリーズ）を引き起こすことがある既知の問題への対策。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import tempfile
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, average_precision_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (  # noqa: E402
    EXPLAINABILITY, ALGORITHM_LABEL, OUTPUT_DIR,
    load_features, load_timeseries, load_metadata,
    model_size_mb, conn_state_recall, prepare_xy,
)

NEEDS_SCALING = {"mlp", "tabnet"}  # 勾配ベースの手法（スケールに敏感）


def evaluate_a_series(model_name, feat_df, meta_df, threshold=0.5):
    scale = model_name in NEEDS_SCALING
    Xtr, ytr, Xte, yte, train, test = prepare_xy(feat_df, scale=scale)

    try:
        if model_name == "random_forest":
            # n_jobs=1: 他ライブラリとのプロセス/スレッド競合を避けるため意図的に単一実行
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
            t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0
            size_mb = model_size_mb(lambda p: __import__("joblib").dump(model, p))

        elif model_name == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                eval_metric="logloss", random_state=42, n_jobs=1,
            )
            t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0
            size_mb = model_size_mb(lambda p: model.save_model(p))

        elif model_name == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                random_state=42, n_jobs=1, verbosity=-1,
            )
            t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0
            size_mb = model_size_mb(lambda p: model.booster_.save_model(p))

        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.08,
                loss_function="Logloss", random_seed=42, verbose=0, thread_count=1,
            )
            t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0
            size_mb = model_size_mb(lambda p: model.save_model(p))

        elif model_name == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
            t0 = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - t0
            t0 = time.perf_counter(); proba = model.predict_proba(Xte)[:, 1]; infer_time = time.perf_counter() - t0
            size_mb = model_size_mb(lambda p: __import__("joblib").dump(model, p))

        else:
            raise ValueError(model_name)

    except ImportError as e:
        return dict(model=model_name, status=f"未実行（ライブラリ未導入: {e.name}）")

    y_pred = (proba >= threshold).astype(int)
    row = dict(
        model=model_name, status="実行済み", algorithm=ALGORITHM_LABEL[model_name],
        n_train=len(Xtr), n_test=len(Xte),
        Accuracy=accuracy_score(yte, y_pred), F1=f1_score(yte, y_pred),
        PR_AUC=average_precision_score(yte, proba),
        train_time_sec=train_time, infer_time_us_per_block=infer_time / max(len(Xte), 1) * 1e6,
        model_size_MB=size_mb, explainability=EXPLAINABILITY[model_name],
    )
    row.update(conn_state_recall(test["block_id"].values, yte, y_pred, meta_df))
    return row


def evaluate_b_series(model_name, ts, meta_df, epochs=30, threshold=0.5):
    try:
        from dl_models import train_torch_model, TORCH_AVAILABLE
    except ImportError:
        return dict(model=model_name, status="未実行（dl_models.pyの読み込みに失敗）")
    if not TORCH_AVAILABLE:
        return dict(model=model_name, status="未実行（ライブラリ未導入: torch）")

    train_mask = ts["split"] == "train"
    test_mask = ts["split"] == "test"
    Xtr, ytr = ts["X"][train_mask], ts["y"][train_mask]
    Xte, yte = ts["X"][test_mask], ts["y"][test_mask]

    try:
        model, proba, train_time, infer_time_total = train_torch_model(
            model_name, Xtr, ytr, Xte, epochs=epochs,
        )
    except RuntimeError as e:
        return dict(model=model_name, status=f"未実行（{e}）")

    import torch
    size_mb = model_size_mb(lambda p: torch.save(model.state_dict(), p))
    y_pred = (proba >= threshold).astype(int)
    row = dict(
        model=model_name, status="実行済み", algorithm=ALGORITHM_LABEL[model_name],
        n_train=len(Xtr), n_test=len(Xte),
        Accuracy=accuracy_score(yte, y_pred), F1=f1_score(yte, y_pred),
        PR_AUC=average_precision_score(yte, proba),
        train_time_sec=train_time,
        infer_time_us_per_block=infer_time_total / max(len(Xte), 1) * 1e6,
        model_size_MB=size_mb, explainability=EXPLAINABILITY[model_name],
    )
    row.update(conn_state_recall(ts["block_id"][test_mask], yte, y_pred, meta_df))
    return row


def run_tabnet_isolated(timeout_sec):
    """
    TabNetは、macOSでPyTorchと他ライブラリのOpenMPランタイムが衝突すると
    フリーズすることがあるため、完全に独立した子プロセス(tabnet_runner.py)で
    実行し、タイムアウトを設けることでMac本体が固まり続けることを防ぐ。
    """
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabnet_runner.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name

    try:
        subprocess.run(
            [sys.executable, runner, "--output", out_path],
            timeout=timeout_sec, check=False,
        )
    except subprocess.TimeoutExpired:
        if os.path.exists(out_path):
            os.remove(out_path)
        return dict(model="tabnet", status=(
            f"未実行（タイムアウト: {timeout_sec}秒超過。OpenMPランタイムの衝突による"
            "フリーズの可能性が高い。--tabnet-timeoutで延長して再試行するか、"
            "--skip-tabnetで一旦除外してください）"
        ))

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return dict(model="tabnet", status="未実行（子プロセスが結果を出力せずに終了した）")

    with open(out_path) as f:
        row = json.load(f)
    os.remove(out_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="DLモデル（B系統）の学習エポック数")
    parser.add_argument("--skip-tabnet", action="store_true",
                        help="TabNetを飛ばして他のモデルを先に完了させる")
    parser.add_argument("--tabnet-timeout", type=int, default=300,
                        help="TabNetの子プロセスのタイムアウト秒数（既定300秒=5分）")
    args = parser.parse_args()

    print("[*] 特徴量テーブルを読み込み中...")
    feat_df = load_features()
    meta_df = load_metadata()

    print("[*] 生の時系列データを読み込み中...")
    try:
        ts = load_timeseries()
        has_ts = True
    except FileNotFoundError:
        print("[警告] dataset_engine_b_timeseries.npz が見つかりません。B系統はスキップします。")
        has_ts = False

    rows = []
    for m in ["random_forest", "xgboost", "lightgbm", "catboost", "mlp"]:
        print(f"\n[*] 実行中（A系統）: {m}")
        sys.stdout.flush()
        row = evaluate_a_series(m, feat_df, meta_df)
        print("   ->", row.get("status"))
        rows.append(row)

    if args.skip_tabnet:
        rows.append(dict(model="tabnet", status="未実行（--skip-tabnetで除外）"))
    else:
        print(f"\n[*] 実行中（A系統）: tabnet（別プロセスで隔離実行。タイムアウト{args.tabnet_timeout}秒）")
        sys.stdout.flush()
        row = run_tabnet_isolated(args.tabnet_timeout)
        print("   ->", row.get("status"))
        rows.append(row)

    if has_ts:
        for m in ["cnn1d", "gru", "tcn"]:
            print(f"\n[*] 実行中（B系統）: {m}")
            sys.stdout.flush()
            row = evaluate_b_series(m, ts, meta_df, epochs=args.epochs)
            print("   ->", row.get("status"))
            rows.append(row)
    else:
        for m in ["cnn1d", "gru", "tcn"]:
            rows.append(dict(model=m, status="未実行（時系列データなし）"))

    result = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    result.to_csv(out_path, index=False)

    print("\n" + "=" * 100)
    print("エンジンB モデル総合比較実験テーブル")
    print("=" * 100)
    cols = [
        "model", "status", "algorithm", "Accuracy", "F1", "PR_AUC",
        "train_time_sec", "infer_time_us_per_block", "model_size_MB",
        "Recall_34-1_S0", "Recall_34-1_S3", "Recall_8-1", "explainability",
    ]
    cols = [c for c in cols if c in result.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result[cols].to_string(index=False))

    print(
        "\n[注記] Recall_34-1_S3（接続が確立したC2通信の再現率）が高いモデルほど、"
        "応答の有無ではなく周期性で判定している可能性が高い（仕様書v2.5 5.3節）。"
        "Recall_34-1_S0のみ高く、S3が低いモデルは近道学習の疑いがあるため、"
        "SHAP監査（可能なモデルのみ）と合わせて確認すること。"
    )
    print(
        "\n[注記] MLP・TabNetの入力はTrainの1/99パーセンタイルでクリップした上でStandardScaler、"
        "CNN・GRU・TCNの入力はチャンネルごとのz-score標準化を適用している"
        "（決定木系は無処理のまま。スケール不変のため）。"
    )
    print(f"\n[+] 結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
