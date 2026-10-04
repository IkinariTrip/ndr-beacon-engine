"""
synthetic_family_augmentation.py
「学習時のC2ファミリの多様性を増やせば、未知ファミリ（Hakai）への汎化は
改善するか」を、本物のデータだけで、今すぐ検証するための実験スクリプト。

【位置づけ】
  University of Navarraリポジトリ・Cobalt Strike Malleable C2プロファイル・
  MITRE Calderaは、いずれも外部環境の構築が必要で、この場では即座に試せない。
  そこで、実在するMirai（7-1・48-1）の通信間隔パターンを基に、間隔のスケールや
  ジッターの強さを変えた「疑似C2ファミリ」を複数人工的に作り、学習データの
  多様性を増やした場合に、本物の未学習ファミリ（Hakai, 8-1）への汎化が
  改善するかどうかを検証する。

【正直な限界（必ず結果と一緒に報告すること）】
  これは本物の多様なデータの代わりにはならない。Hakaiの実際の通信リズムが、
  人工的に作ったバリエーションの範囲から外れていれば、効果は出ない。
  「効果が出なかった」という結果も、「人工的な水増しでは不十分で、本物の
  多様なデータ（Navarra等）への投資が必要」という有益な判断材料になる。

models/engine_b_compare/ に配置して実行する。

実行:
    python models/engine_b_compare/synthetic_family_augmentation.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (  # noqa: E402
    OUTPUT_DIR, load_features, load_metadata, conn_state_recall,
)

TIMING_COLS = ["IAT_Median", "IAT_MAD", "IAT_CV", "Periodic_Peak_Ratio", "Connection_Count"]

# 疑似ファミリの作り方（Miraiの実測値を基準にスケールを変える）。
# iat_scale: 通信間隔を何倍にするか（1.0=Miraiそのまま、0.2=5倍速い、3.0=3倍遅い）
# jitter_noise: IAT_CV・IAT_MADに加える、間隔のばらつきの強さ（ジッターを模す）
# conn_scale: Connection_Countのスケール（間隔を変えると当然逆比例するが、
#             実際のマルウェアは再送戦略が違うこともあるため、独立に少し揺らす）
PSEUDO_FAMILIES = [
    dict(name="家族A（Mirai実測そのまま）", iat_scale=1.0, jitter_noise=0.0, conn_scale=1.0),
    dict(name="家族B（5倍速いビーコン）", iat_scale=0.2, jitter_noise=0.05, conn_scale=4.0),
    dict(name="家族C（3倍遅いビーコン）", iat_scale=3.0, jitter_noise=0.05, conn_scale=0.4),
    dict(name="家族D（強いジッター）", iat_scale=1.0, jitter_noise=0.3, conn_scale=1.0),
    dict(name="家族E（10倍遅い・低頻度）", iat_scale=10.0, jitter_noise=0.1, conn_scale=0.1),
]


def make_pseudo_family(Xtr_mirai_df, params, rng):
    """Mirai実測ブロックを基に、スケール・ジッターを変えた疑似ファミリを1つ作る"""
    df = Xtr_mirai_df.copy()
    scale = params["iat_scale"]
    noise = params["jitter_noise"]

    df["IAT_Median"] = df["IAT_Median"] * scale
    df["IAT_MAD"] = df["IAT_MAD"] * scale * (1 + rng.normal(0, noise, len(df)).clip(-0.9, 2))
    df["IAT_CV"] = (df["IAT_CV"] + rng.normal(0, noise, len(df))).clip(0, None)
    df["Periodic_Peak_Ratio"] = (df["Periodic_Peak_Ratio"] - noise * 0.5).clip(0, 1)
    df["Connection_Count"] = df["Connection_Count"] * params["conn_scale"]
    return df


def build_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                              loss_function="Logloss", random_seed=42,
                              verbose=0, thread_count=1)


def evaluate(Xtr, ytr, Xte, yte, test_meta_ids, meta_df, threshold=0.5):
    from sklearn.metrics import f1_score, average_precision_score

    model = build_model()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    neg = yte == 0

    row = dict(
        F1=f1_score(yte, y_pred),
        PR_AUC=average_precision_score(yte, proba),
        FPR=float((y_pred[neg] == 1).mean()) if neg.any() else None,
    )
    row.update(conn_state_recall(test_meta_ids, yte, y_pred, meta_df))
    return row


def main():
    feat_df = load_features()
    meta_df = load_metadata()

    train = feat_df[feat_df["split"] == "train"].copy()
    test = feat_df[feat_df["split"] == "test"].copy()

    Xtr_df = train[TIMING_COLS].replace([np.inf, -np.inf], np.nan)
    Xtr_df = Xtr_df.fillna(Xtr_df.median())
    Xte_df = test[TIMING_COLS].replace([np.inf, -np.inf], np.nan)
    Xte_df = Xte_df.fillna(Xtr_df.median())  # Trainの中央値で統一（リーク防止）

    ytr = train["label"].values
    yte = test["label"].values
    test_ids = test["block_id"].values

    mirai_mask = ytr == 1
    Xtr_mirai = Xtr_df[mirai_mask].reset_index(drop=True)
    Xtr_benign = Xtr_df[~mirai_mask].reset_index(drop=True)
    n_mirai = len(Xtr_mirai)
    print(f"[*] 実在するMirai陽性ブロック: {n_mirai} 件（これを基に疑似ファミリを作る）")

    rng = np.random.default_rng(42)

    # ---- ベースライン：現状どおり、Miraiのみで学習 ----
    print("\n[*] ベースライン（Miraiのみ、現状の方法）を学習中...")
    row_base = evaluate(Xtr_df.values, ytr, Xte_df.values, yte, test_ids, meta_df)

    # ---- 拡張版：疑似ファミリを混ぜて多様性を持たせて学習 ----
    print(f"[*] 拡張版（Mirai + 疑似ファミリ{len(PSEUDO_FAMILIES) - 1}種）を学習中...")
    aug_X_list = [Xtr_benign]  # 正常データはそのまま
    aug_y_list = [np.zeros(len(Xtr_benign), dtype=int)]
    for params in PSEUDO_FAMILIES:
        fam_df = make_pseudo_family(Xtr_mirai, params, rng)
        aug_X_list.append(fam_df)
        aug_y_list.append(np.ones(len(fam_df), dtype=int))
        print(f"   -> {params['name']}: {len(fam_df)}件 追加"
              f"（IAT_Median中央値={fam_df['IAT_Median'].median():.1f}）")

    Xtr_aug = pd.concat(aug_X_list, ignore_index=True).values
    ytr_aug = np.concatenate(aug_y_list)
    row_aug = evaluate(Xtr_aug, ytr_aug, Xte_df.values, yte, test_ids, meta_df)

    # ---- 結果比較 ----
    res = pd.DataFrame([
        dict(条件="ベースライン（Miraiのみ）", **row_base),
        dict(条件=f"疑似ファミリ拡張（{len(PSEUDO_FAMILIES)}種混合）", **row_aug),
    ])
    out_path = os.path.join(OUTPUT_DIR, "synthetic_family_augmentation_result.csv")
    res.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("学習時の多様性を増やすと、未知ファミリ(Hakai)への汎化は改善するか")
    print("=" * 90)
    with pd.option_context("display.width", 150):
        print(res.round(3).to_string(index=False))

    print("\n[読み方]")
    print("  Recall_8-1 が ベースライン → 拡張版 で上昇 → 多様性を増やす方向性は有効。")
    print("  本物の多様なデータ（Navarra等）への投資を優先する価値が高い")
    print("  Recall_8-1 が変化しない/悪化 → 今回の人工的な水増し方法では効果が出ない。")
    print("  Hakaiの実際のパターンは、Miraiの間隔を単純にスケールしただけでは")
    print("  再現できない可能性が高く、本物の別系統データ（Navarra・Caldera等）が")
    print("  より重要という結論を補強する")
    print("\n[重要な限界]")
    print("  これは本物の多様なデータの代わりにはならない。あくまで")
    print("  「多様性を増やす方向性そのものに価値があるか」を見るための")
    print("  簡易的な予備実験であることに注意。")
    print(f"\n[+] 結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
