"""
shap_dl_channels.py
時系列DL（1D-CNN / GRU / TCN）の判定根拠を SHAP（GradientExplainer）で可視化し、
「どの入力チャンネル（IAT / 送信バイト / 受信バイト）を見て判定したか」を
評価データのグループ別に比較する。models/engine_b_compare/ に配置して実行する。

【確かめたいこと（アブレーションの結果をSHAPで直接確認する）】
  - 34-1 S3（接続確立したC2）: IATチャンネルが主な根拠か
  - 8-1 Hakai（未学習・全件応答なし）: バイトチャンネルが主な根拠か（＝近道）
  → グループによって「見ているチャンネル」が入れ替わっていれば、
    DLが2つの判定根拠（周期と応答の有無）を使い分けていることをSHAPで示せる

【出力（raw_pcap/engine_b_iot23/_output/shap_dl/ 以下）】
  shap_dl_channel_share.png  : グループ別・チャンネル別の寄与率（スライド貼り付け用）
  shap_dl_channel_share.csv  : 寄与率（%）と、C2方向への押し上げ量（符号付き）
  shap_dl_timestep_<model>.png : 20回分の時刻ごとの寄与（補足用）

実行:
    python models/engine_b_compare/shap_dl_channels.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import OUTPUT_DIR, load_timeseries, load_metadata  # noqa: E402
import dl_models  # noqa: E402

MODELS = ["cnn1d", "gru", "tcn"]
CHANNELS = ["IAT", "送信バイト", "受信バイト"]
CH_EN = ["IAT", "Fwd bytes", "Bwd bytes"]
GROUPS = ["34-1 S0", "34-1 S3", "8-1 Hakai", "正常(Test)"]
GROUPS_EN = ["34-1 S0\n(no reply)", "34-1 S3\n(connected)", "8-1 Hakai\n(unseen)", "Normal\n(Test)"]
N_PER_GROUP = 150     # グループごとに説明するブロック数の上限
N_BACKGROUND = 100    # SHAPの基準データ（Trainから無作為抽出）
SEED = 42
EPOCHS = 30
OUT_DIR = os.path.join(OUTPUT_DIR, "shap_dl")


def to_shap_array(sv):
    """SHAPの戻り値の形（バージョン差）を (N, 3, 20) にそろえる"""
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 4:
        sv = sv[..., 0]
    return sv


def main():
    import torch
    import shap

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = load_timeseries()
    meta = load_metadata()[["block_id", "capture", "conn_state_major"]]

    tr = ts["split"] == "train"
    te = ts["split"] == "test"
    Xtr, ytr = ts["X"][tr], ts["y"][tr]
    Xte, yte = ts["X"][te], ts["y"][te]

    # 評価データをグループに分ける
    df = pd.DataFrame({"block_id": ts["block_id"][te], "y": yte})
    df = df.merge(meta, on="block_id", how="left")
    masks = {
        "34-1 S0": (df.y == 1) & (df.capture == "34-1") & (df.conn_state_major == "S0"),
        "34-1 S3": (df.y == 1) & (df.capture == "34-1") & (df.conn_state_major == "S3"),
        "8-1 Hakai": (df.y == 1) & (df.capture == "8-1"),
        "正常(Test)": (df.y == 0),
    }
    rng = np.random.default_rng(SEED)
    pick = {}
    for g, m in masks.items():
        idx = np.where(m.values)[0]
        if len(idx) > N_PER_GROUP:
            idx = rng.choice(idx, N_PER_GROUP, replace=False)
        pick[g] = np.sort(idx)
        print(f"[*] グループ {g}: 説明対象 {len(pick[g])} ブロック")

    # 学習時と同じ基準（Trainの統計量）で正規化した入力をSHAPに渡す
    mean, std = dl_models.compute_channel_stats(Xtr)
    Xtr_n = dl_models.apply_channel_norm(Xtr, mean, std).astype(np.float32)
    Xte_n = dl_models.apply_channel_norm(Xte, mean, std).astype(np.float32)
    bg_idx = rng.choice(len(Xtr_n), min(N_BACKGROUND, len(Xtr_n)), replace=False)
    background = torch.tensor(Xtr_n[bg_idx])

    class Wrap(torch.nn.Module):
        """出力を (batch, 1) にそろえる（SHAPが要求する形）"""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    rows = []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.6), sharey=True)
    colors = ["#0033FF", "#E8175D", "#B26A00"]
    for k, mname in enumerate(MODELS):
        print(f"\n[*] 学習中: {mname}（3チャンネル入力、seed={SEED}）", flush=True)
        model, proba, _, _ = dl_models.train_torch_model(
            mname, Xtr, ytr, Xte, epochs=EPOCHS, seed=SEED)
        model.eval()
        y_pred = (proba >= 0.5).astype(int)

        explainer = shap.GradientExplainer(Wrap(model), background)
        share_tbl = []
        ts_profile = {}
        for g in GROUPS:
            idx = pick[g]
            if len(idx) == 0:
                share_tbl.append([np.nan] * 3)
                continue
            print(f"   -> SHAP計算中: {g}", flush=True)
            sv = to_shap_array(explainer.shap_values(torch.tensor(Xte_n[idx])))   # (n, 3, 20)
            abs_ch = np.abs(sv).sum(axis=2).mean(axis=0)                           # (3,)
            share = abs_ch / abs_ch.sum() * 100
            signed = sv.sum(axis=2).mean(axis=0)                                    # (3,) +はC2方向
            share_tbl.append(share)
            ts_profile[g] = np.abs(sv).mean(axis=0)                                 # (3, 20)
            rows.append(dict(
                model=mname, group=g, n_blocks=len(idx),
                detect_rate=float(y_pred[idx].mean()),
                share_IAT=share[0], share_fwd_bytes=share[1], share_bwd_bytes=share[2],
                share_bytes_total=share[1] + share[2],
                push_IAT=signed[0], push_fwd_bytes=signed[1], push_bwd_bytes=signed[2],
            ))

        # 積み上げ棒グラフ（グループ別のチャンネル寄与率）
        ax = axes[k]
        share_arr = np.array(share_tbl)
        bottom = np.zeros(len(GROUPS))
        for c in range(3):
            ax.bar(range(len(GROUPS)), share_arr[:, c], bottom=bottom, color=colors[c],
                   label=CH_EN[c], width=0.65)
            for i, v in enumerate(share_arr[:, c]):
                if v >= 8:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")
            bottom += np.nan_to_num(share_arr[:, c])
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS_EN, fontsize=9)
        ax.set_title(mname, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        if k == 0:
            ax.set_ylabel("share of mean |SHAP| (%)")

        # 補足：時刻ごとの寄与（20回分）
        f2, ax2 = plt.subplots(1, len(GROUPS), figsize=(15, 3.2), sharey=True)
        for j, g in enumerate(GROUPS):
            if g not in ts_profile:
                continue
            for c in range(3):
                ax2[j].plot(ts_profile[g][c], color=colors[c], label=CH_EN[c])
            ax2[j].set_title(GROUPS_EN[j].replace("\n", " "), fontsize=10)
            ax2[j].set_xlabel("flow index in block (0-19)")
        ax2[0].legend(fontsize=8)
        f2.suptitle(f"{mname}: mean |SHAP| per time step", fontsize=11)
        f2.tight_layout()
        f2.savefig(os.path.join(OUT_DIR, f"shap_dl_timestep_{mname}.png"), dpi=150, bbox_inches="tight")
        plt.close(f2)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.suptitle("Time-series DL: which input channel drives the decision (SHAP, GradientExplainer)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shap_dl_channel_share.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "shap_dl_channel_share.csv"), index=False)
    print("\n" + "=" * 100)
    print("時系列DL：グループ別のチャンネル寄与率（%）と検知率")
    print("=" * 100)
    cols = ["model", "group", "n_blocks", "detect_rate", "share_IAT",
            "share_fwd_bytes", "share_bwd_bytes", "share_bytes_total",
            "push_IAT", "push_fwd_bytes", "push_bwd_bytes"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res[cols].round(3).to_string(index=False))
    print("\n[読み方]")
    print("  share_* : そのグループの判定に、各チャンネルがどれだけ効いたか（合計100%）")
    print("  push_*  : C2方向への押し上げ量（＋はC2寄り、−は正常寄り）")
    print("  8-1 Hakai でバイトの寄与が大きく、34-1 S3 でIATの寄与が大きければ、")
    print("  DLは『周期』と『応答の有無』を使い分けている＝Hakai検知は近道、とSHAPで示せる")
    print(f"\n[+] 出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
