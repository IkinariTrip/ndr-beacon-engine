"""
src/engine_a/features.py
エンジンA（内部スキャン検知）の特徴量計算。app_v5.py と評価スクリプトの両方から使う。
（学習・評価・アプリで計算がずれないよう、計算を1か所にまとめる）

既存の8特徴量（縦スキャン用・CatBoost v4 が学習済み）の計算は、app_v4.py と1行も変えていない。
今回、水平スキャン検知のための特徴量を4つ追加した（既存モデルには入力しない）。

  Unique_Dst_IPs        : ブロック内の宛先IPの種類数
  Dst_IP_Ratio          : 宛先IPの種類数 ÷ フロー数（1に近いほど「毎回違う相手」）
  Top_Port_Share        : 最も多い宛先ポートの割合（1に近いほど「同じポートばかり」）
  No_Data_Return_Ratio  : 相手からデータ（ペイロード）が返ってこなかったフローの割合

水平スキャン＝「同じポートを、毎回違う相手に試し、ほとんど応答がない」
  → Dst_IP_Ratio 高 × Top_Port_Share 高 × No_Data_Return_Ratio 高
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy

VERTICAL_FEATURES = [
    "Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio", "Avg_Flow_Duration",
    "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio", "Down_Up_Ratio", "SYN_Flag_Ratio",
]
HORIZONTAL_FEATURES = ["Unique_Dst_IPs", "Dst_IP_Ratio", "Top_Port_Share", "No_Data_Return_Ratio"]

# 水平スキャン判定の初期値（評価スクリプト eval_engine_a_horizontal.py の結果で見直す）
H_RULE = dict(min_flows=10, ip_ratio=0.8, port_share=0.8, no_return=0.8)


def calculate_entropy(series):
    counts = series.value_counts()
    return float(entropy(counts)) if len(counts) > 1 else 0.0


def flows_to_dataframe(flows):
    """extract_flows_from_pcap の結果をフロー単位の表にする（app_v4.py と同一の計算）。"""
    flow_records = []
    for flow in flows:
        pkts = getattr(flow, "packets", [])
        if not pkts:
            continue
        times = [p[0] for p in pkts]
        lengths = [p[1] for p in pkts]
        directions = [p[2] for p in pkts]
        flags = [p[3] for p in pkts]

        fwd_lengths = [l for l, d in zip(lengths, directions) if d == 1]
        bwd_lengths = [l for l, d in zip(lengths, directions) if d == -1]

        # ヘッダー54バイトを除いた純粋ペイロード長
        fwd_payload_lengths = [max(0, l - 54) for l in fwd_lengths]
        bwd_payload_lengths = [max(0, l - 54) for l in bwd_lengths]

        fwd_len_mean = float(np.mean(fwd_payload_lengths)) if fwd_payload_lengths else 0.0
        bwd_len_mean = float(np.mean(bwd_payload_lengths)) if bwd_payload_lengths else 0.0

        total_fwd = len(fwd_lengths)
        total_bwd = len(bwd_lengths)

        down_up = float(total_bwd / total_fwd) if total_fwd > 0 else (1.0 if total_bwd > 0 else 0.0)
        has_syn = 1.0 if any(flg & 0x02 for flg in flags) else 0.0
        dur = (max(times) - min(times)) * 1e6 if len(times) > 1 else 0.0

        flow_records.append({
            "Src_IP": flow.src_ip, "Dst_IP": flow.dst_ip,
            "Src_Port": flow.src_port, "Dst_Port": flow.dst_port,
            "Flow_Duration": dur, "Fwd_Pkt_Mean": fwd_len_mean, "Bwd_Pkt_Mean": bwd_len_mean,
            "Total_Fwd_Pkts": total_fwd, "Total_Bwd_Pkts": total_bwd,
            "Down_Up_Ratio": down_up, "Has_SYN": has_syn,
            "Start_Time": float(min(times)),          # 追加（表示・評価用。特徴量には使わない）
        })
    return pd.DataFrame(flow_records)


def build_blocks(df_flows, window_size):
    """送信元IPごとに window_size フローずつ区切り、特徴量を計算する（区切り方は app_v4.py と同一）。"""
    agg_blocks = []
    for src_ip, group in df_flows.groupby("Src_IP"):
        for i in range(0, len(group), window_size):
            chunk = group.iloc[i:i + window_size]
            if len(chunk) == 0:
                continue
            n = len(chunk)
            agg_blocks.append({
                "Src_IP": src_ip,
                "Observed_Flows": n,
                # ---- 既存の8特徴量（app_v4.py と同一）----
                "Unique_Dst_Ports": chunk["Dst_Port"].nunique(),
                "Port_Entropy": calculate_entropy(chunk["Dst_Port"]),
                "Zero_Payload_Ratio": (chunk["Fwd_Pkt_Mean"] == 0).mean(),
                "Avg_Flow_Duration": chunk["Flow_Duration"].mean(),
                "Mean_Bwd_Pkt_Len": chunk["Bwd_Pkt_Mean"].mean(),
                "Fwd_Bwd_Pkt_Ratio": (chunk["Total_Fwd_Pkts"] /
                                      (chunk["Total_Fwd_Pkts"] + chunk["Total_Bwd_Pkts"] + 1e-5)).mean(),
                "Down_Up_Ratio": chunk["Down_Up_Ratio"].mean(),
                "SYN_Flag_Ratio": chunk["Has_SYN"].mean(),
                # ---- 追加：水平スキャン用の4特徴量 ----
                "Unique_Dst_IPs": chunk["Dst_IP"].nunique(),
                "Dst_IP_Ratio": chunk["Dst_IP"].nunique() / n,
                "Top_Port_Share": chunk["Dst_Port"].value_counts().iloc[0] / n,
                "No_Data_Return_Ratio": (chunk["Bwd_Pkt_Mean"] == 0).mean(),
                "Top_Dst_Port": int(chunk["Dst_Port"].value_counts().index[0]),
                "Block_Start": float(chunk["Start_Time"].min()) if "Start_Time" in chunk else np.nan,
                "Block_End": float(chunk["Start_Time"].max()) if "Start_Time" in chunk else np.nan,
            })
    return pd.DataFrame(agg_blocks)


def horizontal_scan_flags(df_agg, min_flows=None, ip_ratio=None, port_share=None, no_return=None):
    """水平スキャンの判定（ルール）。True/False の Series を返す。"""
    p = dict(H_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, ip_ratio=ip_ratio,
                                     port_share=port_share, no_return=no_return).items() if v is not None})
    return ((df_agg["Observed_Flows"] >= p["min_flows"]) &
            (df_agg["Dst_IP_Ratio"] >= p["ip_ratio"]) &
            (df_agg["Top_Port_Share"] >= p["port_share"]) &
            (df_agg["No_Data_Return_Ratio"] >= p["no_return"]))
