import numpy as np
import pandas as pd
from scipy.stats import entropy

VERTICAL_FEATURES = [
    "Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio", "Avg_Flow_Duration",
    "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio", "Down_Up_Ratio", "SYN_Flag_Ratio",
]
HORIZONTAL_FEATURES = ["Unique_Dst_IPs", "Dst_IP_Ratio", "Top_Port_Share", "No_Data_Return_Ratio"]
OTHER_FEATURES = ["Block_Duration", "Flow_Rate", "Fwd_Bytes_Total", "Bwd_Bytes_Total",
                  "Byte_Amplification_Ratio"]

H_RULE = dict(min_flows=10, ip_ratio=0.8, port_share=0.8, no_return=0.8)

def calculate_entropy(series):
    counts = series.value_counts()
    return float(entropy(counts)) if len(counts) > 1 else 0.0

def flows_to_dataframe(flows):
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
        fwd_payload_lengths = [max(0, l - 54) for l in fwd_lengths]
        bwd_payload_lengths = [max(0, l - 54) for l in bwd_lengths]

        fwd_len_mean = float(np.mean(fwd_payload_lengths)) if fwd_payload_lengths else 0.0
        bwd_len_mean = float(np.mean(bwd_payload_lengths)) if bwd_payload_lengths else 0.0

        total_fwd = len(fwd_lengths)
        total_bwd = len(bwd_lengths)
        down_up = float(total_bwd / total_fwd) if total_fwd > 0 else (1.0 if total_bwd > 0 else 0.0)

        has_syn = 1.0 if any(flg & 0x02 for flg in flags) else 0.0
        dur = (max(times) - min(times)) * 1e6 if len(times) > 1 else 0.0
        fwd_bytes_total = float(np.sum(fwd_payload_lengths)) if fwd_payload_lengths else 0.0
        bwd_bytes_total = float(np.sum(bwd_payload_lengths)) if bwd_payload_lengths else 0.0

        flow_records.append({
            "Src_IP": flow.src_ip, "Dst_IP": flow.dst_ip,
            "Src_Port": flow.src_port, "Dst_Port": flow.dst_port,
            "Flow_Duration": dur, "Fwd_Pkt_Mean": fwd_len_mean, "Bwd_Pkt_Mean": bwd_len_mean,
            "Total_Fwd_Pkts": total_fwd, "Total_Bwd_Pkts": total_bwd,
            "Down_Up_Ratio": down_up, "Has_SYN": has_syn,
            "Start_Time": float(min(times)),
            "End_Time": float(max(times)),
            "Fwd_Bytes": fwd_bytes_total, "Bwd_Bytes": bwd_bytes_total,
        })
    return pd.DataFrame(flow_records)

def _entropy_from_counts(counts):
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())

def build_blocks(df_flows, window_size):
    if df_flows.empty:
        return pd.DataFrame()
    df = df_flows.sort_values(["Src_IP", "Start_Time"], kind="mergesort").reset_index(drop=True)
    seq_in_group = df.groupby("Src_IP", sort=False).cumcount()
    df["_block_no"] = seq_in_group // window_size
    df["_block_key"] = df["Src_IP"].astype(str) + "__" + df["_block_no"].astype(str)
    g = df.groupby("_block_key", sort=False)

    base = g.agg(
        Src_IP=("Src_IP", "first"),
        Observed_Flows=("Src_IP", "size"),
        Unique_Dst_Ports=("Dst_Port", "nunique"),
        Zero_Payload_Ratio=("Fwd_Pkt_Mean", lambda s: (s == 0).mean()),
        Avg_Flow_Duration=("Flow_Duration", "mean"),
        Mean_Bwd_Pkt_Len=("Bwd_Pkt_Mean", "mean"),
        Down_Up_Ratio=("Down_Up_Ratio", "mean"),
        SYN_Flag_Ratio=("Has_SYN", "mean"),
        Unique_Dst_IPs=("Dst_IP", "nunique"),
        No_Data_Return_Ratio=("Bwd_Pkt_Mean", lambda s: (s == 0).mean()),
        Block_Start=("Start_Time", "min"),
        Block_End=("End_Time", "max"),
        Fwd_Bytes_Total=("Fwd_Bytes", "sum"),
        Bwd_Bytes_Total=("Bwd_Bytes", "sum"),
    )

    df["_fbr"] = df["Total_Fwd_Pkts"] / (df["Total_Fwd_Pkts"] + df["Total_Bwd_Pkts"] + 1e-5)
    base["Fwd_Bwd_Pkt_Ratio"] = g["_fbr"].mean()

    port_counts = df.groupby(["_block_key", "Dst_Port"], sort=False).size().rename("cnt").reset_index()
    port_counts = port_counts.sort_values(["_block_key", "cnt", "Dst_Port"],
                                          ascending=[True, False, True], kind="mergesort")
    top_port_row = port_counts.groupby("_block_key", sort=False).first()
    base["Top_Dst_Port"] = top_port_row["Dst_Port"].reindex(base.index)
    base["Top_Port_Share"] = (top_port_row["cnt"].reindex(base.index) / base["Observed_Flows"])
    base["Port_Entropy"] = (port_counts.groupby("_block_key", sort=False)["cnt"]
                            .apply(lambda s: _entropy_from_counts(s.values)).reindex(base.index))

    ip_counts = df.groupby(["_block_key", "Dst_IP"], sort=False).size().rename("cnt").reset_index()
    ip_counts = ip_counts.sort_values(["_block_key", "cnt", "Dst_IP"],
                                      ascending=[True, False, True], kind="mergesort")
    top_ip_row = ip_counts.groupby("_block_key", sort=False).first()
    base["Top_Dst_IP"] = top_ip_row["Dst_IP"].reindex(base.index)

    base = base.reset_index(drop=True)
    base["Dst_IP_Ratio"] = base["Unique_Dst_IPs"] / base["Observed_Flows"]
    duration = (base["Block_End"] - base["Block_Start"]).clip(lower=1e-3)
    base["Block_Duration"] = duration
    base["Flow_Rate"] = base["Observed_Flows"] / duration
    base["Byte_Amplification_Ratio"] = base["Bwd_Bytes_Total"] / base["Fwd_Bytes_Total"].clip(lower=1.0)

    cols = ["Src_IP", "Observed_Flows", "Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio",
            "Avg_Flow_Duration", "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio", "Down_Up_Ratio",
            "SYN_Flag_Ratio", "Unique_Dst_IPs", "Dst_IP_Ratio", "Top_Port_Share",
            "No_Data_Return_Ratio", "Top_Dst_Port", "Top_Dst_IP", "Block_Start", "Block_End",
            "Block_Duration", "Flow_Rate", "Fwd_Bytes_Total", "Bwd_Bytes_Total",
            "Byte_Amplification_Ratio"]
    return base[cols]

def _build_blocks_loop(df_flows, window_size):
    agg_blocks = []
    for src_ip, group in df_flows.sort_values(["Src_IP", "Start_Time"]).groupby("Src_IP"):
        for i in range(0, len(group), window_size):
            chunk = group.iloc[i:i + window_size]
            if len(chunk) == 0:
                continue
            n = len(chunk)
            block_start = float(chunk["Start_Time"].min())
            block_end = float(chunk["End_Time"].max())
            duration = max(block_end - block_start, 1e-3)
            fwd_total = float(chunk["Fwd_Bytes"].sum())
            bwd_total = float(chunk["Bwd_Bytes"].sum())
            agg_blocks.append({
                "Src_IP": src_ip, "Observed_Flows": n,
                "Unique_Dst_Ports": chunk["Dst_Port"].nunique(),
                "Port_Entropy": calculate_entropy(chunk["Dst_Port"]),
                "Zero_Payload_Ratio": (chunk["Fwd_Pkt_Mean"] == 0).mean(),
                "Avg_Flow_Duration": chunk["Flow_Duration"].mean(),
                "Mean_Bwd_Pkt_Len": chunk["Bwd_Pkt_Mean"].mean(),
                "Fwd_Bwd_Pkt_Ratio": (chunk["Total_Fwd_Pkts"] /
                                      (chunk["Total_Fwd_Pkts"] + chunk["Total_Bwd_Pkts"] + 1e-5)).mean(),
                "Down_Up_Ratio": chunk["Down_Up_Ratio"].mean(),
                "SYN_Flag_Ratio": chunk["Has_SYN"].mean(),
                "Unique_Dst_IPs": chunk["Dst_IP"].nunique(),
                "Dst_IP_Ratio": chunk["Dst_IP"].nunique() / n,
                "Top_Port_Share": chunk["Dst_Port"].value_counts().iloc[0] / n,
                "No_Data_Return_Ratio": (chunk["Bwd_Pkt_Mean"] == 0).mean(),
                "Top_Dst_Port": int(chunk["Dst_Port"].value_counts().index[0]),
                "Top_Dst_IP": chunk["Dst_IP"].value_counts().index[0],
                "Block_Start": block_start, "Block_End": block_end,
                "Block_Duration": duration, "Flow_Rate": n / duration,
                "Fwd_Bytes_Total": fwd_total, "Bwd_Bytes_Total": bwd_total,
                "Byte_Amplification_Ratio": bwd_total / max(fwd_total, 1.0),
            })
    return pd.DataFrame(agg_blocks)

def horizontal_scan_flags(df_agg, min_flows=None, ip_ratio=None, port_share=None, no_return=None):
    p = dict(H_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, ip_ratio=ip_ratio,
                                         port_share=port_share, no_return=no_return).items() if v is not None})
    return ((df_agg["Observed_Flows"] >= p["min_flows"]) &
            (df_agg["Dst_IP_Ratio"] >= p["ip_ratio"]) &
            (df_agg["Top_Port_Share"] >= p["port_share"]) &
            (df_agg["No_Data_Return_Ratio"] >= p["no_return"]))