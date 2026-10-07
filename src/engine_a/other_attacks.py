import numpy as np
import pandas as pd

FLOOD_RULE = dict(min_flows=10, max_dst_ips=2, port_share=0.9, min_rate=20.0)
DDOS_DEST_RULE = dict(min_flows=15, min_src_ips=15, max_window_sec=10.0)
REFLECTION_RULE = dict(min_amp_ratio=10.0, min_bytes=2000.0)

REFLECTION_PORTS = {53: "DNS", 123: "NTP", 1900: "SSDP", 19: "CharGEN",
                    161: "SNMP", 11211: "Memcached", 389: "CLDAP/LDAP", 1434: "MS-SQL"}

def flood_flags(df_agg, min_flows=None, max_dst_ips=None, port_share=None, min_rate=None):
    p = dict(FLOOD_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, max_dst_ips=max_dst_ips,
                                    port_share=port_share, min_rate=min_rate).items() if v is not None})
    return ((df_agg["Observed_Flows"] >= p["min_flows"]) &
            (df_agg["Unique_Dst_IPs"] <= p["max_dst_ips"]) &
            (df_agg["Top_Port_Share"] >= p["port_share"]) &
            (df_agg["Flow_Rate"] >= p["min_rate"]))

def reflection_flags(df_agg, min_amp_ratio=None, min_bytes=None):
    p = dict(REFLECTION_RULE)
    p.update({k: v for k, v in dict(min_amp_ratio=min_amp_ratio, min_bytes=min_bytes).items()
             if v is not None})
    is_reflection_port = df_agg["Top_Dst_Port"].isin(REFLECTION_PORTS.keys())
    return (is_reflection_port &
            (df_agg["Byte_Amplification_Ratio"] >= p["min_amp_ratio"]) &
            (df_agg["Bwd_Bytes_Total"] >= p["min_bytes"]))

def build_dest_side_blocks(df_flows, window_size):
    if df_flows.empty:
        return pd.DataFrame()
    df = df_flows.sort_values(["Dst_IP", "Dst_Port", "Start_Time"], kind="mergesort").reset_index(drop=True)
    key = df["Dst_IP"].astype(str) + "|" + df["Dst_Port"].astype(str)
    seq_in_group = df.groupby(key, sort=False).cumcount()
    block_no = seq_in_group // window_size
    df["_block_key"] = key.astype(str) + "__" + block_no.astype(str)
    out = df.groupby("_block_key", sort=False).agg(
        Dst_IP=("Dst_IP", "first"),
        Dst_Port=("Dst_Port", "first"),
        Observed_Flows=("Dst_IP", "size"),
        Unique_Src_IPs=("Src_IP", "nunique"),
        Block_Start=("Start_Time", "min"),
        Block_End=("End_Time", "max"),
    ).reset_index(drop=True)
    out["Dst_Port"] = out["Dst_Port"].astype(int)
    out["Window_Sec"] = (out["Block_End"] - out["Block_Start"]).clip(lower=1e-3)
    out["Top_Src_IPs"] = ""
    return out[["Dst_IP", "Dst_Port", "Observed_Flows", "Unique_Src_IPs", "Window_Sec",
               "Block_Start", "Block_End", "Top_Src_IPs"]]

def attach_top_src_ips(df_flows, dest_blocks, window_size, top_n=5):
    if dest_blocks.empty:
        return dest_blocks
    out = dest_blocks.copy()
    results = []
    for row in out.itertuples():
        mask = (df_flows["Dst_IP"] == row.Dst_IP)
        mask = mask & (df_flows["Dst_Port"] == row.Dst_Port)
        mask = mask & (df_flows["Start_Time"] >= row.Block_Start)
        mask = mask & (df_flows["Start_Time"] <= row.Block_End)
        sub = df_flows.loc[mask]
        top_ips = sub["Src_IP"].value_counts().index[:top_n].tolist()
        results.append(", ".join(top_ips))
    out["Top_Src_IPs"] = results
    return out

def _build_dest_side_blocks_loop(df_flows, window_size):
    rows = []
    key_cols = ["Dst_IP", "Dst_Port"]
    sorted_df = df_flows.sort_values(key_cols + ["Start_Time"])
    for (dst_ip, dst_port), group in sorted_df.groupby(key_cols):
        g = group.reset_index(drop=True)
        for i in range(0, len(g), window_size):
            chunk = g.iloc[i:i + window_size]
            if len(chunk) == 0:
                continue
            start = float(chunk["Start_Time"].min())
            end = float(chunk["End_Time"].max())
            top_ips = chunk["Src_IP"].value_counts().index[:5].tolist()
            rows.append(dict(
                Dst_IP=dst_ip, Dst_Port=int(dst_port), Observed_Flows=len(chunk),
                Unique_Src_IPs=chunk["Src_IP"].nunique(),
                Window_Sec=max(end - start, 1e-3),
                Block_Start=start, Block_End=end,
                Top_Src_IPs=", ".join(top_ips),
            ))
    return pd.DataFrame(rows)

def ddos_dest_flags(df_dest, min_flows=None, min_src_ips=None, max_window_sec=None):
    p = dict(DDOS_DEST_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, min_src_ips=min_src_ips,
                                    max_window_sec=max_window_sec).items() if v is not None})
    return ((df_dest["Observed_Flows"] >= p["min_flows"]) &
            (df_dest["Unique_Src_IPs"] >= p["min_src_ips"]) &
            (df_dest["Window_Sec"] <= p["max_window_sec"]))

def _epoch(x):
    return float(x)

def flood_filters(row):
    src = row["Src_IP"]
    dst = row["Top_Dst_IP"]
    port = int(row["Top_Dst_Port"])
    proto = "tcp" if row.get("SYN_Flag_Ratio", 0) > 0 else "udp"
    rate = row["Flow_Rate"]
    n_flows = row["Observed_Flows"]
    duration = row["Block_Duration"]
    b_start = _epoch(row["Block_Start"])
    b_end = _epoch(row["Block_End"])
    
    label1 = "① 送信元→宛先の全通信（" + str(n_flows) + "フロー）"
    label2 = "② 宛先ポート" + str(port) + "/" + proto + "に絞り込み"
    label3 = "③ 観測期間のみ（約" + format(duration, ".1f") + "秒間、" + format(rate, ".1f") + "フロー/秒）"
    
    filters = {}
    filters[label1] = "ip.src==" + src + " && ip.dst==" + dst
    filters[label2] = "ip.src==" + src + " && ip.dst==" + dst + " && " + proto + ".port==" + str(port)
    filters[label3] = ("ip.src==" + src + " && ip.dst==" + dst +
                       " && frame.time_epoch>=" + format(b_start, ".0f") +
                       " && frame.time_epoch<=" + format(b_end, ".0f"))
    
    if proto == "tcp":
        label4 = "④ SYNのみ（コネクション確立前の接続試行を抽出）"
        filters[label4] = "ip.src==" + src + " && ip.dst==" + dst + " && tcp.flags.syn==1 && tcp.flags.ack==0"
    return filters

def ddos_dest_filters(row):
    dst = row["Dst_IP"]
    port = int(row["Dst_Port"])
    n_src = row["Unique_Src_IPs"]
    n_flows = row["Observed_Flows"]
    window_sec = row["Window_Sec"]
    b_start = _epoch(row["Block_Start"])
    b_end = _epoch(row["Block_End"])
    
    label1 = "① 宛先への全通信（異なる送信元IP " + str(n_src) + "件・" + str(n_flows) + "フロー）"
    label2 = "② 観測期間のみ（約" + format(window_sec, ".1f") + "秒間に集中）"
    label3 = "③ 上位の送信元IP（最大5件、カンマ区切りを ||区切りに置換して使用）"
    
    filters = {}
    filters[label1] = "ip.dst==" + dst + " && tcp.port==" + str(port) + " || ip.dst==" + dst + " && udp.port==" + str(port)
    filters[label2] = ("ip.dst==" + dst +
                       " && frame.time_epoch>=" + format(b_start, ".0f") +
                       " && frame.time_epoch<=" + format(b_end, ".0f"))
    
    src_list = [ip for ip in row["Top_Src_IPs"].split(", ") if ip]
    filters[label3] = " || ".join("ip.src==" + ip for ip in src_list)
    return filters

def reflection_filters(row):
    src = row["Src_IP"]
    dst = row["Top_Dst_IP"]
    port = int(row["Top_Dst_Port"])
    port_name = REFLECTION_PORTS.get(port, str(port))
    amp = row["Byte_Amplification_Ratio"]
    bwd_total = row["Bwd_Bytes_Total"]
    fwd_total = row["Fwd_Bytes_Total"]
    req_len = int(fwd_total) if fwd_total > 0 else 60
    b_start = _epoch(row["Block_Start"])
    b_end = _epoch(row["Block_End"])
    
    label1 = "① 増幅ポート（" + port_name + "/" + str(port) + "）への全通信"
    label2 = "② 応答パケットのみ（増幅率 約" + format(amp, ".1f") + "倍、推定" + format(bwd_total, ".0f") + "バイト受信）"
    label3 = "③ 観測期間のみ"
    
    filters = {}
    filters[label1] = "ip.addr==" + src + " && ip.addr==" + dst + " && udp.port==" + str(port)
    filters[label2] = ("ip.src==" + dst + " && ip.dst==" + src + " && udp.srcport==" + str(port) +
                       " && udp.length>=" + str(req_len))
    filters[label3] = ("ip.addr==" + src + " && ip.addr==" + dst +
                       " && frame.time_epoch>=" + format(b_start, ".0f") +
                       " && frame.time_epoch<=" + format(b_end, ".0f"))
    return filters