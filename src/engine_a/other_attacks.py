"""
src/engine_a/other_attacks.py
エンジンA：ルールベースによる「その他攻撃の疑い」判定（v5.2 新規）。

【位置づけ】
  正解ラベル（Zeek）は実運用では得られないため、ポートスキャン（縦・水平）と同じく
  フローの統計的な振る舞いだけで判定する。C2・ポートスキャンのどちらにも断定できない
  通信を、「疑いあり（参考）」として提示し、調査の優先順位付けに使う。
  仕様書には記載のない拡張機能であり、CRITICAL/WARNINGのような断定的な表現は使わない。

【検知する3種類】
  1. フラッド（DoS）の疑い
       送信元1台が、ごく少数の宛先（1〜2台）の同一ポートに対して、
       異常に高い頻度で接続を試みている状態。
  2. 分散アクセス（DDoS着弾）の疑い
       視点を反転し、1つの宛先IP:ポートに、短時間で非常に多くの
       異なる送信元IPからアクセスが集中している状態。
  3. リフレクション/増幅の疑い
       DNS・NTP・SSDP等、増幅攻撃の踏み台として悪用されやすいポートへの通信で、
       応答のデータ量が要求のデータ量に対して著しく大きい状態。

【しきい値の根拠】
  いずれも初期値であり、実データでの評価（eval_engine_a_horizontal.py と同様の
  手法）を行った上で、必要に応じて見直すことを前提とする。
"""
import numpy as np
import pandas as pd

FLOOD_RULE = dict(min_flows=10, max_dst_ips=2, port_share=0.9, min_rate=20.0)
DDOS_DEST_RULE = dict(min_flows=15, min_src_ips=15, max_window_sec=10.0)
REFLECTION_RULE = dict(min_amp_ratio=10.0, min_bytes=2000.0)
REFLECTION_PORTS = {53: "DNS", 123: "NTP", 1900: "SSDP", 19: "CharGEN",
                    161: "SNMP", 11211: "Memcached", 389: "CLDAP/LDAP", 1434: "MS-SQL"}


def flood_flags(df_agg, min_flows=None, max_dst_ips=None, port_share=None, min_rate=None):
    """フラッド（DoS）の疑い：少数の宛先・同一ポートへの高頻度接続。"""
    p = dict(FLOOD_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, max_dst_ips=max_dst_ips,
                                     port_share=port_share, min_rate=min_rate).items() if v is not None})
    return ((df_agg["Observed_Flows"] >= p["min_flows"]) &
            (df_agg["Unique_Dst_IPs"] <= p["max_dst_ips"]) &
            (df_agg["Top_Port_Share"] >= p["port_share"]) &
            (df_agg["Flow_Rate"] >= p["min_rate"]))


def reflection_flags(df_agg, min_amp_ratio=None, min_bytes=None):
    """リフレクション/増幅の疑い：既知の増幅ポートへの、応答過多な通信。"""
    p = dict(REFLECTION_RULE)
    p.update({k: v for k, v in dict(min_amp_ratio=min_amp_ratio, min_bytes=min_bytes).items()
             if v is not None})
    is_reflection_port = df_agg["Top_Dst_Port"].isin(REFLECTION_PORTS.keys())
    return (is_reflection_port &
            (df_agg["Byte_Amplification_Ratio"] >= p["min_amp_ratio"]) &
            (df_agg["Bwd_Bytes_Total"] >= p["min_bytes"]))


def build_dest_side_blocks(df_flows, window_size):
    """視点を反転：宛先IP:ポートごとに window_size フローずつ区切り、
    異なる送信元IPの数を数える（DDoS着弾＝分散アクセスの疑いの検出用）。"""
    rows = []
    key_cols = ["Dst_IP", "Dst_Port"]
    for (dst_ip, dst_port), group in df_flows.groupby(key_cols):
        g = group.sort_values("Start_Time").reset_index(drop=True)
        for i in range(0, len(g), window_size):
            chunk = g.iloc[i:i + window_size]
            if len(chunk) == 0:
                continue
            start, end = float(chunk["Start_Time"].min()), float(chunk["End_Time"].max())
            rows.append(dict(
                Dst_IP=dst_ip, Dst_Port=int(dst_port), Observed_Flows=len(chunk),
                Unique_Src_IPs=chunk["Src_IP"].nunique(),
                Window_Sec=max(end - start, 1e-3),
                Block_Start=start, Block_End=end,
                Top_Src_IPs=", ".join(chunk["Src_IP"].value_counts().index[:5].tolist()),
            ))
    return pd.DataFrame(rows)


def ddos_dest_flags(df_dest, min_flows=None, min_src_ips=None, max_window_sec=None):
    """分散アクセス（DDoS着弾）の疑い：短時間に多数の異なる送信元が集中。"""
    p = dict(DDOS_DEST_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, min_src_ips=min_src_ips,
                                     max_window_sec=max_window_sec).items() if v is not None})
    return ((df_dest["Observed_Flows"] >= p["min_flows"]) &
            (df_dest["Unique_Src_IPs"] >= p["min_src_ips"]) &
            (df_dest["Window_Sec"] <= p["max_window_sec"]))


# ---------------------------------------------------------------------------
# Wireshark フィルタ生成（具体的な数値つき）
# ---------------------------------------------------------------------------
def _epoch(x):
    return float(x)


def flood_filters(row):
    """フラッド（DoS）の疑いに対するフィルタ群。row は flood_flags==True の1行。"""
    src, dst, port = row["Src_IP"], row["Top_Dst_IP"], int(row["Top_Dst_Port"])
    proto = "tcp" if row.get("SYN_Flag_Ratio", 0) > 0 else "udp"
    rate = row["Flow_Rate"]
    filters = {
        f"① 送信元→宛先の全通信（{row['Observed_Flows']}フロー）":
            f"ip.src=={src} && ip.dst=={dst}",
        f"② 宛先ポート{port}/{proto}に絞り込み":
            f"ip.src=={src} && ip.dst=={dst} && {proto}.port=={port}",
        f"③ 観測期間のみ（約{row['Block_Duration']:.1f}秒間、{rate:.1f}フロー/秒）":
            f"ip.src=={src} && ip.dst=={dst} && "
            f"frame.time_epoch>={_epoch(row['Block_Start']):.0f} && "
            f"frame.time_epoch<={_epoch(row['Block_End']):.0f}",
    }
    if proto == "tcp":
        filters[f"④ SYNのみ（コネクション確立前の接続試行を抽出）"] = \
            f"ip.src=={src} && ip.dst=={dst} && tcp.flags.syn==1 && tcp.flags.ack==0"
    return filters


def ddos_dest_filters(row):
    """分散アクセス（DDoS着弾）の疑いに対するフィルタ群。row は ddos_dest_flags==True の1行。"""
    dst, port = row["Dst_IP"], int(row["Dst_Port"])
    return {
        f"① 宛先への全通信（異なる送信元IP {row['Unique_Src_IPs']}件・{row['Observed_Flows']}フロー）":
            f"ip.dst=={dst} && tcp.port=={port} || ip.dst=={dst} && udp.port=={port}",
        f"② 観測期間のみ（約{row['Window_Sec']:.1f}秒間に集中）":
            f"ip.dst=={dst} && "
            f"frame.time_epoch>={_epoch(row['Block_Start']):.0f} && "
            f"frame.time_epoch<={_epoch(row['Block_End']):.0f}",
        "③ 上位の送信元IP（最大5件、カンマ区切りを ||区切りに置換して使用）":
            " || ".join(f"ip.src=={ip}" for ip in row["Top_Src_IPs"].split(", ") if ip),
    }


def reflection_filters(row):
    """リフレクション/増幅の疑いに対するフィルタ群。row は reflection_flags==True の1行。"""
    src, dst, port = row["Src_IP"], row["Top_Dst_IP"], int(row["Top_Dst_Port"])
    port_name = REFLECTION_PORTS.get(port, str(port))
    amp = row["Byte_Amplification_Ratio"]
    return {
        f"① 増幅ポート（{port_name}/{port}）への全通信":
            f"ip.addr=={src} && ip.addr=={dst} && udp.port=={port}",
        f"② 応答パケットのみ（増幅率 約{amp:.1f}倍、推定{row['Bwd_Bytes_Total']:.0f}バイト受信）":
            f"ip.src=={dst} && ip.dst=={src} && udp.srcport=={port} && udp.length>={int(row['Fwd_Bytes_Total']) if row['Fwd_Bytes_Total'] > 0 else 60}",
        f"③ 観測期間のみ":
            f"ip.addr=={src} && ip.addr=={dst} && "
            f"frame.time_epoch>={_epoch(row['Block_Start']):.0f} && "
            f"frame.time_epoch<={_epoch(row['Block_End']):.0f}",
    }
