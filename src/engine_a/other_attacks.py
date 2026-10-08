"""
src/engine_a/other_attacks.py
エンジンA：ルールベースによる「その他攻撃の疑い」判定
（v5.2 新規、v5.3 ベクトル化、v5.4 社外条件、v5.5 流量条件）。
【位置づけ】
  正解ラベル（Zeek）は実運用では得られないため、ポートスキャン（縦・水平）と同じく
  フローの統計的な振る舞いだけで判定する。C2・ポートスキャンのどちらにも断定できない
  通信を、「疑いあり（参考）」として提示し、調査の優先順位付けに使う。
  仕様書には記載のない拡張機能であり、CRITICAL/WARNINGのような断定的な表現は使わない。
【検知する3種類】
  1. フラッド（DoS）の疑い
  2. 分散アクセス（DDoS着弾）の疑い
  3. リフレクション/増幅の疑い
       DNS・NTP・SSDP等、増幅攻撃の踏み台として悪用されやすいポートへの通信で、
       応答のデータ量が要求のデータ量に対して著しく大きく、かつ短時間に集中している状態。
【v5.5 変更点：リフレクションの判定に「流量（フロー/秒）」の条件を追加】
  34-1（Mirai）で、端末192.168.1.195が社外のNTPサーバー（89.221.210.188）と、
  約17分おきに行った通常のNTP通信の1ブロックが、リフレクションの疑いと判定された（誤検知）。
  正解ラベルでは、この宛先の通信102件すべてが「ラベルなし」だった。
  誤検知の原因は、要求が合計8バイトと極端に小さく、割り算の結果の増幅率が1,228倍に
  なったことである。25フローが約3.3時間にわたっており、流量は0.002フロー/秒だった。
  実際のリフレクション攻撃は、短時間に大量のパケットが集中するため、
  Flow_Rate（フロー/秒）が一定値以上であることを条件に追加した（既定は1フロー/秒）。
  なお「要求バイト数が一定以上」という条件は採用しなかった。Echo 5-1のDNS
  （要求6,268バイト、増幅率11.4倍）のように、要求が大きくても誤検知が起きたため。
  【注意】1フロー/秒は、手元の誤検知2件（0.002と0.077）を除外し、
  攻撃では通常超える値として置いた初期値。本物のリフレクション攻撃を含むPCAPでの
  検証はできていない。
【v5.5 変更点：「社外」条件を、宛先だけでなく送信元も見る形に修正】
  v5.4 では「宛先（Top_Dst_IP）が社外」を条件にしていた。しかし、社内のサーバーが
  増幅の踏み台にされる場合は、送信元（要求を送る側）が社外で、宛先が社内になる。
  v5.4 のままだと、この場合を除外してしまう。
  そこで「送信元か宛先の少なくとも片方が社外」（社内と社外の通信）を条件にした。
  Echo 5-1のように、社内同士（192.168.2.3 → ルーター192.168.2.1）の通信は、
  従来どおり対象外になる。
【v5.3 変更点：build_dest_side_blocks のベクトル化】
  宛先ベースの集計は、size/nunique/min/max だけを使ってループなしで計算し、
  表示用の上位送信元IPは、しきい値を満たしたブロックにだけ attach_top_src_ips() で付ける。
  ループ版（_build_dest_side_blocks_loop）は、結果一致確認のためだけに残してある。
"""
import ipaddress
import numpy as np
import pandas as pd

FLOOD_RULE = dict(min_flows=10, max_dst_ips=2, port_share=0.9, min_rate=20.0)
DDOS_DEST_RULE = dict(min_flows=15, min_src_ips=15, max_window_sec=10.0)
REFLECTION_RULE = dict(min_amp_ratio=10.0, min_bytes=2000.0, min_flow_rate=1.0)
REFLECTION_PORTS = {53: "DNS", 123: "NTP", 1900: "SSDP", 19: "CharGEN",
                    161: "SNMP", 11211: "Memcached", 389: "CLDAP/LDAP", 1434: "MS-SQL"}

def is_external_ip(ip):
    """社外（グローバル）のIPなら True。
    社内（プライベート）・マルチキャスト・リンクローカル・ループバック・未指定・
    ブロードキャストは False。エンジンB（common.py の is_external）と同じ条件。
    IPアドレスとして読めない値も False とする。"""
    try:
        addr = ipaddress.ip_address(str(ip))
    except ValueError:
        return False
    if addr.is_private or addr.is_multicast or addr.is_link_local:
        return False
    if addr.is_loopback or addr.is_unspecified:
        return False
    return str(addr) != "255.255.255.255"

def _external_mask(ip_series):
    """IPアドレスの列から、「社外か」の真偽値の列を作る。
    同じIPを何度も判定しないよう、種類ごとに1回だけ判定して使い回す。"""
    cache = {}
    for ip in ip_series.unique():
        cache[ip] = is_external_ip(ip)
    return ip_series.map(cache).fillna(False).astype(bool)

def flood_flags(df_agg, min_flows=None, max_dst_ips=None, port_share=None, min_rate=None):
    """フラッド（DoS）の疑い：少数の宛先・同一ポートへの高頻度接続。"""
    p = dict(FLOOD_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, max_dst_ips=max_dst_ips,
                                     port_share=port_share, min_rate=min_rate).items() if v is not None})
    return ((df_agg["Observed_Flows"] >= p["min_flows"]) &
            (df_agg["Unique_Dst_IPs"] <= p["max_dst_ips"]) &
            (df_agg["Top_Port_Share"] >= p["port_share"]) &
            (df_agg["Flow_Rate"] >= p["min_rate"]))

def reflection_flags(df_agg, min_amp_ratio=None, min_bytes=None, min_flow_rate=None,
                     external_only=True):
    """リフレクション/増幅の疑い：既知の増幅ポートへの、応答過多で、短時間に集中した通信。
    条件（すべてを満たすブロック）：
      ・宛先ポートが既知の増幅ポート（DNS・NTP等）
      ・応答バイト数／要求バイト数 が min_amp_ratio 以上、応答が min_bytes 以上
      ・Flow_Rate（フロー/秒）が min_flow_rate 以上（v5.5）
      ・external_only=True（既定）のとき、送信元か宛先の少なくとも片方が社外（v5.5）
    """
    p = dict(REFLECTION_RULE)
    p.update({k: v for k, v in dict(min_amp_ratio=min_amp_ratio, min_bytes=min_bytes,
                                     min_flow_rate=min_flow_rate).items() if v is not None})
    is_reflection_port = df_agg["Top_Dst_Port"].isin(REFLECTION_PORTS.keys())
    cand = (is_reflection_port &
            (df_agg["Byte_Amplification_Ratio"] >= p["min_amp_ratio"]) &
            (df_agg["Bwd_Bytes_Total"] >= p["min_bytes"]) &
            (df_agg["Flow_Rate"] >= p["min_flow_rate"]))
    flags = cand.copy()
    if external_only and cand.any():
        # 条件を満たした少数のブロックだけ、社内外の通信かを確認する（全ブロックは調べない）
        src_ext = _external_mask(df_agg.loc[cand, "Src_IP"]).values
        dst_ext = _external_mask(df_agg.loc[cand, "Top_Dst_IP"]).values
        flags.loc[cand] = src_ext | dst_ext
    return flags

def build_dest_side_blocks(df_flows, window_size):
    """視点を反転：宛先IP:ポートごとに window_size フローずつ区切り、
    異なる送信元IPの数を数える（DDoS着弾＝分散アクセスの疑いの検出用）。ベクトル化版。
    【重要】水平スキャン型の通信（宛先IPがほぼ毎回異なる）では、
    groupby(["Dst_IP","Dst_Port"]) のグループ数が母数（フロー数）に近い規模まで
    膨れ上がる。この関数は numpy/pandas の組み込み集約（size/nunique/min/max）
    だけを使い、グループごとに呼ばれるPython関数（.agg(lambda ...) 等）を
    一切含まない設計にしている。表示用の「上位5送信元IP」は、ここでは計算せず、
    attach_top_src_ips() で、しきい値を満たした（＝ごく少数に絞られた）
    ブロックに対してだけ後から計算する。
    """
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
    """build_dest_side_blocks() の結果（通常はしきい値を満たした少数の行に絞った後）に、
    表示用の「上位送信元IP」を付与する。対象行数が少ないこと（DDoS着弾候補のみ）を
    前提に、対象ブロックだけをピンポイントで再抽出して計算する。
    全ブロックに対して行うと build_dest_side_blocks と同じ速度問題が再発するため、
    必ず ddos_dest_flags() 等で絞り込んだ後の dest_blocks に対して呼び出すこと。"""
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
    """【検証専用】旧実装（Pythonループ版）。ベクトル化版との結果一致確認のためだけに残す。
    戻り値には Top_Src_IPs も含む（新実装では attach_top_src_ips と組み合わせて比較する）。"""
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
    """分散アクセス（DDoS着弾）の疑い：短時間に多数の異なる送信元が集中。"""
    p = dict(DDOS_DEST_RULE)
    p.update({k: v for k, v in dict(min_flows=min_flows, min_src_ips=min_src_ips,
                                     max_window_sec=max_window_sec).items() if v is not None})
    return ((df_dest["Observed_Flows"] >= p["min_flows"]) &
            (df_dest["Unique_Src_IPs"] >= p["min_src_ips"]) &
            (df_dest["Window_Sec"] <= p["max_window_sec"]))

def _epoch(x):
    return float(x)

def flood_filters(row):
    """フラッド（DoS）の疑いに対するフィルタ群。row は flood_flags==True の1行。"""
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
                       " && frame.time_epoch>=" + format(b_start - 0.001, ".3f") +
                       " && frame.time_epoch<=" + format(b_end + 0.001, ".3f"))
    if proto == "tcp":
        label4 = "④ SYNのみ（コネクション確立前の接続試行を抽出）"
        filters[label4] = "ip.src==" + src + " && ip.dst==" + dst + " && tcp.flags.syn==1 && tcp.flags.ack==0"
    return filters

def ddos_dest_filters(row):
    """分散アクセス（DDoS着弾）の疑いに対するフィルタ群。row は ddos_dest_flags==True の1行
    （attach_top_src_ips 適用後、Top_Src_IPs が埋まっている前提）。"""
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
                       " && frame.time_epoch>=" + format(b_start - 0.001, ".3f") +
                       " && frame.time_epoch<=" + format(b_end + 0.001, ".3f"))
    src_list = [ip for ip in row["Top_Src_IPs"].split(", ") if ip]
    filters[label3] = " || ".join("ip.src==" + ip for ip in src_list)
    return filters

def reflection_filters(row):
    """リフレクション/増幅の疑いに対するフィルタ群。row は reflection_flags==True の1行。"""
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
                       " && frame.time_epoch>=" + format(b_start - 0.001, ".3f") +
                       " && frame.time_epoch<=" + format(b_end + 0.001, ".3f"))
    return filters
