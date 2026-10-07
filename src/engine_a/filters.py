"""
src/engine_a/filters.py
エンジンA：縦スキャン・水平スキャンの Wireshark 表示フィルタ生成。
v5.7 の修正：
  ・縦スキャンの⑤：旧版は「スキャンする側が送ったRST」を条件にしており、ラベルの
    「応答のない接続」と合っていなかった。相手（宛先）側から返る拒否応答（RST）に変更した。
  ・水平スキャンの③：旧版は②と同じ意味の条件（SYNのみ）に、必ず成り立つ否定条件を
    付けただけで、「応答のない宛先」を絞り込めていなかった。表示フィルタでは「応答が
    無いこと」を直接書けないため、応答があった宛先（SYN+ACK）だけを出す形に変更した。
    その件数が少ないほど、無応答率が高い（スキャンが空振りしている）ことを確認できる。
  ・水平スキャンで、TCPの特徴（SYN）がないブロックはUDPとして扱うようにした。
"""

def _epoch(x):
    return float(x)

def vertical_filters(row):
    """縦スキャン（1台の相手に多数のポート）の疑いに対するフィルタ群。"""
    src = row["Src_IP"]
    dst = row["Top_Dst_IP"]
    n_ports = int(row["Unique_Dst_Ports"])
    n_flows = row["Observed_Flows"]
    b_start = _epoch(row["Block_Start"])
    b_end = _epoch(row["Block_End"])
    rate = row.get("Flow_Rate", None)
    label1 = "① 送信元の全通信（" + str(n_flows) + "フロー、宛先ポート種類 " + str(n_ports) + "）"
    label2 = "② 主な宛先" + str(dst) + "への通信のみ"
    label3 = "③ SYNのみ（接続試行の一覧化。ポート走査順の確認に有効）"
    label4 = "④ 観測期間のみ"
    label5 = "⑤ 相手側からの拒否応答（RST）のみ（返ってきた数＝閉じていたポートの数）"
    if rate is not None:
        label5 = "⑤ 相手側からの拒否応答（RST）のみ（走査の速さ 推定" + format(rate, ".1f") + "フロー/秒。閉じていたポートの数を確認）"
    filters = {}
    filters[label1] = "ip.src==" + src
    filters[label2] = "ip.src==" + src + " && ip.dst==" + dst
    filters[label3] = "ip.src==" + src + " && ip.dst==" + dst + " && tcp.flags.syn==1 && tcp.flags.ack==0"
    filters[label4] = ("ip.src==" + src +
                       " && frame.time_epoch>=" + format(b_start, ".0f") +
                       " && frame.time_epoch<=" + format(b_end, ".0f"))
    filters[label5] = "ip.src==" + dst + " && ip.dst==" + src + " && tcp.flags.reset==1"
    return filters

def horizontal_filters(row):
    """水平スキャン（同一ポートを多数の相手に）の疑いに対するフィルタ群。"""
    src = row["Src_IP"]
    port = int(row["Top_Dst_Port"])
    n_ips = int(row["Unique_Dst_IPs"])
    ratio = row["Dst_IP_Ratio"]
    no_ret = row["No_Data_Return_Ratio"]
    b_start = _epoch(row["Block_Start"])
    b_end = _epoch(row["Block_End"])
    is_tcp = row.get("SYN_Flag_Ratio", 1.0) > 0
    label1 = "① 送信元の全通信（宛先IP " + str(n_ips) + "種、重複なし率 " + format(ratio, ".2f") + "）"
    label4 = "④ 観測期間のみ"
    filters = {}
    filters[label1] = "ip.src==" + src
    if is_tcp:
        label2 = "② 主な宛先ポート" + str(port) + "/tcpへの接続試行（SYNのみ）"
        label3 = ("③ 応答があった宛先のみ（SYN+ACK。無応答率 " + format(no_ret, ".2f") +
                  " のため、少数なら空振りのスキャン。多ければスキャンが成功した宛先なので要確認）")
        filters[label2] = ("ip.src==" + src + " && tcp.dstport==" + str(port) +
                           " && tcp.flags.syn==1 && tcp.flags.ack==0")
        filters[label3] = ("ip.dst==" + src + " && tcp.srcport==" + str(port) +
                           " && tcp.flags.syn==1 && tcp.flags.ack==1")
    else:
        label2 = "② 主な宛先ポート" + str(port) + "/udpへの送信"
        label3 = ("③ 応答があった宛先のみ（宛先ポートからのUDP応答。無応答率 " + format(no_ret, ".2f") +
                  " のため、少数なら空振りのスキャン）")
        filters[label2] = "ip.src==" + src + " && udp.dstport==" + str(port)
        filters[label3] = "ip.dst==" + src + " && udp.srcport==" + str(port)
    filters[label4] = ("ip.src==" + src +
                       " && frame.time_epoch>=" + format(b_start, ".0f") +
                       " && frame.time_epoch<=" + format(b_end, ".0f"))
    return filters
