def _epoch(x):
    return float(x)

def vertical_filters(row):
    """縦スキャン（1台の相手に多数のポート）の疑いに対するフィルタ群。"""
    src, dst = row["Src_IP"], row["Top_Dst_IP"]
    n_ports = int(row["Unique_Dst_Ports"])
    rate = row["Flow_Rate"] if "Flow_Rate" in row else None
    
    filters = {
        f"① 送信元の全通信（{row['Observed_Flows']}フロー、宛先ポート種類 {n_ports}）":
            f"ip.src=={src}",
        f"② 主な宛先{dst}への通信のみ":
            f"ip.src=={src} && ip.dst=={dst}",
        "③ SYNのみ（接続試行の一覧化。ポート走査順の確認に有効）":
            f"ip.src=={src} && ip.dst=={dst} && tcp.flags.syn==1 && tcp.flags.ack==0",
        "④ 観測期間のみ":
            f"ip.src=={src} && "
            f"frame.time_epoch>={_epoch(row['Block_Start']):.0f} && "
            f"frame.time_epoch<={_epoch(row['Block_End']):.0f}",
    }
    
    if rate is not None:
        filters[f"⑤ 応答のない接続のみ（走査の典型パターン、推定{rate:.1f}フロー/秒）"] = (
            f"ip.src=={src} && ip.dst=={dst} && tcp.flags.reset==1 || " 
            f"ip.src=={src} && ip.dst=={dst} && tcp.flags.syn==1 && tcp.analysis.lost_segment"
        )
        
    return filters

def horizontal_filters(row):
    """水平スキャン（同一ポートを多数の相手に）の疑いに対するフィルタ群。"""
    src = row["Src_IP"]
    port = int(row["Top_Dst_Port"])
    proto = "tcp"
    n_ips = int(row["Unique_Dst_IPs"])
    ratio = row["Dst_IP_Ratio"]
    
    return {
        f"① 送信元の全通信（宛先IP {n_ips}種、重複なし率 {ratio:.2f}）":
            f"ip.src=={src}",
        f"② 主な宛先ポート{port}/{proto}への接続試行（SYNのみ）":
            f"ip.src=={src} && {proto}.dstport=={port} && tcp.flags.syn==1 && tcp.flags.ack==0",
        f"③ 応答のない宛先のみ（無応答率 {row['No_Data_Return_Ratio']:.2f}）":
            f"ip.src=={src} && {proto}.dstport=={port} && tcp.flags.syn==1 && tcp.flags.ack==0 && "
            f"!(tcp.flags.syn==1 && tcp.flags.ack==1)",
        "④ 観測期間のみ":
            f"ip.src=={src} && "
            f"frame.time_epoch>={_epoch(row['Block_Start']):.0f} && "
            f"frame.time_epoch<={_epoch(row['Block_End']):.0f}",
    }