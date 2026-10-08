# triage.py の C2用フィルタ④⑤⑥を、実機（Wireshark）で動くものに差し替える
import pandas as pd
path = "src/engine_b/triage.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

START = "def _c2_iat_filter("
END = "def apply_flood_downgrade("

if "_c2_extra_filters" in text:
    print("すでに適用済みです。triage.py は変更していません。")
elif text.count(START) != 1 or text.count(END) != 1 or text.find(START) > text.find(END):
    print("適用できませんでした（目印の関数が1件ずつ見つかりません）。triage.py は変更していません。")
else:
    new_block = '''def _c2_extra_filters(host_ip, peer_ip, peer_port, proto, iat_median, iat_mad, byte_ratio_median):
    """C2疑いペアの調査用フィルタ（④周期性・⑤相手の応答・⑥データの有無）を作る。
    【実機確認で分かったこと】
    ・frame.time_delta_displayed は、表示フィルタの条件には使えない（常に0件になる）。
      そこで④は「ホストが送った接続試行・データ送信だけ」を表示し、間隔は
      Wiresharkの時刻表示を「直前の表示パケットからの経過時間」にして読み取る方式にした。
    ・7-1のC2は、接続試行（SYN）に対し相手がRSTで拒否しており、データは1バイトも流れない。
      そのため「データあり」の抽出は0件になるのが正しい結果。⑤で相手の応答を直接見られるようにした。
    """
    proto = str(proto).strip().lower()
    host, peer, port = str(host_ip), str(peer_ip), str(int(peer_port))
    out_pair = "ip.src==" + host + " && ip.dst==" + peer + " && " + proto + ".port==" + port
    in_pair = "ip.src==" + peer + " && ip.dst==" + host + " && " + proto + ".port==" + port
    both = "ip.addr==" + host + " && ip.addr==" + peer + " && " + proto + ".port==" + port
    # --- ④ 周期性の確認 ---
    if iat_median is None or pd.isna(iat_median):
        check = "Time列の値から通信間隔を確認"
    elif float(iat_median) < 0.05:
        check = "Time列の値がほぼ0秒（連続送信）か確認"
    else:
        mad = iat_mad if (iat_mad is not None and not pd.isna(iat_mad)) else 0.0
        check = ("Time列の値が約" + format(float(iat_median), ".1f") + "秒（±" +
                 format(float(mad), ".1f") + "秒）付近か確認")
    how = ("→ 表示後、［表示］→［時刻表示形式］→［直前の表示パケットからの経過時間］に切り替えて、" + check)
    if proto == "udp":
        label4 = "④ 周期性の確認用（ホストが送ったパケットのみ）" + how
        flt4 = out_pair
    else:
        label4 = "④ 周期性の確認用（ホストの接続試行・送信のみ）" + how
        flt4 = out_pair + " && ((tcp.flags.syn==1 && tcp.flags.ack==0) || tcp.len>0)"
    # --- ⑤ 相手の応答 ---
    if proto == "udp":
        label5 = "⑤ 相手からの応答のみ（0件なら応答なし）"
        flt5 = in_pair
    else:
        label5 = "⑤ 相手からの応答のみ（RST=接続拒否／SYN+ACK=接続成立。RSTだけなら相手は停止中の可能性）"
        flt5 = in_pair
    # --- ⑥ データの有無 ---
    if byte_ratio_median is not None and not pd.isna(byte_ratio_median):
        ratio = "送信比率 約" + format(float(byte_ratio_median), ".0%") + "。"
    else:
        ratio = ""
    if proto == "udp":
        label6 = "⑥ データのあるパケットのみ（" + ratio + "0件なら中身の送受信なし）"
        flt6 = both + " && udp.length>8"
    else:
        label6 = "⑥ データのあるパケットのみ（" + ratio + "0件なら中身の送受信なし）"
        flt6 = both + " && tcp.len>0"
    return [(label4, flt4), (label5, flt5), (label6, flt6)]

def enrich_c2_filters(base_filters, host_ip, peer_ip, peer_port, proto,
                      iat_median=None, iat_mad=None, byte_ratio_median=None):
    """C2疑いペアの代表ブロックの特徴量を使い、調査に役立つフィルタを3つ追加する（④⑤⑥）。"""
    result = dict(base_filters)
    for label, flt in _c2_extra_filters(host_ip, peer_ip, peer_port, proto,
                                        iat_median, iat_mad, byte_ratio_median):
        result[label] = flt
    return result
'''
    i, j = text.find(START), text.find(END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:i] + new_block + text[j:])
    print("triage.py を更新しました。")
