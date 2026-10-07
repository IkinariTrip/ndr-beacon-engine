import pandas as pd
path = "src/engine_b/triage.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

START = "def enrich_c2_filters("
END = "def apply_flood_downgrade("
if "_c2_iat_filter" in text:
    print("すでに適用済みです。triage.py は変更していません。")
elif text.count(START) != 1 or text.count(END) != 1 or text.find(START) > text.find(END):
    print("適用できませんでした。triage.py は変更していません。")
else:
    new_func = '''def _c2_iat_filter(pair_port, proto, iat_median, iat_mad):
    """通信間隔の確認用フィルタ（ラベルとフィルタ文字列）を返す。
    間隔がほぼ0秒のときは範囲が逆転しないよう、上限だけの条件にする。"""
    if iat_median < 0.05:
        label = "④ 周期性の確認用（②適用後に追加。通信間隔 ほぼ0秒＝連続送信。0.05秒以下のみ表示）"
        return label, pair_port + " && frame.time_delta_displayed<=0.05"
    mad = iat_mad if (iat_mad is not None and not pd.isna(iat_mad)) else 0.0
    mad = max(mad, iat_median * 0.1)  # ばらつきが0でも、幅を持たせる
    lo = max(iat_median - mad, 0.01)
    hi = iat_median + mad
    digits = ".2f" if mad < 1 else ".1f"
    label = ("④ 周期性の確認用（②適用後に追加。通信間隔 約" + format(iat_median, ".1f") +
             "秒±" + format(mad, digits) + "秒）")
    flt = (pair_port + " && frame.time_delta_displayed>=" + format(lo, ".2f") +
           " && frame.time_delta_displayed<=" + format(hi, ".2f"))
    return label, flt

def enrich_c2_filters(base_filters, host_ip, peer_ip, peer_port, proto,
                      iat_median=None, iat_mad=None, byte_ratio_median=None):
    """C2疑いペアの代表ブロックの特徴量を使い、調査に役立つフィルタを2つ追加する。
    ⑤は、TCPなら tcp.len、UDPなら udp.length（ヘッダ8バイトを除く）でペイロードの有無を見る。"""
    result = dict(base_filters)
    proto = str(proto).strip().lower()
    pair_port = ("ip.addr==" + str(host_ip) + " && ip.addr==" + str(peer_ip) +
                 " && " + proto + ".port==" + str(int(peer_port)))
    if iat_median is not None and not pd.isna(iat_median):
        label4, flt4 = _c2_iat_filter(pair_port, proto, float(iat_median), iat_mad)
        result[label4] = flt4
    if byte_ratio_median is not None and not pd.isna(byte_ratio_median):
        label5 = "⑤ 応答の有無を確認（送信比率 約" + format(byte_ratio_median, ".0%") + "。ペイロードありのみ抽出）"
        if proto == "udp":
            result[label5] = pair_port + " && udp.length>8"
        else:
            result[label5] = pair_port + " && tcp.len>0"
    return result

'''
    i, j = text.find(START), text.find(END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:i] + new_func + text[j:])
    print("triage.py を更新しました。")
