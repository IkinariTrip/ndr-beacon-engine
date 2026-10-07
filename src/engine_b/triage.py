"""
triage.py
エンジンB 判定結果の多段トリアージ（仕様書v2.4 6章・7章、v2.5 6章）。
v5.2：C2のWiresharkフィルタに「周期性の確認用」「応答有無の確認用」の2種類を追加した。
v5.6：エンジンAのフラッドの疑いと宛先が一致するCRITICALを、WARNINGへ引き下げる
      （DoSをC2と誤判定する問題への対策。列 a_flood_match は cross_check.py が付ける）。
"""
import numpy as np
import pandas as pd

CRITICAL_TH = 0.85
SAFE_TH = 0.35
SAFE_MAX_BYTES = 5 * 1024 * 1024
EXFIL_BYTES = 5 * 1024 * 1024
EXFIL_BURST = 0.8
SUSPECT_PORTS = {8080, 8888, 8443, 5555, 7777}
STANDARD_PORTS = {443, 80, 53}

LEVEL_ORDER = ["CRITICAL", "WARNING", "SAFE", "FILTERED"]
FEATURE_JA = {
    "IAT_Median": "通信間隔", "IAT_MAD": "間隔のばらつき", "IAT_CV": "間隔のゆらぎ",
    "Periodic_Peak_Ratio": "周期の強さ", "Connection_Count": "接続回数",
}

def triage_level(max_proba, total_bytes=None, burst_ratio=None):
    has_bytes = total_bytes is not None and not pd.isna(total_bytes)
    if has_bytes and total_bytes >= EXFIL_BYTES and burst_ratio is not None \
            and not pd.isna(burst_ratio) and burst_ratio > EXFIL_BURST:
        return "WARNING", "大量送受信かつ継続的（データ持ち出しの疑い）"
    if max_proba >= CRITICAL_TH:
        return "CRITICAL", "C2確率が0.85以上"
    if max_proba < SAFE_TH and (not has_bytes or total_bytes < SAFE_MAX_BYTES):
        return "SAFE", "C2確率が0.35未満"
    return "WARNING", "C2確率が0.35〜0.85"

def suspect_port(port):
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False
    return port not in STANDARD_PORTS and port in SUSPECT_PORTS

def _epoch(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    return pd.Timestamp(x).timestamp()

def base_wireshark_filters(host_ip, peer_ip, peer_port, proto, first_epoch, last_epoch):
    """従来からある3段階（ペア全体／ペア＋ポート／ペア＋観測期間）。"""
    pair = "ip.addr==" + str(host_ip) + " && ip.addr==" + str(peer_ip)
    with_port = pair + " && " + str(proto) + ".port==" + str(int(peer_port))
    filters = {}
    filters["① ペア全体"] = pair
    filters["② ペア＋ポート"] = with_port
    filters["③ ペア＋観測期間"] = (with_port + " && frame.time_epoch >= " + format(first_epoch, ".0f") +
                              " && frame.time_epoch <= " + format(last_epoch, ".0f"))
    return filters

def _c2_iat_filter(pair_port, proto, iat_median, iat_mad):
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

def apply_flood_downgrade(s):
    """列 a_flood_match が True のCRITICALを、WARNINGへ引き下げる。
    s は level が文字列の状態の表。印の列が無い場合は何もしない。"""
    if len(s) == 0 or "a_flood_match" not in s.columns:
        return s
    matched = s["a_flood_match"].fillna(False).astype(bool) & (s["level"] == "CRITICAL")
    if not matched.any():
        return s
    reasons = []
    for proba in s.loc[matched, "max_c2_proba"]:
        reasons.append("エンジンAが「フラッドの疑い」とした宛先と一致（C2確率" + format(proba, ".2f") +
                       "）。高頻度のDoSをC2のリズムと誤認した可能性があるため、CRITICALからWARNINGへ引き下げ")
    s.loc[matched, "level"] = "WARNING"
    s.loc[matched, "level_reason"] = reasons
    return s

def apply_triage(summary, n_filtered_pairs=0):
    """summarize() の結果（ペア単位）に、区分・理由・IOC・3段階フィルタを付ける。"""
    s = summary.copy()
    if len(s) == 0:
        s = pd.DataFrame(columns=list(s.columns) + ["level", "level_reason", "suspect_port", "filters"])
    else:
        tb = s["total_bytes"] if "total_bytes" in s else pd.Series([None] * len(s), index=s.index)
        br = s["burst_ratio"] if "burst_ratio" in s else pd.Series([None] * len(s), index=s.index)
        res = [triage_level(p, b, r) for p, b, r in zip(s["max_c2_proba"], tb, br)]
        s["level"] = [r[0] for r in res]
        s["level_reason"] = [r[1] for r in res]
        s = apply_flood_downgrade(s)
        s["suspect_port"] = s["peer_port"].map(suspect_port)
        s["filters"] = [base_wireshark_filters(r.host_ip, r.peer_ip, r.peer_port, r.proto,
                                               _epoch(r.first_seen), _epoch(r.last_seen))
                        for r in s.itertuples()]
        s["level"] = pd.Categorical(s["level"], LEVEL_ORDER, ordered=True)
        s = s.sort_values(["level", "max_c2_proba"], ascending=[True, False]).reset_index(drop=True)

    counts = {lv: int((s["level"] == lv).sum()) if len(s) else 0 for lv in LEVEL_ORDER[:3]}
    counts["FILTERED"] = int(n_filtered_pairs)
    return s, counts

def _fmt_reason(name, value, shap_value):
    return FEATURE_JA.get(name, name) + "（" + format(value, ".2f") + "）+" + format(shap_value, ".2f")

def top_reasons(blocks, model, meta, k=2):
    """各ブロックの判定根拠（C2寄りに効いた上位k特徴量）を、CatBoostのSHAP値で求める。"""
    from catboost import Pool
    cols = meta["features"]
    X = blocks[cols].replace([np.inf, -np.inf], np.nan).fillna(pd.Series(meta["fill_medians"]))
    sv = np.asarray(model.get_feature_importance(data=Pool(X[cols].values), type="ShapValues"))[:, :-1]
    out = []
    for row, vals in zip(X[cols].values, sv):
        order = np.argsort(-vals)[:k]
        parts = [_fmt_reason(cols[i], row[i], vals[i]) for i in order if vals[i] > 0]
        out.append("、".join(parts) if parts else "C2寄りの特徴なし")
    return out
