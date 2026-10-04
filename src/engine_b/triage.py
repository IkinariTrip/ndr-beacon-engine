"""
triage.py
エンジンB 判定結果の多段トリアージ（仕様書v2.4 6章・7章、v2.5 6章）。
src/engine_b/ に配置する。表示（Streamlit）に依存しない純粋な処理だけを置く。

  1. 4段階トリアージ  : CRITICAL / WARNING / SAFE / FILTERED
  2. 補助IOC照合      : Suspect Port（443/80/53以外で 8080・8888・8443・5555・7777）
                        ※ 判定レベルは変えず、タグとして表示する
                        ※ CS Stager照合は、HTTPのURI抽出（tshark項目追加）が必要なため未実装
  3. Wiresharkフィルタ: 用途別3段階（①ペア全体 ②ペア＋ポート ③ペア＋観測期間）
  4. 判定根拠        : CatBoost自身のSHAP値（get_feature_importance）で上位2特徴量を出す
"""
import numpy as np
import pandas as pd

CRITICAL_TH = 0.85          # v2.4 6章（v2.5 6章：フェーズA後に再調整予定の暫定値）
SAFE_TH = 0.35
SAFE_MAX_BYTES = 5 * 1024 * 1024   # 5MB
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
    """1ペアの区分を返す。total_bytes が無い場合はバイト条件を使わない。"""
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


def wireshark_filters(host_ip, peer_ip, peer_port, proto, first_epoch, last_epoch):
    pair = f"ip.addr=={host_ip} && ip.addr=={peer_ip}"
    return {
        "① ペア全体": pair,
        "② ペア＋ポート": f"{pair} && {proto}.port=={int(peer_port)}",
        "③ ペア＋観測期間": (f"{pair} && {proto}.port=={int(peer_port)} && "
                          f"frame.time_epoch >= {first_epoch:.0f} && frame.time_epoch <= {last_epoch:.0f}"),
    }


def _epoch(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    return pd.Timestamp(x).timestamp()


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
        s["suspect_port"] = s["peer_port"].map(suspect_port)
        s["filters"] = [wireshark_filters(r.host_ip, r.peer_ip, r.peer_port, r.proto,
                                          _epoch(r.first_seen), _epoch(r.last_seen))
                        for r in s.itertuples()]
        s["level"] = pd.Categorical(s["level"], LEVEL_ORDER, ordered=True)
        s = s.sort_values(["level", "max_c2_proba"], ascending=[True, False]).reset_index(drop=True)
    counts = {lv: int((s["level"] == lv).sum()) if len(s) else 0 for lv in LEVEL_ORDER[:3]}
    counts["FILTERED"] = int(n_filtered_pairs)
    return s, counts


def top_reasons(blocks, model, meta, k=2):
    """各ブロックの判定根拠（C2寄りに効いた上位k特徴量）を、CatBoostのSHAP値で求める。"""
    from catboost import Pool
    cols = meta["features"]
    X = blocks[cols].replace([np.inf, -np.inf], np.nan).fillna(pd.Series(meta["fill_medians"]))
    sv = np.asarray(model.get_feature_importance(data=Pool(X[cols].values), type="ShapValues"))[:, :-1]
    out = []
    for row, vals in zip(X[cols].values, sv):
        order = np.argsort(-vals)[:k]
        out.append("、".join(f"{FEATURE_JA.get(cols[i], cols[i])}（{row[i]:.2f}）+{vals[i]:.2f}"
                            for i in order if vals[i] > 0) or "C2寄りの特徴なし")
    return out
