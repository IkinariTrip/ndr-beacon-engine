"""
src/engine_a/cross_check.py
エンジンAとエンジンBの結果を突き合わせる処理（v5.6 新規）。
【目的：エンジンBがDoSをC2と誤判定する問題への対策】
  7-1（Mirai）で、エンジンBが 64.237.233.111:80/udp をC2確率0.91でCRITICALと判定した。
  正解ラベルではDoS攻撃（39,581件）の宛先で、通信間隔が約0.0秒の高頻度通信を
  C2のリズムと誤認したものだった。
  同じ宛先を、エンジンAは「フラッドの疑い」として検知している。
  そこで、エンジンAがフラッドの疑いとした「送信元・宛先IP・宛先ポート」の組み合わせが、
  エンジンBの通信ペア（社内側IP・相手IP・相手ポート）と一致した場合に、
  そのペアへ印（a_flood_match）を付ける。
  印を付けたCRITICALを、WARNINGへ引き下げる処理は triage.apply_triage が行う。
【設計上の注意】
  ・印を付けるだけで、ペアの削除や、C2確率の書き換えはしない（判断の根拠を残す）。
  ・一致の条件は「送信元IP・宛先IP・宛先ポート」。プロトコル（TCP/UDP）は見ない。
  ・エンジンAのフラッド判定は、1秒あたり20フロー以上の高頻度通信だけを拾う。
    C2のビーコンはこの頻度にならないため、本物のC2を巻き込む可能性は低い（未検証）。
  ・エンジンAのフラッド判定が働かなかったDoS（低頻度のもの）は、引き下げの対象外。
"""
import pandas as pd

def _flood_keys(attack_blocks):
    """エンジンAがフラッドの疑いとしたブロックから、（送信元IP, 宛先IP, 宛先ポート）の集合を作る。"""
    if attack_blocks is None or len(attack_blocks) == 0:
        return set()
    for col in ("Flood", "Src_IP", "Top_Dst_IP", "Top_Dst_Port"):
        if col not in attack_blocks.columns:
            return set()
    flood = attack_blocks[attack_blocks["Flood"].fillna(False).astype(bool)]
    if len(flood) == 0:
        return set()
    ports = pd.to_numeric(flood["Top_Dst_Port"], errors="coerce").fillna(-1).astype(int)
    return set(zip(flood["Src_IP"].astype(str), flood["Top_Dst_IP"].astype(str), ports))

def mark_flood_matches(summary_b, attack_blocks):
    """エンジンBの通信ペアの表に、列 a_flood_match（True/False）を付けて返す。
    元の表は変更しない。"""
    out = summary_b.copy()
    keys = _flood_keys(attack_blocks)
    need = ("host_ip", "peer_ip", "peer_port")
    if len(out) == 0 or len(keys) == 0 or any(c not in out.columns for c in need):
        out["a_flood_match"] = False
        return out
    ports = pd.to_numeric(out["peer_port"], errors="coerce").fillna(-1).astype(int)
    flags = []
    for host, peer, port in zip(out["host_ip"], out["peer_ip"], ports):
        flags.append((str(host), str(peer), int(port)) in keys)
    out["a_flood_match"] = flags
    return out
