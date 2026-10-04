"""
app_v5.py（v5.1：エンジンAに水平スキャン検知を追加）
【仮称】インシデンス・コックピット改5：エンジンA（内部探索）×エンジンB（外部C2）の並列デュアル構成。
リポジトリ直下に配置し、streamlit run app_v5.py で起動する。

v5.0 からの変更点:
  - エンジンAの計算を src/engine_a/features.py に移した（既存8特徴量の計算は変更なし）
  - エンジンAに「水平スキャン」判定を追加
      縦スキャン ：既存の CatBoost v4（1台の相手に多数のポート）
      水平スキャン：新しいルール（同じポートを毎回違う相手に試し、ほぼ応答なし）
    どちらかに該当すればエンジンAの「スキャン」とし、種別を表示する
  - 相関判定は、縦・水平どちらのスキャンでもエンジンAの該当として扱う
"""
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.flow_generator import extract_flows_from_pcap
from src.engine_a.features import (flows_to_dataframe, build_blocks, horizontal_scan_flags,
                                   VERTICAL_FEATURES, H_RULE)

ENGINE_B_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b")
if ENGINE_B_DIR not in sys.path:
    sys.path.insert(0, ENGINE_B_DIR)
from inference import analyze_pcap, DEFAULT_MODEL as ENGINE_B_MODEL  # noqa: E402
from engine_b_view import render_engine_b_results, triage_pairs  # noqa: E402

st.set_page_config(page_title="【仮称】インシデンス・コックピット改5", page_icon="🛡️", layout="wide")
st.title("🛡️ 【仮称】インシデンス・コックピット改5（デュアルエンジン版）")
st.caption("エンジンA：内部ポートスキャン検知（縦：CatBoost v4／水平：ルール）／"
           "エンジンB：外部C2ビーコン検知（CatBoost・タイミング5特徴量）")

MODEL_PATH = "models/model_catboost_v4.joblib"


@st.cache_resource
def load_detection_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_names", [])
    return payload, []


model, trained_features = load_detection_model()
if model is None:
    st.error(f"⚠️ モデルファイルが見つかりません。'{MODEL_PATH}' を配置してください。")
    st.stop()

with st.sidebar:
    st.header("⚙️ 監視パラメータ")
    st.subheader("エンジンA（内部スキャン）")
    threshold = st.slider("縦スキャン：異常判定しきい値", 0.1, 0.9, 0.5, step=0.05)
    window_size = st.number_input("時間窓ブロックサイズ (フロー数)", min_value=5, max_value=100, value=25)
    st.caption("ブロックサイズの変更は、解析を再実行すると反映されます")
    use_h = st.checkbox("水平スキャン判定を有効にする", value=True)
    st.caption(f"水平スキャン：宛先IPの重複なし率≥{H_RULE['ip_ratio']}、同一ポート率≥{H_RULE['port_share']}、"
               f"応答データなし率≥{H_RULE['no_return']}")
    st.markdown("---")
    st.subheader("エンジンB（外部C2）")
    st.caption("4段階トリアージ：CRITICAL ≥0.85／WARNING／SAFE <0.35／FILTERED（通信10回未満）")
    st.markdown("---")
    st.markdown("""
**【改5】構成:**
- **エンジンA:** 空間（宛先ポート・宛先IPの散らばり）からスキャンを検知
- **エンジンB:** 時間（通信間隔のリズム）からC2を検知
- **相関判定:** 両方に該当する端末を最優先で提示
""")


def run_engine_a(pcap_path, window_size):
    flows, total_pkts = extract_flows_from_pcap(pcap_path)
    if not flows:
        return None, None, total_pkts
    df_flows = flows_to_dataframe(flows)
    if df_flows.empty:
        return df_flows, None, total_pkts
    df_agg = build_blocks(df_flows, window_size)
    use_cols = trained_features if trained_features else VERTICAL_FEATURES
    df_agg["Anomaly_Score"] = model.predict_proba(df_agg[use_cols])[:, 1]
    return df_flows, df_agg, total_pkts


def engine_a_hits(df_agg, threshold, use_h):
    """縦スキャン（モデル）・水平スキャン（ルール）の判定を付け、該当ブロックを返す。"""
    d = df_agg.copy()
    d["Vertical"] = d["Anomaly_Score"] >= threshold
    d["Horizontal"] = horizontal_scan_flags(d) if use_h else False
    d["Scan_Type"] = np.select([d["Vertical"] & d["Horizontal"], d["Vertical"], d["Horizontal"]],
                               ["縦＋水平", "縦スキャン", "水平スキャン"], default="")
    return d, d[d["Vertical"] | d["Horizontal"]]


def correlate(attack_blocks, pairs_b):
    if len(attack_blocks):
        a = (attack_blocks.groupby("Src_IP")
             .agg(A_type=("Scan_Type", lambda s: "／".join(sorted(set(s)))),
                  A_max_score=("Anomaly_Score", "max"), A_scan_blocks=("Scan_Type", "size"),
                  A_ports=("Top_Dst_Port", lambda s: ", ".join(map(str, pd.Series(s).value_counts().index[:3]))))
             .reset_index().rename(columns={"Src_IP": "ip"}))
    else:
        a = pd.DataFrame(columns=["ip", "A_type", "A_max_score", "A_scan_blocks", "A_ports"])

    if len(pairs_b):
        alert_b = pairs_b[pairs_b["level"].isin(["CRITICAL", "WARNING"])].copy()
        alert_b["level"] = alert_b["level"].astype(str)
        alert_b["peer"] = alert_b["peer_ip"] + ":" + alert_b["peer_port"].astype(str)
        b = (alert_b.groupby("host_ip")
             .agg(B_level=("level", lambda s: "CRITICAL" if (s == "CRITICAL").any() else "WARNING"),
                  B_max_c2=("max_c2_proba", "max"),
                  B_c2_peers=("peer", lambda s: ", ".join(s.head(5)) + (" ほか" if len(s) > 5 else "")))
             .reset_index().rename(columns={"host_ip": "ip"}))
    else:
        b = pd.DataFrame(columns=["ip", "B_level", "B_max_c2", "B_c2_peers"])

    m = a.merge(b, on="ip", how="outer")
    in_a, in_b = m["A_type"].notna(), m["B_level"].notna()
    m["priority"] = np.select([in_a & in_b, in_b & (m["B_level"] == "CRITICAL"), in_b, in_a],
                              ["① 最優先（スキャン＋C2）", "② C2の疑い（CRITICAL）",
                               "③ C2の疑い（WARNING）", "④ スキャンのみ"], default="")
    return m.sort_values(["priority", "B_max_c2", "A_scan_blocks"], ascending=[True, False, False]).reset_index(drop=True)


uploaded_file = st.file_uploader("解析対象のパケットキャプチャ (PCAP / PCAPNG) を選択", type=["pcap", "pcapng"])

if uploaded_file is not None:
    st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")
    if st.button("🚀 パケット解析・行動スクリーニング実行（エンジンA＋B）", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_pcap_path = tmp.name
        result = {"name": os.path.splitext(uploaded_file.name)[0], "window_size": int(window_size)}
        try:
            with st.spinner("エンジンA：フロー抽出・スキャン判定中…"):
                result["A"] = run_engine_a(tmp_pcap_path, int(window_size))
            with st.spinner("エンジンB：通信ペア化・C2判定中…（数十秒〜数分）"):
                try:
                    result["B"] = analyze_pcap(tmp_pcap_path, model_path=ENGINE_B_MODEL)
                except Exception as e:
                    result["B_error"] = f"{type(e).__name__}: {e}"
        finally:
            if os.path.exists(tmp_pcap_path):
                os.remove(tmp_pcap_path)
        st.session_state["result"] = result

res = st.session_state.get("result")
if res:
    df_flows, df_agg, total_pkts = res["A"]
    has_a = df_agg is not None and len(df_agg) > 0
    if has_a:
        df_agg, attack_blocks = engine_a_hits(df_agg, threshold, use_h)
    else:
        attack_blocks = pd.DataFrame()

    has_b = "B" in res
    if has_b:
        summary_b, blocks_b, stats_b = res["B"]
        pairs_b, counts_b = triage_pairs(summary_b, stats_b)
    else:
        pairs_b, counts_b = pd.DataFrame(), {"CRITICAL": 0, "WARNING": 0, "SAFE": 0, "FILTERED": 0}

    st.success(f"解析完了：{res['name']}（エンジンAブロックサイズ {res['window_size']}）")
    tab_all, tab_a, tab_b = st.tabs(["🧭 統合判定", "🔍 エンジンA：内部スキャン", "📡 エンジンB：外部C2"])

    with tab_all:
        corr = correlate(attack_blocks, pairs_b)
        top = corr[corr["priority"].str.startswith("①")]
        c = st.columns(4)
        c[0].metric("総解析パケット数", f"{total_pkts:,}")
        c[1].metric("A：スキャン送信元", f"{attack_blocks['Src_IP'].nunique() if len(attack_blocks) else 0} 台")
        c[2].metric("B：C2の疑い（CRITICAL）", f"{counts_b['CRITICAL']} ペア")
        c[3].metric("① 最優先端末", f"{len(top)} 台")

        if len(top):
            st.error("🚨 **最優先の侵害端末（スキャンと外部C2の両方に該当）：" + "、".join(top["ip"]) +
                     "**　→ 即時隔離を検討してください")
        elif len(corr):
            st.warning("両エンジンに同時に該当する端末はありません。下表の優先度順に確認してください。")
        else:
            st.success("どちらのエンジンでも要対応の端末は検出されませんでした。")
        if not has_b:
            st.warning(f"エンジンBは実行できませんでした（{res.get('B_error', '不明')}）。相関判定はエンジンAのみです。")

        if len(corr):
            show = corr[["priority", "ip", "A_type", "A_scan_blocks", "A_ports", "B_level", "B_max_c2", "B_c2_peers"]]
            st.dataframe(show.rename(columns={
                "priority": "優先度", "ip": "端末IP", "A_type": "A：スキャン種別", "A_scan_blocks": "A：該当ブロック数",
                "A_ports": "A：主な宛先ポート", "B_level": "B：区分", "B_max_c2": "B：最大C2確率",
                "B_c2_peers": "B：C2の疑いがある相手"}).round(3), hide_index=True)
            if len(top):
                st.markdown("**Wireshark表示フィルタ（最優先端末の全通信）**")
                st.code(" || ".join(f"ip.addr == {ip}" for ip in top["ip"]), language="text")
        st.caption("優先度：① 両エンジンに該当 ＞ ② C2 CRITICAL ＞ ③ C2 WARNING ＞ ④ スキャンのみ。"
                   "エンジンAは送信元IP、エンジンBは社内側IPで判定しています。")

    with tab_a:
        if not has_a:
            st.warning("有効なTCP/UDPフローが検出されませんでした。")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("総解析パケット数", f"{total_pkts:,} パック")
            col2.metric("生成フロー数", f"{len(df_flows):,} の流れ")
            col3.metric("縦スキャン ブロック", f"{int(df_agg['Vertical'].sum()):,}")
            col4.metric("水平スキャン ブロック", f"{int(df_agg['Horizontal'].sum()):,}")
            st.markdown("---")
            st.subheader("🚨 侵害端末候補スクリーニング結果")
            if not attack_blocks.empty:
                for stype, g in attack_blocks.groupby("Scan_Type"):
                    st.error(f"🚨 **{stype}の送信元: {', '.join(g['Src_IP'].unique())}**")
                st.write("##### 異常判定ブロック一覧")
                display_cols = ["Src_IP", "Scan_Type", "Anomaly_Score", "Unique_Dst_Ports", "Unique_Dst_IPs",
                                "Top_Dst_Port", "Dst_IP_Ratio", "Top_Port_Share", "No_Data_Return_Ratio",
                                "Zero_Payload_Ratio", "SYN_Flag_Ratio", "Observed_Flows"]
                st.dataframe(attack_blocks[display_cols].sort_values(["Scan_Type", "Anomaly_Score"],
                                                                     ascending=[True, False]).round(3))
                st.write("##### 🔍 調査用 Wireshark フィルタ")
                v_ips = attack_blocks.loc[attack_blocks["Vertical"], "Src_IP"].unique()
                h = attack_blocks[attack_blocks["Horizontal"]]
                if len(v_ips):
                    st.caption("縦スキャン（送信元の全通信）")
                    st.code(" || ".join(f"ip.src == {ip}" for ip in v_ips), language="bash")
                if len(h):
                    st.caption("水平スキャン（送信元 × 主な宛先ポート、SYNのみ）")
                    pairs = h.groupby("Src_IP")["Top_Dst_Port"].agg(lambda s: sorted(set(s))[:3])
                    st.code(" || ".join(f"(ip.src == {ip} && tcp.dstport in {{{' '.join(map(str, ps))}}} && "
                                        f"tcp.flags.syn == 1 && tcp.flags.ack == 0)" for ip, ps in pairs.items()),
                            language="bash")
            else:
                st.info("縦スキャン・水平スキャンとも、該当する行動は検出されませんでした。")
            with st.expander("📊 全ホストの行動集約特徴量を展開"):
                st.dataframe(df_agg)

    with tab_b:
        if has_b:
            render_engine_b_results(summary_b, blocks_b, stats_b, ENGINE_B_MODEL, res["name"])
        else:
            st.error(f"エンジンBの解析に失敗しました：{res.get('B_error', '不明')}")
