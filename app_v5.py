"""
app_v5.py（v5.2：その他攻撃〔フラッド・DDoS着弾・リフレクション〕のルールベース検知を追加）
【仮称】インシデンス・コックピット改5：エンジンA（内部探索）×エンジンB（外部C2）の並列デュアル構成。
リポジトリ直下に配置し、streamlit run app_v5.py で起動する。

v5.1 → v5.2 の変更点:
  - エンジンAに「その他攻撃の疑い」判定を追加（仕様書には記載のない拡張。断定しない参考表示）
      送信元ベース：フラッド（DoS）の疑い、リフレクション/増幅の疑い
      宛先ベース　：分散アクセス（DDoS着弾）の疑い（視点を反転した集計）
  - 縦スキャン・水平スキャン・C2（エンジンB）のWiresharkフィルタに、具体的な数値を
    盛り込んだ追加フィルタ（SYNのみ／観測期間／周期性確認／応答有無確認 等）を追加
  - 統合判定の優先度ロジックに「その他攻撃の疑い」を組み込み
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
from src.engine_a.other_attacks import (flood_flags, reflection_flags, build_dest_side_blocks,
                                        ddos_dest_flags, attach_top_src_ips,
                                        flood_filters, reflection_filters, ddos_dest_filters,
                                        FLOOD_RULE, DDOS_DEST_RULE, REFLECTION_RULE, REFLECTION_PORTS)
from src.engine_a.filters import vertical_filters, horizontal_filters

ENGINE_B_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b")
if ENGINE_B_DIR not in sys.path:
    sys.path.insert(0, ENGINE_B_DIR)
from inference import analyze_pcap, DEFAULT_MODEL as ENGINE_B_MODEL  # noqa: E402
from engine_b_view import render_engine_b_results, triage_pairs  # noqa: E402

st.set_page_config(page_title="【仮称】インシデンス・コックピット改5", page_icon="🛡️", layout="wide")
st.title("🛡️ 【仮称】インシデンス・コックピット改5（デュアルエンジン版）")
st.caption("エンジンA：内部スキャン検知（縦：CatBoost v4／水平・その他：ルール）／"
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
               f"応答なし率≥{H_RULE['no_return']}")
    st.markdown("---")
    use_other = st.checkbox("その他攻撃の疑い（参考表示）を有効にする", value=True)
    st.caption(f"フラッド：宛先IP種類≤{FLOOD_RULE['max_dst_ips']}、同一ポート率≥{FLOOD_RULE['port_share']}、"
               f"{FLOOD_RULE['min_rate']:.0f}フロー/秒以上")
    st.caption(f"DDoS着弾：異なる送信元IP≥{DDOS_DEST_RULE['min_src_ips']}件が{DDOS_DEST_RULE['max_window_sec']:.0f}秒以内に集中")
    st.caption(f"リフレクション：増幅率≥{REFLECTION_RULE['min_amp_ratio']:.0f}倍（DNS/NTP/SSDP等の既知ポート）")
    st.caption("※ いずれもルールベースの参考表示です。断定ではなく、調査の優先順位付けとして使用してください。")
    st.markdown("---")
    st.subheader("エンジンB（外部C2）")
    st.caption("4段階トリアージ：CRITICAL ≥0.85／WARNING／SAFE <0.35／FILTERED（通信10回未満）")


def run_engine_a(pcap_path, window_size):
    flows, total_pkts = extract_flows_from_pcap(pcap_path)
    if not flows:
        return None, None, None, total_pkts
    df_flows = flows_to_dataframe(flows)
    if df_flows.empty:
        return df_flows, None, None, total_pkts
    df_agg = build_blocks(df_flows, window_size)
    use_cols = trained_features if trained_features else VERTICAL_FEATURES
    df_agg["Anomaly_Score"] = model.predict_proba(df_agg[use_cols])[:, 1]
    df_dest = build_dest_side_blocks(df_flows, window_size)
    return df_flows, df_agg, df_dest, total_pkts


def engine_a_hits(df_agg, threshold, use_h, use_other):
    """縦スキャン（モデル）・水平スキャン／フラッド／リフレクション（ルール）の判定を付ける。"""
    d = df_agg.copy()
    d["Vertical"] = d["Anomaly_Score"] >= threshold
    d["Horizontal"] = horizontal_scan_flags(d) if use_h else False
    d["Flood"] = flood_flags(d) if use_other else False
    d["Reflection"] = reflection_flags(d) if use_other else False

    def _types(row):
        t = []
        if row["Vertical"]:
            t.append("縦スキャン")
        if row["Horizontal"]:
            t.append("水平スキャン")
        if row["Flood"]:
            t.append("フラッドの疑い")
        if row["Reflection"]:
            t.append("リフレクションの疑い")
        return "／".join(t)

    d["Scan_Type"] = d.apply(_types, axis=1)
    d["Is_Scan"] = d["Vertical"] | d["Horizontal"]
    d["Is_Other"] = d["Flood"] | d["Reflection"]
    hit = d[d["Vertical"] | d["Horizontal"] | d["Flood"] | d["Reflection"]]
    return d, hit


def correlate(attack_blocks, pairs_b):
    if len(attack_blocks):
        a = (attack_blocks.groupby("Src_IP")
             .agg(A_type=("Scan_Type", lambda s: "／".join(sorted(set("／".join(s).split("／"))))),
                  A_max_score=("Anomaly_Score", "max"), A_blocks=("Scan_Type", "size"),
                  A_is_scan=("Is_Scan", "any"), A_is_other=("Is_Other", "any"),
                  A_ports=("Top_Dst_Port", lambda s: ", ".join(map(str, pd.Series(s).value_counts().index[:3]))))
             .reset_index().rename(columns={"Src_IP": "ip"}))
    else:
        a = pd.DataFrame(columns=["ip", "A_type", "A_max_score", "A_blocks", "A_is_scan",
                                  "A_is_other", "A_ports"])

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
    in_a_scan = m["A_is_scan"].fillna(False).astype(bool)
    in_a_other = m["A_is_other"].fillna(False).astype(bool)
    in_b = m["B_level"].notna()
    m["priority"] = np.select(
        [in_a_scan & in_b, in_b & (m["B_level"] == "CRITICAL"), in_b,
         in_a_scan, in_a_other],
        ["① 最優先（スキャン＋C2）", "② C2の疑い（CRITICAL）", "③ C2の疑い（WARNING）",
         "④ スキャンのみ", "⑤ その他攻撃の疑い（参考）"],
        default="")
    return m.sort_values(["priority", "B_max_c2", "A_blocks"],
                         ascending=[True, False, False]).reset_index(drop=True)


uploaded_file = st.file_uploader("解析対象のパケットキャプチャ (PCAP / PCAPNG) を選択", type=["pcap", "pcapng"])

if uploaded_file is not None:
    st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")
    if st.button("🚀 パケット解析・行動スクリーニング実行（エンジンA＋B）", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_pcap_path = tmp.name
        result = {"name": os.path.splitext(uploaded_file.name)[0], "window_size": int(window_size)}
        try:
            with st.spinner("エンジンA：フロー抽出・スキャン判定・その他攻撃判定中…"):
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
    df_flows, df_agg, df_dest, total_pkts = res["A"]
    has_a = df_agg is not None and len(df_agg) > 0
    if has_a:
        df_agg, attack_blocks = engine_a_hits(df_agg, threshold, use_h, use_other)
        dest_hit = (df_dest[ddos_dest_flags(df_dest)] if (df_dest is not None and len(df_dest) and use_other)
                   else pd.DataFrame())
        dest_hit = attach_top_src_ips(df_flows, dest_hit, int(window_size)) if len(dest_hit) else dest_hit
    else:
        attack_blocks, dest_hit = [], pd.DataFrame()

    has_b = "B" in res
    if has_b:
        summary_b, blocks_b, stats_b = res["B"]
        from src.engine_a.cross_check import mark_flood_matches
        summary_b = mark_flood_matches(summary_b, attack_blocks)
        n_flood_match = int(summary_b["a_flood_match"].sum()) if "a_flood_match" in summary_b.columns else 0
        if n_flood_match:
            st.info("エンジンAの「フラッドの疑い」と宛先が一致するC2判定の通信ペアが " + str(n_flood_match) + " 件あります。このうちCRITICALのものは、高頻度のDoSをC2と誤認した可能性があるため、WARNINGに引き下げて表示しています。")
        pairs_b, counts_b = triage_pairs(summary_b, stats_b)
    else:
        pairs_b, counts_b = pd.DataFrame(), {"CRITICAL": 0, "WARNING": 0, "SAFE": 0, "FILTERED": 0}

    st.success(f"解析完了：{res['name']}（エンジンAブロックサイズ {res['window_size']}）")
    tab_all, tab_a, tab_b = st.tabs(["🧭 統合判定", "🔍 エンジンA：内部スキャン・その他攻撃", "📡 エンジンB：外部C2"])

    # ================= 統合判定 =================
    with tab_all:
        corr = correlate(attack_blocks, pairs_b)
        top = corr[corr["priority"].str.startswith("①")]
        c = st.columns(5)
        c[0].metric("総解析パケット数", f"{total_pkts:,}")
        c[1].metric("A：スキャン送信元", f"{attack_blocks[attack_blocks['Is_Scan']]['Src_IP'].nunique() if len(attack_blocks) else 0} 台")
        c[2].metric("A：その他攻撃の疑い", f"{attack_blocks[attack_blocks['Is_Other']]['Src_IP'].nunique() if len(attack_blocks) else 0} 台")
        c[3].metric("B：C2の疑い（CRITICAL）", f"{counts_b['CRITICAL']} ペア")
        c[4].metric("① 最優先端末", f"{len(top)} 台")

        if len(top):
            st.error("🚨 **最優先の侵害端末（スキャンと外部C2の両方に該当）：" + "、".join(top["ip"]) +
                     "**　→ 即時隔離を検討してください")
        elif len(corr[corr["priority"] != ""]):
            st.warning("両エンジンに同時に該当する端末はありません。下表の優先度順に確認してください。")
        else:
            st.success("どちらのエンジンでも要対応の端末は検出されませんでした。")
        if not has_b:
            st.warning(f"エンジンBは実行できませんでした（{res.get('B_error', '不明')}）。相関判定はエンジンAのみです。")
        if len(dest_hit):
            st.info(f"⑤ 宛先ベースの分散アクセス（DDoS着弾）の疑いが {len(dest_hit)} 件あります。"
                    "詳細はエンジンAタブの「その他攻撃の疑い」をご確認ください。")

        show_cols = corr[corr["priority"] != ""]
        if len(show_cols):
            show = show_cols[["priority", "ip", "A_type", "A_blocks", "A_ports", "B_level", "B_max_c2", "B_c2_peers"]]
            st.dataframe(show.rename(columns={
                "priority": "優先度", "ip": "端末IP", "A_type": "A：種別", "A_blocks": "A：該当ブロック数",
                "A_ports": "A：主な宛先ポート", "B_level": "B：区分", "B_max_c2": "B：最大C2確率",
                "B_c2_peers": "B：C2の疑いがある相手"}).round(3), hide_index=True)
            if len(top):
                st.markdown("**Wireshark表示フィルタ（最優先端末の全通信）**")
                st.code(" || ".join(f"ip.addr == {ip}" for ip in top["ip"]), language="text")
        st.caption("優先度：① 両エンジンに該当 ＞ ② C2 CRITICAL ＞ ③ C2 WARNING ＞ ④ スキャンのみ ＞ "
                   "⑤ その他攻撃の疑い（参考・断定ではありません）。"
                   "エンジンAは送信元IP、エンジンBは社内側IPで判定しています。")

    # ================= エンジンA =================
    with tab_a:
        if not has_a:
            st.warning("有効なTCP/UDPフローが検出されませんでした。")
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("総解析パケット数", f"{total_pkts:,} パック")
            col2.metric("生成フロー数", f"{len(df_flows):,} の流れ")
            col3.metric("縦スキャン", f"{int(df_agg['Vertical'].sum()):,} ブロック")
            col4.metric("水平スキャン", f"{int(df_agg['Horizontal'].sum()):,} ブロック")
            col5.metric("その他攻撃の疑い", f"{int(df_agg['Is_Other'].sum()):,} ブロック")

            st.markdown("---")
            st.subheader("🚨 侵害端末候補スクリーニング結果（縦・水平スキャン）")
            scan_blocks = attack_blocks[attack_blocks["Is_Scan"]] if len(attack_blocks) else attack_blocks
            if len(scan_blocks):
                for stype, g in scan_blocks.groupby("Scan_Type"):
                    if "スキャン" in stype:
                        st.error(f"🚨 **{stype}の送信元: {', '.join(g['Src_IP'].unique())}**")
                st.write("##### 異常判定ブロック一覧")
                display_cols = ["Src_IP", "Scan_Type", "Anomaly_Score", "Unique_Dst_Ports", "Unique_Dst_IPs",
                                "Top_Dst_Port", "Dst_IP_Ratio", "Top_Port_Share", "No_Data_Return_Ratio",
                                "Observed_Flows"]
                st.dataframe(scan_blocks[display_cols].sort_values(["Scan_Type", "Anomaly_Score"],
                                                                   ascending=[True, False]).round(3))
                st.write("##### 🔍 調査用 Wireshark フィルタ（送信元ごと・代表ブロック）")
                for src_ip, g in scan_blocks.groupby("Src_IP"):
                    row = g.iloc[0]
                    with st.expander(f"{src_ip}　種別：{row['Scan_Type']}"):
                        if row["Vertical"]:
                            st.caption("縦スキャン（1台の相手に多数のポート）")
                            for name, f in vertical_filters(row).items():
                                st.caption(name)
                                st.code(f, language="text")
                        if row["Horizontal"]:
                            st.caption("水平スキャン（同一ポートを多数の相手に）")
                            for name, f in horizontal_filters(row).items():
                                st.caption(name)
                                st.code(f, language="text")
            else:
                st.info("縦スキャン・水平スキャンとも、該当する行動は検出されませんでした。")

            st.markdown("---")
            st.subheader("🟡 その他攻撃の疑い（参考・ルールベース／断定ではありません）")
            other_blocks = attack_blocks[attack_blocks["Is_Other"]] if len(attack_blocks) else pd.DataFrame()
            if len(other_blocks):
                st.caption("フラッド（DoS）の疑い・リフレクション/増幅の疑い（送信元ベース）")
                oc = ["Src_IP", "Scan_Type", "Top_Dst_IP", "Top_Dst_Port", "Observed_Flows",
                      "Flow_Rate", "Byte_Amplification_Ratio", "Bwd_Bytes_Total"]
                st.dataframe(other_blocks[oc].round(2))
                for src_ip, g in other_blocks.groupby("Src_IP"):
                    row = g.iloc[0]
                    with st.expander(f"{src_ip}　種別：{row['Scan_Type']}"):
                        if row["Flood"]:
                            st.caption(f"フラッド（DoS）の疑い：{row['Flow_Rate']:.1f}フロー/秒、"
                                      f"宛先IP種類 {int(row['Unique_Dst_IPs'])}、"
                                      f"同一ポート率 {row['Top_Port_Share']:.2f}")
                            for name, f in flood_filters(row).items():
                                st.caption(name)
                                st.code(f, language="text")
                        if row["Reflection"]:
                            port_name = REFLECTION_PORTS.get(int(row["Top_Dst_Port"]), str(int(row["Top_Dst_Port"])))
                            st.caption(f"リフレクション/増幅の疑い：{port_name}ポート、"
                                      f"増幅率 約{row['Byte_Amplification_Ratio']:.1f}倍")
                            for name, f in reflection_filters(row).items():
                                st.caption(name)
                                st.code(f, language="text")
            else:
                st.caption("送信元ベースのフラッド・リフレクションの疑いは検出されませんでした。")

            st.caption("分散アクセス（DDoS着弾）の疑い（宛先ベース：視点を反転した集計）")
            if len(dest_hit):
                dc = ["Dst_IP", "Dst_Port", "Unique_Src_IPs", "Observed_Flows", "Window_Sec", "Top_Src_IPs"]
                st.dataframe(dest_hit[dc].round(2))
                for _, row in dest_hit.iterrows():
                    with st.expander(f"宛先 {row['Dst_IP']}:{int(row['Dst_Port'])}　"
                                     f"異なる送信元IP {int(row['Unique_Src_IPs'])}件"):
                        st.caption(f"{row['Window_Sec']:.1f}秒間に{int(row['Observed_Flows'])}フローが集中")
                        for name, f in ddos_dest_filters(row).items():
                            st.caption(name)
                            st.code(f, language="text")
            else:
                st.caption("宛先ベースの分散アクセスの疑いは検出されませんでした。")

            with st.expander("📊 全ホストの行動集約特徴量を展開"):
                st.dataframe(df_agg)

    # ================= エンジンB =================
    with tab_b:
        if has_b:
            render_engine_b_results(summary_b, blocks_b, stats_b, ENGINE_B_MODEL, res["name"])
        else:
            st.error(f"エンジンBの解析に失敗しました：{res.get('B_error', '不明')}")
