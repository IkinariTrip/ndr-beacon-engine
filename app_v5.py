"""
app_v5.py
【仮称】インシデンス・コックピット改5：エンジンA（内部探索）×エンジンB（外部C2）の並列デュアル構成。
リポジトリ直下に配置し、streamlit run app_v5.py で起動する。app_v4.py は残したまま使える。

app_v4.py からの変更点:
  1. 解析ボタン押下時に、エンジンAとエンジンBを両方実行する（エンジンAの計算処理はapp_v4と同一）
  2. 結果を st.session_state に保存する
     → しきい値スライダーや展開ボタンを操作しても、結果が消えず、再解析も起きない
  3. 画面を3タブ構成にする：🧭 統合判定 ／ 🔍 エンジンA ／ 📡 エンジンB
  4. 相関判定（仕様 Phase D）：同じIPが「スキャン判定（A）」と「C2 CRITICAL/WARNING（B）」の
     両方に該当した場合、最優先の侵害端末（即時隔離の検討対象）として表示する
"""
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st

from src.flow_generator import extract_flows_from_pcap

# エンジンBのモジュール（src/engine_b/ 内は相互に平置きのimportを使うため、パスを先頭に追加）
ENGINE_B_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "engine_b")
if ENGINE_B_DIR not in sys.path:
    sys.path.insert(0, ENGINE_B_DIR)
from inference import analyze_pcap, DEFAULT_MODEL as ENGINE_B_MODEL  # noqa: E402
from engine_b_view import render_engine_b_results, triage_pairs  # noqa: E402

st.set_page_config(page_title="【仮称】インシデンス・コックピット改5", page_icon="🛡️", layout="wide")
st.title("🛡️ 【仮称】インシデンス・コックピット改5（デュアルエンジン版）")
st.caption("エンジンA：内部ポートスキャン検知（CatBoost v4）／エンジンB：外部C2ビーコン検知（CatBoost・タイミング5特徴量）")

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
    threshold = st.slider("異常判定しきい値", 0.1, 0.9, 0.5, step=0.05)
    window_size = st.number_input("時間窓ブロックサイズ (フロー数)", min_value=5, max_value=100, value=25)
    st.caption("ブロックサイズの変更は、解析を再実行すると反映されます")
    st.markdown("---")
    st.subheader("エンジンB（外部C2）")
    st.caption("4段階トリアージ：CRITICAL ≥0.85／WARNING／SAFE <0.35／FILTERED（通信10回未満）")
    st.markdown("---")
    st.markdown("""
**【改5】構成:**
- **エンジンA:** 空間（宛先ポートの散らばり）からスキャンを検知
- **エンジンB:** 時間（通信間隔のリズム）からC2を検知
- **相関判定:** 両方に該当する端末を最優先で提示
""")


def calculate_entropy(series):
    counts = series.value_counts()
    return float(entropy(counts)) if len(counts) > 1 else 0.0


# ---------------------------------------------------------------------------
# エンジンA：app_v4.py の処理をそのまま関数化（計算内容は一切変更していない）
# ---------------------------------------------------------------------------
def run_engine_a(pcap_path, window_size):
    flows, total_pkts = extract_flows_from_pcap(pcap_path)
    if not flows:
        return None, None, total_pkts

    flow_records = []
    for flow in flows:
        pkts = getattr(flow, "packets", [])
        if not pkts:
            continue
        times = [p[0] for p in pkts]
        lengths = [p[1] for p in pkts]
        directions = [p[2] for p in pkts]
        flags = [p[3] for p in pkts]

        fwd_lengths = [l for l, d in zip(lengths, directions) if d == 1]
        bwd_lengths = [l for l, d in zip(lengths, directions) if d == -1]

        # ヘッダー54バイトを除いた純粋ペイロード長
        fwd_payload_lengths = [max(0, l - 54) for l in fwd_lengths]
        bwd_payload_lengths = [max(0, l - 54) for l in bwd_lengths]

        fwd_len_mean = float(np.mean(fwd_payload_lengths)) if fwd_payload_lengths else 0.0
        bwd_len_mean = float(np.mean(bwd_payload_lengths)) if bwd_payload_lengths else 0.0

        total_fwd = len(fwd_lengths)
        total_bwd = len(bwd_lengths)

        down_up = float(total_bwd / total_fwd) if total_fwd > 0 else (1.0 if total_bwd > 0 else 0.0)
        has_syn = 1.0 if any(flg & 0x02 for flg in flags) else 0.0
        dur = (max(times) - min(times)) * 1e6 if len(times) > 1 else 0.0

        flow_records.append({
            "Src_IP": flow.src_ip, "Dst_IP": flow.dst_ip,
            "Src_Port": flow.src_port, "Dst_Port": flow.dst_port,
            "Flow_Duration": dur, "Fwd_Pkt_Mean": fwd_len_mean, "Bwd_Pkt_Mean": bwd_len_mean,
            "Total_Fwd_Pkts": total_fwd, "Total_Bwd_Pkts": total_bwd,
            "Down_Up_Ratio": down_up, "Has_SYN": has_syn,
        })

    df_flows = pd.DataFrame(flow_records)
    if df_flows.empty:
        return df_flows, None, total_pkts

    agg_blocks = []
    for src_ip, group in df_flows.groupby("Src_IP"):
        for i in range(0, len(group), window_size):
            chunk = group.iloc[i:i + window_size]
            if len(chunk) == 0:
                continue
            agg_blocks.append({
                "Src_IP": src_ip,
                "Observed_Flows": len(chunk),
                "Unique_Dst_Ports": chunk["Dst_Port"].nunique(),
                "Port_Entropy": calculate_entropy(chunk["Dst_Port"]),
                "Zero_Payload_Ratio": (chunk["Fwd_Pkt_Mean"] == 0).mean(),
                "Avg_Flow_Duration": chunk["Flow_Duration"].mean(),
                "Mean_Bwd_Pkt_Len": chunk["Bwd_Pkt_Mean"].mean(),
                "Fwd_Bwd_Pkt_Ratio": (chunk["Total_Fwd_Pkts"] /
                                      (chunk["Total_Fwd_Pkts"] + chunk["Total_Bwd_Pkts"] + 1e-5)).mean(),
                "Down_Up_Ratio": chunk["Down_Up_Ratio"].mean(),
                "SYN_Flag_Ratio": chunk["Has_SYN"].mean(),
            })
    df_agg = pd.DataFrame(agg_blocks)

    feature_cols = ["Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio", "Avg_Flow_Duration",
                    "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio", "Down_Up_Ratio", "SYN_Flag_Ratio"]
    use_cols = trained_features if trained_features else feature_cols
    df_agg["Anomaly_Score"] = model.predict_proba(df_agg[use_cols])[:, 1]
    return df_flows, df_agg, total_pkts


# ---------------------------------------------------------------------------
# 相関判定（Phase D）
# ---------------------------------------------------------------------------
def correlate(attack_blocks, pairs_b):
    """IPごとに、エンジンAのスキャン判定とエンジンBのC2判定を突き合わせる。"""
    a = (attack_blocks.groupby("Src_IP")
         .agg(A_max_score=("Anomaly_Score", "max"), A_scan_blocks=("Anomaly_Score", "size"))
         .reset_index().rename(columns={"Src_IP": "ip"})) if len(attack_blocks) else \
        pd.DataFrame(columns=["ip", "A_max_score", "A_scan_blocks"])

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
    in_a, in_b = m["A_max_score"].notna(), m["B_level"].notna()
    m["priority"] = np.select([in_a & in_b, in_b & (m["B_level"] == "CRITICAL"), in_b, in_a],
                              ["① 最優先（スキャン＋C2）", "② C2の疑い（CRITICAL）",
                               "③ C2の疑い（WARNING）", "④ スキャンのみ"], default="")
    return m.sort_values(["priority", "B_max_c2", "A_max_score"], ascending=[True, False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
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
                except Exception as e:  # tshark未導入など。エンジンAの結果は表示する
                    result["B_error"] = f"{type(e).__name__}: {e}"
        finally:
            if os.path.exists(tmp_pcap_path):
                os.remove(tmp_pcap_path)
        st.session_state["result"] = result

res = st.session_state.get("result")
if res:
    df_flows, df_agg, total_pkts = res["A"]
    has_a = df_agg is not None and len(df_agg) > 0
    attack_blocks = df_agg[df_agg["Anomaly_Score"] >= threshold] if has_a else pd.DataFrame()

    has_b = "B" in res
    if has_b:
        summary_b, blocks_b, stats_b = res["B"]
        pairs_b, counts_b = triage_pairs(summary_b, stats_b)
    else:
        pairs_b, counts_b = pd.DataFrame(), {"CRITICAL": 0, "WARNING": 0, "SAFE": 0, "FILTERED": 0}

    st.success(f"解析完了：{res['name']}（エンジンAブロックサイズ {res['window_size']}）")
    tab_all, tab_a, tab_b = st.tabs(["🧭 統合判定", "🔍 エンジンA：内部スキャン", "📡 エンジンB：外部C2"])

    # ---------------- 統合判定 ----------------
    with tab_all:
        corr = correlate(attack_blocks, pairs_b)
        top = corr[corr["priority"].str.startswith("①")]
        c = st.columns(4)
        c[0].metric("総解析パケット数", f"{total_pkts:,}")
        c[1].metric("A：スキャン送信元", f"{attack_blocks['Src_IP'].nunique() if len(attack_blocks) else 0} 台")
        c[2].metric("B：C2の疑い（CRITICAL）", f"{counts_b['CRITICAL']} ペア")
        c[3].metric("① 最優先端末", f"{len(top)} 台")

        if len(top):
            st.error("🚨 **最優先の侵害端末（内部スキャンと外部C2の両方に該当）：" + "、".join(top["ip"]) +
                     "**　→ 即時隔離を検討してください")
        elif len(corr):
            st.warning("両エンジンに同時に該当する端末はありません。下表の優先度順に確認してください。")
        else:
            st.success("どちらのエンジンでも要対応の端末は検出されませんでした。")
        if not has_b:
            st.warning(f"エンジンBは実行できませんでした（{res.get('B_error', '不明')}）。相関判定はエンジンAのみです。")

        if len(corr):
            show = corr[["priority", "ip", "A_max_score", "A_scan_blocks", "B_level", "B_max_c2", "B_c2_peers"]]
            st.dataframe(show.rename(columns={
                "priority": "優先度", "ip": "端末IP", "A_max_score": "A：最大スキャンスコア",
                "A_scan_blocks": "A：異常ブロック数", "B_level": "B：区分", "B_max_c2": "B：最大C2確率",
                "B_c2_peers": "B：C2の疑いがある相手"}).round(3), hide_index=True)
            if len(top):
                st.markdown("**Wireshark表示フィルタ（最優先端末の全通信）**")
                st.code(" || ".join(f"ip.addr == {ip}" for ip in top["ip"]), language="text")
        st.caption("優先度：① 両エンジンに該当 ＞ ② C2 CRITICAL ＞ ③ C2 WARNING ＞ ④ スキャンのみ。"
                   "エンジンAは通信方向を区別せず送信元IPで判定、エンジンBは社内側IPで判定しています。")

    # ---------------- エンジンA ----------------
    with tab_a:
        if not has_a:
            st.warning("有効なTCP/UDPフローが検出されませんでした。")
        else:
            total_blocks = len(df_agg)
            attack_ratio = len(attack_blocks) / total_blocks * 100 if total_blocks else 0.0
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("総解析パケット数", f"{total_pkts:,} パック")
            col2.metric("生成フロー数", f"{len(df_flows):,} の流れ")
            col3.metric("異常行動ブロック数", f"{len(attack_blocks):,} ブロック")
            col4.metric("異常判定率", f"{attack_ratio:.2f}%")
            st.markdown("---")
            st.subheader("🚨 侵害端末候補スクリーニング結果")
            if not attack_blocks.empty:
                attacker_ips = attack_blocks["Src_IP"].unique()
                st.error(f"🚨 **悪意あるスキャン送信元ホストを特定: {', '.join(attacker_ips)}**")
                st.write("##### 異常判定ブロック一覧")
                display_cols = ["Src_IP", "Anomaly_Score", "Unique_Dst_Ports", "Port_Entropy",
                                "Zero_Payload_Ratio", "SYN_Flag_Ratio", "Mean_Bwd_Pkt_Len", "Observed_Flows"]
                st.dataframe(attack_blocks[display_cols].sort_values("Anomaly_Score", ascending=False))
                st.write("##### 🔍 調査用 Wireshark フィルタ")
                st.code(" || ".join(f"ip.src == {ip}" for ip in attacker_ips), language="bash")
            else:
                st.info("指定されたしきい値を超える不審なスキャン行動は検出されませんでした。")
            with st.expander("📊 全ホストの行動集約特徴量を展開"):
                st.dataframe(df_agg)

    # ---------------- エンジンB ----------------
    with tab_b:
        if has_b:
            render_engine_b_results(summary_b, blocks_b, stats_b, ENGINE_B_MODEL, res["name"])
        else:
            st.error(f"エンジンBの解析に失敗しました：{res.get('B_error', '不明')}")
