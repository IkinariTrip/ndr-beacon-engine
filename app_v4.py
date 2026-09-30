import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st
from src.flow_generator import extract_flows_from_pcap

st.set_page_config(
    page_title="【仮称】インシデンス・コックピット改4",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 【仮称】インシデンス・コックピット改4 (CatBoost版)")
st.caption("学術論文（Rosay et al., 2022）準拠・8次元対称木推論エンジン (model: CatBoost v4)")

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
    threshold = st.slider("異常判定しきい値", 0.1, 0.9, 0.5, step=0.05)
    window_size = st.number_input("時間窓ブロックサイズ (フロー数)", min_value=5, max_value=100, value=25)
    st.markdown("---")
    st.markdown("""
    **【改4 CatBoost】仕様:**
    - **モデル:** 対称木構造により汎化性能を最大化した CatBoost Classifier
    - **ヘッダー分離:** L2/L3/L4ヘッダーを除去しペイロード実データ長を評価
    - **整合性修復:** 論文に基づくSYNフラグ反転バグを補正
    """)

def calculate_entropy(series):
    counts = series.value_counts()
    return float(entropy(counts)) if len(counts) > 1 else 0.0

uploaded_file = st.file_uploader("解析対象のパケットキャプチャ (PCAP / PCAPNG) を選択", type=["pcap", "pcapng"])

if uploaded_file is not None:
    st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")

    if st.button("🚀 パケット解析・行動スクリーニング実行", type="primary"):
        with st.spinner("生パケットを抽出・CatBoostパイプラインで解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pcap_path = tmp_file.name

            try:
                flows, total_pkts = extract_flows_from_pcap(tmp_pcap_path)
                if not flows:
                    st.warning("有効なTCP/UDPフローが検出されませんでした。")
                    st.stop()

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
                        "Src_IP": flow.src_ip,
                        "Dst_IP": flow.dst_ip,
                        "Src_Port": flow.src_port,
                        "Dst_Port": flow.dst_port,
                        "Flow_Duration": dur,
                        "Fwd_Pkt_Mean": fwd_len_mean,
                        "Bwd_Pkt_Mean": bwd_len_mean,
                        "Total_Fwd_Pkts": total_fwd,
                        "Total_Bwd_Pkts": total_bwd,
                        "Down_Up_Ratio": down_up,
                        "Has_SYN": has_syn
                    })

                df_flows = pd.DataFrame(flow_records)
                if df_flows.empty:
                    st.warning("解析可能な有効パケットが含まれていませんでした。")
                    st.stop()

                agg_blocks = []
                for src_ip, group in df_flows.groupby("Src_IP"):
                    for i in range(0, len(group), window_size):
                        chunk = group.iloc[i:i + window_size]
                        if len(chunk) == 0:
                            continue

                        u_ports = chunk["Dst_Port"].nunique()
                        p_ent = calculate_entropy(chunk["Dst_Port"])
                        zero_ratio = (chunk["Fwd_Pkt_Mean"] == 0).mean()
                        avg_dur = chunk["Flow_Duration"].mean()

                        mean_bwd_len = chunk["Bwd_Pkt_Mean"].mean()
                        fwd_bwd_ratio = (chunk["Total_Fwd_Pkts"] / (chunk["Total_Fwd_Pkts"] + chunk["Total_Bwd_Pkts"] + 1e-5)).mean()
                        mean_down_up = chunk["Down_Up_Ratio"].mean()
                        syn_ratio = chunk["Has_SYN"].mean()

                        agg_blocks.append({
                            "Src_IP": src_ip,
                            "Observed_Flows": len(chunk),
                            "Unique_Dst_Ports": u_ports,
                            "Port_Entropy": p_ent,
                            "Zero_Payload_Ratio": zero_ratio,
                            "Avg_Flow_Duration": avg_dur,
                            "Mean_Bwd_Pkt_Len": mean_bwd_len,
                            "Fwd_Bwd_Pkt_Ratio": fwd_bwd_ratio,
                            "Down_Up_Ratio": mean_down_up,
                            "SYN_Flag_Ratio": syn_ratio
                        })

                df_agg = pd.DataFrame(agg_blocks)

                feature_cols = [
                    "Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio",
                    "Avg_Flow_Duration", "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio",
                    "Down_Up_Ratio", "SYN_Flag_Ratio"
                ]
                use_cols = trained_features if trained_features else feature_cols
                X_input = df_agg[use_cols]

                # CatBoost 推論
                probs = model.predict_proba(X_input)[:, 1]
                df_agg["Anomaly_Score"] = probs

                st.success("【改4 CatBoost】全パイプラインの解析が完了しました！")

                total_blocks = len(df_agg)
                attack_blocks = df_agg[df_agg["Anomaly_Score"] >= threshold]
                attack_ratio = (len(attack_blocks) / total_blocks * 100) if total_blocks > 0 else 0.0

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
                    display_cols = [
                        "Src_IP", "Anomaly_Score", "Unique_Dst_Ports", "Port_Entropy",
                        "Zero_Payload_Ratio", "SYN_Flag_Ratio", "Mean_Bwd_Pkt_Len", "Observed_Flows"
                    ]
                    st.dataframe(attack_blocks[display_cols].sort_values("Anomaly_Score", ascending=False))

                    st.write("##### 🔍 調査用 Wireshark フィルタ")
                    filter_str = " || ".join([f"ip.src == {ip}" for ip in attacker_ips])
                    st.code(filter_str, language="bash")
                else:
                    st.info("指定されたしきい値を超える不審なスキャン行動は検出されませんでした。")

                with st.expander("📊 全ホストの行動集約特徴量を展開"):
                    st.dataframe(df_agg)

            finally:
                if os.path.exists(tmp_pcap_path):
                    os.remove(tmp_pcap_path)