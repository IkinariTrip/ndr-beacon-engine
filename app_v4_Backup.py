import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st
from src.flow_generator import extract_flows_from_pcap

# --- ページ基本設定 ---
st.set_page_config(
    page_title="【仮称】インシデンス・コックピット改4",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 【仮称】インシデンス・コックピット改4")
st.caption("学術論文（Rosay et al., 2022）準拠・TCPヘッダー分離ペイロード判定エンジン (推論モデル: model_lightgbm_v4)")

# --- モデル読み込み ---
MODEL_PATH = "models/model_lightgbm_v4.joblib"

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

# --- サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 監視パラメータ")
    threshold = st.slider("異常判定しきい値", 0.1, 0.9, 0.5, step=0.05)
    window_size = st.number_input("時間窓ブロックサイズ (フロー数)", min_value=5, max_value=100, value=25)
    st.markdown("---")
    st.markdown("""
    **【改4】実装アップデート:**
    - **モデル更新:** 論文指摘のフラグ反転バグ（SYN↔PSH）補正済みモデル `v4` を採用
    - **ペイロード厳密判定:** L2/L3/L4ヘッダー（54バイト）を除去し、空パケットを正しく 0 バイト判定
    - **SYN比率の正規化:** 時間窓内の純粋スキャン開始率を 0.0〜1.0 で正確に集計
    - **多次元抑制:** 正常サーバー応答（AD/DNS）の誤検知抑制とスキャン高精度検知の両立
    """)

# --- エントロピー計算関数 ---
def calculate_entropy(series):
    counts = series.value_counts()
    return float(entropy(counts)) if len(counts) > 1 else 0.0

# --- メインエリア：PCAPアップロード ---
uploaded_file = st.file_uploader("解析対象のパケットキャプチャ (PCAP / PCAPNG) を選択", type=["pcap", "pcapng"])

if uploaded_file is not None:
    st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")

    if st.button("🚀 パケット解析・行動スクリーニング実行", type="primary"):
        with st.spinner("生パケットを抽出・改4パイプラインで特徴量計算中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pcap_path = tmp_file.name

            try:
                # 1. 生パケットからフロー抽出
                flows, total_pkts = extract_flows_from_pcap(tmp_pcap_path)
                if not flows:
                    st.warning("有効なTCP/UDPフローが検出されませんでした。")
                    st.stop()

                # 2. フローごとの特徴量計算（ヘッダーとペイロードを厳密分離）
                flow_records = []
                for flow in flows:
                    pkts = getattr(flow, 'packets', [])
                    if not pkts:
                        continue

                    times = [p[0] for p in pkts]
                    lengths = [p[1] for p in pkts]
                    directions = [p[2] for p in pkts]
                    flags = [p[3] for p in pkts]

                    fwd_lengths = [l for l, d in zip(lengths, directions) if d == 1]
                    bwd_lengths = [l for l, d in zip(lengths, directions) if d == -1]

                    # 【不備修正】TCP/IPヘッダー（最低54バイト）を差し引いた純粋データ長
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

                # 3. 25フロー時間窓ブロック集約
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

                # 4. 推論実行
                feature_cols = [
                    "Unique_Dst_Ports", "Port_Entropy", "Zero_Payload_Ratio",
                    "Avg_Flow_Duration", "Mean_Bwd_Pkt_Len", "Fwd_Bwd_Pkt_Ratio",
                    "Down_Up_Ratio", "SYN_Flag_Ratio"
                ]
                use_cols = trained_features if trained_features else feature_cols
                X_input = df_agg[use_cols]

                probs = model.predict_proba(X_input)[:, 1]
                df_agg["Anomaly_Score"] = probs

                # --- 画面表示 ---
                st.success("【改4】全パイプラインの解析が完了しました！")

                total_blocks = len(df_agg)
                attack_blocks = df_agg[df_agg["Anomaly_Score"] >= threshold]
                attack_ratio = (len(attack_blocks) / total_blocks * 100) if total_blocks > 0 else 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("総解析パケット数", f"{total_pkts:,} パック")
                col2.metric("生成フロー数", f"{len(df_flows):,} の流れ")
                col3.metric("異常行動ブロック数", f"{len(attack_blocks):,} ブロック")
                col4.metric("異常判定率", f"{attack_ratio:.2f}%")

                st.markdown("---")

                # 侵害端末の特定表示
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