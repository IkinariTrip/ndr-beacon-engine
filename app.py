import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

from src.flow_generator import extract_flows_from_pcap
from src.feature_engineering import build_feature_dataframe, DETERMINED_15_FEATURES

st.set_page_config(page_title="PCAPNG Anomaly Pattern Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ PCAPNG Anomaly Pattern Detector (Incidence Cockpit)")
st.caption("生パケットから5-tuple通信Flowを自動抽出し、確定15特徴量に基づき異常通信を瞬時にスクリーニングします。")

@st.cache_resource
def load_detection_model():
    model_path = "models/anomaly_detector.joblib"
    if not os.path.exists(model_path):
        st.error(f"モデルファイルが見つかりません: {model_path}")
        return None
    payload = joblib.load(model_path)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload

model = load_detection_model()

st.sidebar.header("⚙️ 判定パラメータ")
threshold = st.sidebar.slider("異常スコア判定閾値", min_value=0.10, max_value=0.99, value=0.50, step=0.05)

uploaded_file = st.file_uploader("Wireshark キャプチャファイル (.pcap / .pcapng) をドラッグ＆ドロップ", type=["pcap", "pcapng"])

if uploaded_file is not None and model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcapng") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")

    if st.button("🚀 パケット解析・異常スクリーニング実行", type="primary"):
        status_box = st.status("解析パイプラインを実行中...", expanded=True)
        
        status_box.write("▶ Phase A: 生パケット読み込み & 5-tuple セッション集約中...")
        flows, total_packets = extract_flows_from_pcap(tmp_path)
        status_box.write(f"  └ 総パケット: {total_packets:,} 件 / 生成Flow数: {len(flows):,} 件")
        
        status_box.write("▶ Phase B: 確定15特徴量のメモリ上高速算出中...")
        X_features, df_meta = build_feature_dataframe(flows)
        
        status_box.write("▶ Phase C: LightGBM推論エンジンによる判定中...")
        X_input = X_features.copy()
        X_input.columns = [c.replace(' ', '_') for c in X_input.columns]
        
        try:
            anomaly_scores = model.predict_proba(X_input)[:, 1]
        except Exception:
            anomaly_scores = model.predict_proba(X_features)[:, 1]

        status_box.update(label="✅ 全パイプラインの解析が完了しました！", state="complete", expanded=False)
        os.remove(tmp_path)

        df_result = df_meta.copy()
        df_result["Anomaly_Score"] = anomaly_scores
        df_result["Is_Anomaly"] = df_result["Anomaly_Score"] >= threshold
        df_result["Wireshark_Filter"] = df_result.apply(
            lambda r: f"ip.src == {r['Src_IP']} && ip.dst == {r['Dst_IP']}", axis=1
        )

        st.markdown("---")
        st.subheader("📊 解析サマリー")
        c1, c2, c3, c4 = st.columns(4)
        total_flows = len(df_result)
        anomaly_flows = int(df_result["Is_Anomaly"].sum())
        anomaly_rate = (anomaly_flows / total_flows * 100) if total_flows > 0 else 0.0

        c1.metric("総解析パケット数", f"{total_packets:,} pkts")
        c2.metric("生成Flow数", f"{total_flows:,} flows")
        c3.metric("異常候補Flow数", f"{anomaly_flows:,} flows")
        c4.metric("異常候補率", f"{anomaly_rate:.2f} %")

        st.markdown("---")
        st.subheader(f"⚠️ 調査優先度 High: 異常候補セッション一覧 (閾値 >= {threshold})")
        
        df_anomalies = df_result[df_result["Is_Anomaly"]].sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
        
        if not df_anomalies.empty:
            scanner_ips = df_anomalies["Src_IP"].value_counts()
            st.error(f"🚨 侵害端末（スキャン送信元ホスト）候補を特定: **{scanner_ips.index[0]}** (検知セッション数: {scanner_ips.values[0]} 件)")
            
            disp_df = df_anomalies[["Src_IP", "Src_Port", "Dst_IP", "Dst_Port", "Protocol", "Anomaly_Score", "Total_Packets", "Wireshark_Filter"]].copy()
            disp_df["Anomaly_Score"] = disp_df["Anomaly_Score"].map(lambda x: f"{x:.4f}")
            st.dataframe(disp_df, use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 抽出特徴量 ドリルダウン検証")
            selected_idx = st.selectbox("詳細を確認するFlowを選択", options=df_anomalies.index)
            st.write(f"**Flow詳細:** `{df_anomalies.loc[selected_idx, 'Src_IP']}` ➔ `{df_anomalies.loc[selected_idx, 'Dst_IP']}` (Score: {df_anomalies.loc[selected_idx, 'Anomaly_Score']:.4f})")
            st.dataframe(X_features.iloc[[selected_idx]].T.rename(columns={selected_idx: "値"}), use_container_width=True)
        else:
            st.success("指定された閾値を超える異常通信は検出されませんでした。")