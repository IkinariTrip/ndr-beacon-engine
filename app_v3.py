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
    page_title="【仮称】インシデンス・コックピット 改3",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ 【仮称】インシデンス・コックピット 改3")
st.caption(
    "次世代双方向プロトコル振る舞い監視エンジン (推論モデル: 8次元特徴量"
    " LightGBM GBDT / F1: 99.73%)"
)

# --- モデル読み込み ---
MODEL_PATH = "models/model_lightgbm_v3.joblib"


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
  st.error(
      f"⚠️ モデルファイルが見つかりません。'{MODEL_PATH}'"
      " を配置してください。"
  )
  st.stop()

# --- サイドバー設定 ---
with st.sidebar:
  st.header("⚙️ 監視パラメータ")
  threshold = st.slider("異常判定しきい値", 0.1, 0.9, 0.5, step=0.05)
  window_size = st.number_input(
      "時間窓ブロックサイズ (フロー数)", min_value=5, max_value=100, value=25
  )
  st.markdown("---")
  st.markdown("""
    **新機能 (改3 - 双方向プロトコル適応):**
    - **戻りデータサイズ (Mean Bwd Pkt Len)** で正規応答を識別
    - **方向非対称度 (Fwd/Bwd Ratio)** で片道ノックを検知
    - **Down/Up Ratio** および **SYN比率** による総合判定
    - **ブラックリスト不使用**: 多次元空間でサーバー誤検知を自動抑制
    """)


# --- エントロピー計算関数 ---
def calculate_entropy(series):
  counts = series.value_counts()
  return float(entropy(counts)) if len(counts) > 1 else 0.0


# --- メインエリア：PCAPアップロード ---
uploaded_file = st.file_uploader(
    "解析対象のパケットキャプチャ (PCAP / PCAPNG) を選択",
    type=["pcap", "pcapng"],
)

if uploaded_file is not None:
  st.info(f"解析対象ファイル: **{uploaded_file.name}** を受付完了。")

  if st.button("🚀 パケット解析・行動スクリーニング実行", type="primary"):
    with st.spinner("生パケットを抽出・8次元特徴量を直接計算中..."):
      with tempfile.NamedTemporaryFile(
          delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
      ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pcap_path = tmp_file.name

      try:
        # 1. 生パケットからフローを抽出
        flows, total_pkts = extract_flows_from_pcap(tmp_pcap_path)
        if not flows:
          st.warning("有効なTCP/UDPフローが検出されませんでした。")
          st.stop()

        # 2. 各フローの生パケットから直接8次元の中間特徴量を計算
        flow_records = []
        for flow in flows:
          pkts = getattr(flow, "packets", [])
          if not pkts:
            continue

          times = [p[0] for p in pkts]
          lengths = [p[1] for p in pkts]
          directions = [p[2] for p in pkts]  # 1: Fwd, -1: Bwd
          flags = [p[3] for p in pkts]

          # 上り（Fwd）と下り（Bwd）の分離計算
          fwd_lengths = [l for l, d in zip(lengths, directions) if d == 1]
          bwd_lengths = [l for l, d in zip(lengths, directions) if d == -1]

          fwd_len_mean = float(np.mean(fwd_lengths)) if fwd_lengths else 0.0
          bwd_len_mean = float(np.mean(bwd_lengths)) if bwd_lengths else 0.0

          total_fwd = len(fwd_lengths)
          total_bwd = len(bwd_lengths)

          down_up = (
              float(total_bwd / total_fwd)
              if total_fwd > 0
              else (1.0 if total_bwd > 0 else 0.0)
          )
          syn_cnt = sum(1 for flg in flags if (flg & 0x02))

          # 継続時間（マイクロ秒）
          if len(times) > 1:
            dur = (max(times) - min(times)) * 1e6
          elif hasattr(flow, "start_time") and hasattr(flow, "end_time"):
            dur = (flow.end_time - flow.start_time) * 1e6
          else:
            dur = 0.0

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
              "SYN_Count": syn_cnt,
          })

        df_flows = pd.DataFrame(flow_records)
        if df_flows.empty:
          st.warning("解析可能な有効パケットが含まれていませんでした。")
          st.stop()

        # 3. 25フロー時間窓ブロック集約（送信元IP単位）
        agg_blocks = []
        for src_ip, group in df_flows.groupby("Src_IP"):
          for i in range(0, len(group), window_size):
            chunk = group.iloc[i : i + window_size]
            if len(chunk) == 0:
              continue

            u_ports = chunk["Dst_Port"].nunique()
            p_ent = calculate_entropy(chunk["Dst_Port"])
            zero_ratio = (chunk["Fwd_Pkt_Mean"] == 0).mean()
            avg_dur = chunk["Flow_Duration"].mean()

            # 新特徴量の正確な集約
            mean_bwd_len = chunk["Bwd_Pkt_Mean"].mean()
            fwd_bwd_ratio = (
                chunk["Total_Fwd_Pkts"]
                / (chunk["Total_Fwd_Pkts"] + chunk["Total_Bwd_Pkts"] + 1e-5)
            ).mean()
            mean_down_up = chunk["Down_Up_Ratio"].mean()
            syn_ratio = chunk["SYN_Count"].mean()

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
                "SYN_Flag_Ratio": syn_ratio,
            })

        df_agg = pd.DataFrame(agg_blocks)

        # 4. LightGBM 8次元推論実行
        feature_cols = [
            "Unique_Dst_Ports",
            "Port_Entropy",
            "Zero_Payload_Ratio",
            "Avg_Flow_Duration",
            "Mean_Bwd_Pkt_Len",
            "Fwd_Bwd_Pkt_Ratio",
            "Down_Up_Ratio",
            "SYN_Flag_Ratio",
        ]
        # 学習時のカラム名順序に厳密に合わせる
        use_cols = trained_features if trained_features else feature_cols
        X_input = df_agg[use_cols]

        probs = model.predict_proba(X_input)[:, 1]
        df_agg["Anomaly_Score"] = probs

        # --- 画面表示 ---
        st.success("全パイプラインの解析が完了しました！")

        # サマリーメトリクス
        total_blocks = len(df_agg)
        attack_blocks = df_agg[df_agg["Anomaly_Score"] >= threshold]
        attack_ratio = (
            (len(attack_blocks) / total_blocks * 100)
            if total_blocks > 0
            else 0.0
        )

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
          st.error(
              f"🚨 **悪意あるスキャン送信元ホストを特定: {', '.join(attacker_ips)}**"
          )

          st.write("##### 異常と判定された行動ブロック一覧")
          display_cols = [
              "Src_IP",
              "Anomaly_Score",
              "Unique_Dst_Ports",
              "Port_Entropy",
              "Mean_Bwd_Pkt_Len",
              "Fwd_Bwd_Pkt_Ratio",
              "Observed_Flows",
          ]
          st.dataframe(
              attack_blocks[display_cols].sort_values(
                  "Anomaly_Score", ascending=False
              )
          )

          # Wireshark用フィルタ
          st.write("##### 🔍 調査用 Wireshark フィルタ")
          filter_str = " || ".join([f"ip.src == {ip}" for ip in attacker_ips])
          st.code(filter_str, language="bash")
        else:
          st.info("指定されたしきい値を超える不審なスキャン行動は検出されませんでした。")

        # 詳細ドリルダウン
        with st.expander("📊 全ホストの行動集約特徴量を展開"):
          st.dataframe(df_agg)

      finally:
        if os.path.exists(tmp_pcap_path):
          os.remove(tmp_pcap_path)