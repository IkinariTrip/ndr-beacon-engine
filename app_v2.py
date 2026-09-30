import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st

from src.feature_engineering import build_feature_dataframe
from src.flow_generator import extract_flows_from_pcap

# --- ページ基本設定 ---
st.set_page_config(
    page_title="【仮称】インシデンスコックピット改２",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ 【仮称】インシデンスコックピット改２")
st.caption(
    "次世代ホスト行動プロファイリング監視エンジン (推論モデル: LightGBM"
    " GBDT / F1: 99.30%)"
)

# --- モデル読み込み ---
MODEL_PATH = "models/model_lightgbm.joblib"


@st.cache_resource
def load_detection_model():
  if not os.path.exists(MODEL_PATH):
    return None
  payload = joblib.load(MODEL_PATH)
  return (
      payload["model"]
      if isinstance(payload, dict) and "model" in payload
      else payload
  )


model = load_detection_model()
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
    **新機能 (改１):**
    - 単一Flowから時間窓行動集約へ転換
    - **TCPステート判定による受動応答（SYN/ACK）の完全排除**
    - ポート番号偽装攻撃（-g オプション）への耐性獲得
    - 特徴量: ポート数 / エントロピー / ペイロード比率
    - 推論エンジン: 高精度 LightGBM
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
    with st.spinner("生パケットを抽出・時間窓集約中..."):
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

        # 2. 既存パイプラインからメタ情報と特徴量を安全に取得
        X_raw, df_meta = build_feature_dataframe(flows)
        df_flows = df_meta.copy()
        df_flows["Fwd_Pkt_Mean"] = X_raw["Fwd Packet Length Mean"]

        # フロー継続時間（マイクロ秒）を安全に取得
        durations = []
        for idx, flow in enumerate(flows):
          d = 0.0
          if hasattr(flow, "get_duration"):
            d = flow.get_duration() * 1e6
          elif hasattr(flow, "duration"):
            d = flow.duration * 1e6
          elif hasattr(flow, "start_time") and hasattr(flow, "end_time"):
            d = (flow.end_time - flow.start_time) * 1e6
          elif "Flow IAT Mean" in X_raw.columns:
            d = X_raw.iloc[idx]["Flow IAT Mean"] * df_flows.iloc[idx].get(
                "Total_Packets", 1
            )
          durations.append(d)
        df_flows["Flow_Duration"] = durations

        # =========================================================================
        # ▼▼▼【新規追加】2.5 サーバー受動応答（SYN/ACK等）のフィルタリング ▼▼▼
        # =========================================================================
        # サーバーがクライアントの動的ポートへ返す受動パケットはスキャンではないため排除。
        # 単純なポート除外ではなく、TCPステート（SYN単体の自発接続か）を判定基準とする。
        syn_cols = [
            c
            for c in X_raw.columns
            if "SYN" in c.upper() and "FLAG" in c.upper()
        ]
        ack_cols = [
            c
            for c in X_raw.columns
            if "ACK" in c.upper() and "FLAG" in c.upper()
        ]
        syn_col = syn_cols[0] if syn_cols else None
        ack_col = ack_cols[0] if ack_cols else None

        valid_flow_mask = []
        for idx, flow in enumerate(flows):
          is_passive_response = False

          # A. flowオブジェクトから直接パケット/フラグを取得できる場合（Scapyベース等）
          if hasattr(flow, "packets") and len(flow.packets) > 0:
            first_pkt = flow.packets[0]
            if hasattr(first_pkt, "haslayer") and first_pkt.haslayer("TCP"):
              flags = int(first_pkt["TCP"].flags)
              is_syn = bool(flags & 0x02)
              is_ack = bool(flags & 0x10)
              payload_len = len(first_pkt["TCP"].payload)

              # SYN/ACK (0x12) はサーバーからの応答
              if is_syn and is_ack:
                is_passive_response = True
              # 単なるACK応答で中身（ペイロード）もない通信は除外
              elif not is_syn and payload_len == 0:
                is_passive_response = True

          # B. X_raw のフラグ統計カラムから判定できる場合
          elif syn_col and ack_col:
            syn_cnt = X_raw.iloc[idx][syn_col]
            ack_cnt = X_raw.iloc[idx][ack_col]
            fwd_len = df_flows.iloc[idx].get("Fwd_Pkt_Mean", 0)
            if syn_cnt > 0 and ack_cnt > 0:
              is_passive_response = True
            elif syn_cnt == 0 and ack_cnt > 0 and fwd_len == 0:
              is_passive_response = True

          # C. メタ情報によるフォールバック（インフラポートからの戻り通信判定）
          sport = df_flows.iloc[idx].get("Src_Port", None)
          dport = df_flows.iloc[idx].get("Dst_Port", None)
          fwd_len = df_flows.iloc[idx].get("Fwd_Pkt_Mean", 0)
          if sport is not None and dport is not None:
            # サーバー側サービスポート（DNS, Kerberos, RPC, LDAP, SMB）からクライアント動的ポートへの空応答
            infra_ports = {53, 88, 135, 389, 445}
            if sport in infra_ports and dport > 1024 and fwd_len == 0:
              is_passive_response = True

          valid_flow_mask.append(not is_passive_response)

        # フィルタを適用し、自発的接続試行（スキャン等）と実データ通信のみを残す
        df_flows = df_flows[valid_flow_mask].reset_index(drop=True)
        # ▲▲▲【追加ここまで】▲▲▲

        if df_flows.empty:
          st.info(
              "パケットを精査しましたが、監視対象となる自発的な接続試行フローは存在しませんでした。"
          )
          st.stop()

        # 3. ホスト行動集約（送信元IP × ウィンドウサイズで集約）
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

            agg_blocks.append({
                "Src_IP": src_ip,
                "Observed_Flows": len(chunk),
                "Unique_Dst_Ports": u_ports,
                "Port_Entropy": p_ent,
                "Zero_Payload_Ratio": zero_ratio,
                "Avg_Flow_Duration": avg_dur,
            })

        df_agg = pd.DataFrame(agg_blocks)

        # 4. LightGBM推論
        feature_cols = [
            "Unique_Dst_Ports",
            "Port_Entropy",
            "Zero_Payload_Ratio",
            "Avg_Flow_Duration",
        ]
        X_input = df_agg[feature_cols]
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
              "Zero_Payload_Ratio",
              "Avg_Flow_Duration",
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