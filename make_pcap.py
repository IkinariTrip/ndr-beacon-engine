import time
import joblib
import pandas as pd
import scapy.all as scapy
from src.feature_engineering import build_feature_dataframe
from src.flow_generator import extract_flows_from_pcap

pcap_filename = "scan_detected_psh.pcap"
packets = []
base_time = time.time()

# 1. 正常通信（Webアクセス: 実データあり）
for i in range(5):
  t = base_time + i * 0.2
  p1 = (
      scapy.IP(src="192.168.1.50", dst="93.184.216.34")
      / scapy.TCP(sport=50000 + i, dport=80, flags="PA", window=65535)
      / ("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n" * 5)
  )
  p1.time = t
  packets.append(p1)

# 2. ポートスキャン通信（攻撃端末 10.0.0.99: PSHフラグ付きプローブ）
# 主力リーフ条件: Avg_Fwd <= 2.19, PSH > 0, Var > 4.25, Flow_IAT <= 14908
for port in [21, 22, 23, 80, 443, 3306, 8080]:
  t = base_time + 2.0 + (port * 0.005)
  # パケット1: 攻撃側 PSH+SYN プローブ (データ長 0バイト)
  p_probe = scapy.IP(src="10.0.0.99", dst="192.168.1.10") / scapy.TCP(
      sport=45000 + port, dport=port, flags="PS", window=1024
  )
  p_probe.time = t
  # パケット2: サーバー側 拒絶応答 (データ長 10バイト)
  p_res = (
      scapy.IP(src="192.168.1.10", dst="10.0.0.99")
      / scapy.TCP(sport=port, dport=45000 + port, flags="RA", window=0)
      / ("REJECTED!!")
  )
  p_res.time = t + 0.002
  packets.extend([p_probe, p_res])

scapy.wrpcap(pcap_filename, packets)
print(f"=== 1. 新規ファイル作成完了: {pcap_filename} ===")

# パイプラインで推論実行
flows, _ = extract_flows_from_pcap(pcap_filename)
X_features, df_meta = build_feature_dataframe(flows)

payload = joblib.load("models/anomaly_detector.joblib")
model = (
    payload["model"]
    if isinstance(payload, dict) and "model" in payload
    else payload
)

X_input = X_features.copy()
X_input.columns = [c.replace(" ", "_") for c in X_input.columns]
scores = model.predict_proba(X_input)[:, 1]

df_meta["Anomaly_Score"] = scores
df_meta["PSH_Cnt"] = X_features["PSH Flag Count"]

print("\n=== 2. AIモデルの判定スコア一覧 ===")
print(
    df_meta[["Src_IP", "Dst_Port", "Anomaly_Score", "PSH_Cnt"]].to_string(
        index=False
    )
)