"""
config.py
エンジンB データ構築パイプライン設定ファイル（仕様書v2.5準拠）

このファイルだけを編集すれば、パス・対象キャプチャ・ラベル規則を調整できる。
他のファイル（stage1〜4）は基本的に触らなくてよい。
"""
import os

# ===== パス設定 =====
BASE_DIR = os.environ.get(
    "ENGINE_B_BASE_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "raw_pcap", "engine_b_iot23",
    ),
)
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "_intermediate")
OUTPUT_DIR = os.path.join(BASE_DIR, "_output")

# ===== ブロック生成の条件（3.6節） =====
WINDOW_SIZE = 20
WINDOW_STRIDE = 5
MIN_PAIR_FLOWS = 10

# ===== Biflowのタイムアウト（3.4節） =====
TCP_TIMEOUT_SEC = 120
UDP_TIMEOUT_SEC = 30

# ===== Zeekラベルとの時刻照合の許容誤差（3.5節） =====
LABEL_MATCH_TOLERANCE_SEC = 1.0

# =====================================================================
# キャプチャ台帳（2.2節・付録A.1準拠）
# =====================================================================
CAPTURES = {
    "7-1": dict(
        dir="01_Train_Positive/CTU-IoT-Malware-Capture-7-1",
        split="train", host_ip="192.168.100.108",
        c2_server_ips=["185.130.215.13"],
    ),
    "48-1": dict(
        dir="01_Train_Positive/CTU-IoT-Malware-Capture-48-1",
        split="train", host_ip="192.168.1.200",
        c2_server_ips=["167.99.182.238"],
    ),
    "20-1": dict(
        dir="03_Train_Negative/CTU-IoT-Malware-Capture-20-1",
        split="train", host_ip="192.168.100.103",
        c2_server_ips=["66.85.157.90"], reference_only=True,
    ),
    "3-1": dict(
        dir="03_Train_Negative/CTU-IoT-Malware-Capture-3-1",
        split="train", host_ip="192.168.2.5",
        c2_server_ips=[],
    ),
    "hue-4-1": dict(
        dir="03_Train_Negative/CTU-Honeypot-Capture-4-1",
        split="train", host_ip="192.168.1.132",
        c2_server_ips=[],
    ),
    "34-1": dict(
        dir="02_Test_Positive/CTU-IoT-Malware-Capture-34-1",
        split="test", host_ip="192.168.1.195",
        c2_server_ips=["185.244.25.235"],
    ),
    "8-1": dict(
        dir="02_Test_Positive/CTU-IoT-Malware-Capture-8-1",
        split="test", host_ip="192.168.100.113",
        c2_server_ips=["178.128.185.250", "128.185.250.50"],
    ),
    "21-1": dict(
        dir="04_Test_Negative/CTU-IoT-Malware-Capture-21-1",
        split="test", host_ip="192.168.100.113",
        c2_server_ips=["66.85.157.90"], reference_only=True,
    ),
    "echo-5-1": dict(
        dir="04_Test_Negative/CTU-Honeypot-Capture-5-1",
        split="test", host_ip="192.168.2.3",
        c2_server_ips=[],
    ),
}
for _cap in CAPTURES.values():
    _cap.setdefault("reference_only", False)

KNOWN_C2_NETWORKS = [
    "185.130.215.0/24", "167.99.182.0/24", "185.244.25.0/24",
    "178.128.185.0/24", "128.185.250.0/24", "66.85.157.0/24",
]

POSITIVE_DETAILED_LABELS = {
    "7-1": {"command-and-control"},
    "34-1": {"c&c"},
    "8-1": {"c&c"},
    "48-1": {"c&c", "c&c-heartbeat-attack", "c&c-partofahorizontalportscan",
              "c&c-heartbeat-filedownload"},
}
POSITIVE_REQUIRES_C2_PEER = {"48-1"}

EXCLUDE_DETAILED_LABELS = {
    "command-and-control-reflection", "horizontal-scan", "horizontal-scan-reflection",
    "dos", "dos-gre", "dos-reflection", "ddos", "partofahorizontalportscan",
    "attack", "from-malware", "filedownload", "c&c-filedownload", "answers",
}

REFERENCE_ONLY_DETAILED_LABELS = {"c&c-torii"}

# ===== モデル比較用（本リリースで追加） =====
# A系統（集計特徴量）: 木構造 + 全結合 + 表形式注目機構
# B系統（生の時系列） : 畳み込み/再帰/時系列畳み込み
MODEL_LIST_A = ["random_forest", "xgboost", "lightgbm", "catboost", "mlp", "tabnet"]
MODEL_LIST_B = ["cnn1d", "gru", "tcn"]

# 生の時系列として書き出す系列（IP・ポート・サービス名は含めない。2026-09-30 合意事項）
RAW_SERIES = ["iat", "fwd_bytes", "bwd_bytes"]
