# PCAP Anomaly Detector (NDR Beacon Engine)

ネットワーク検知・対応 (NDR) のためのパケット解析および、機械学習ベースの異常検知（C2ビーコン・通信パターン検知）開発プロジェクトです。

## 概要
本プロジェクトでは、Mirai C2トラフィック（正例）と、IoT実機（Philips Hue、Amazon Echo等）および通常のWebブラウジングトラフィック（負例）を組み合わせたハイブリッドデータセットを用い、パケット到着間隔 (IAT) やパケット長などの統計的特徴量抽出から異常検知を行うパイプラインを構築しています。

## 主要スクリプトとファイル構成
- **`app_v4.py`**: **【エンジンAの完成形】** 特徴量抽出から機械学習モデルによる判定ロジックまでが統合されたメイン実行スクリプト。
- **`src/`**
  - `flow_generator.py`: パケットをBiflow（双方向フロー）に分割・生成するモジュール。
  - `feature_engineering.py`: 統計的特徴量（中央値、MAD等）を抽出するモジュール。
- **`models/`**: 学習済みモデル（LightGBM、CatBoost等）のバイナリ（`.joblib`）を格納。
- **`raw_pcap/`**: 実PCAPファイル格納ディレクトリ（※数100MB〜1GB超の巨大ファイルのため、`.gitignore` によりGitHub管理外としローカル/Google Driveで保持）。

## 実行方法
仮想環境を有効化した上で、エンジンAの完成形である `app_v4.py` を実行します。

```bash
# 仮想環境の有効化（必要な場合）
source venv/bin/activate

# エンジンA（完成形）の実行
python app_v4.py