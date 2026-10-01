import os
import gc
import numpy as np
import pandas as pd
from src.flow_generator import extract_flows_from_pcap
from src.feature_engineering_engine_b import calculate_engine_b_features

def process_pcap_to_features(pcap_path, label):
    print(f"\n[*] 処理開始: {pcap_path} (Label={label})")
    if not os.path.exists(pcap_path):
        print(f"⚠ ファイルが存在しません: {pcap_path}")
        return pd.DataFrame()

    # 巨大ファイル処理前のメモリクリーンアップ
    gc.collect()

    try:
        # パケット読み込みとフロー生成
        flows, total_pkts = extract_flows_from_pcap(pcap_path)
        print(f"[*] 総パケット数: {total_pkts:,} / 生成フロー数: {len(flows) if flows else 0}")
        
        if not flows:
            print(f"⚠️ フローが検出されませんでした: {pcap_path}")
            return pd.DataFrame()

        flow_records = []
        for flow in flows:
            pkts = getattr(flow, "packets", [])
            if not pkts:
                continue
            times = [p[0] for p in pkts]
            lengths = [p[1] for p in pkts]
            directions = [p[2] for p in pkts]
            
            fwd_bytes = sum(l for l, d in zip(lengths, directions) if d == 1)
            bwd_bytes = sum(l for l, d in zip(lengths, directions) if d == -1)
            
            flow_records.append({
                "Src_IP": flow.src_ip,
                "Dst_IP": flow.dst_ip,
                "Start_Time": flow.start_time,
                "Payload_Length": np.mean(lengths) if lengths else 0.0,
                "Fwd_Bytes": fwd_bytes,
                "Bwd_Bytes": bwd_bytes
            })

        df_flows = pd.DataFrame(flow_records)
        if df_flows.empty:
            return pd.DataFrame()

        results = []
        for (src, dst), group in df_flows.groupby(["Src_IP", "Dst_IP"]):
            sorted_group = group.sort_values("Start_Time")
            feats = calculate_engine_b_features(sorted_group)
            
            # 観測量フィルタ（仕様書準拠: Connection_Count < 10 の単発通信はノイズとして除外）
            if feats['Connection_Count'] < 10:
                continue
                
            feats["Src_IP"] = src
            feats["Dst_IP"] = dst
            feats["Label"] = label
            results.append(feats)

        df_features = pd.DataFrame(results)
        
        # 処理完了後にメモリを明示的に解放
        del flows, flow_records, df_flows
        gc.collect()
        
        return df_features

    except Exception as e:
        print(f"❌ エラー発生 ({pcap_path}): {e}")
        gc.collect()
        return pd.DataFrame()

if __name__ == "__main__":
    # データセットソースの定義（正例C2には Label=1, 負例には Label=0 を付与）
    dataset_sources = [
        # 負例A: IoT実機 Heartbeat群 (Label = 0)
        ("raw_pcap/heartbeat_philipshue.pcap", 0),
        ("raw_pcap/2018-09-21-11-40-22-192.168.2.3.pcap", 0),
        # 負例B: 人間操作Webトラフィック (Label = 0)
        ("raw_pcap/human_web_normal27.pcap", 0),
        # 正例C2: Mirai C2 ビーコン通信本体 (Label = 1)
        ("raw_pcap/2019-07-03-15-15-47-192.168.1.158.pcap", 1),
    ]

    all_dfs = []
    for path, lbl in dataset_sources:
        df_part = process_pcap_to_features(path, lbl)
        if not df_part.empty:
            all_dfs.append(df_part)
            print(f"[+] 特徴量抽出成功: 形状 {df_part.shape}")

    if all_dfs:
        df_dataset = pd.concat(all_dfs, ignore_index=True)
        
        # クレンジング処理（無限大・欠損値の置換）
        df_dataset = df_dataset.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        os.makedirs("models", exist_ok=True)
        out_path = "models/dataset_engine_b.csv"
        df_dataset.to_csv(out_path, index=False)
        
        print(f"\n[+] 統合データセットの構築完了！ 出力先: {out_path}")
        print(f"[+] 統合データ形状: {df_dataset.shape}")
        print("\n--- クラス別サンプル数 (Label) ---")
        print(df_dataset["Label"].value_counts())
    else:
        print("⚠️ 有効な特徴量が抽出されたデータがありませんでした。")