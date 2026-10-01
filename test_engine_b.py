import os
import numpy as np
import pandas as pd
from src.flow_generator import extract_flows_from_pcap
from src.feature_engineering_engine_b import calculate_engine_b_features, ENGINE_B_8_FEATURES

# raw_pcap 内の検証用PCAPファイルパス（例としてPhilips HueのHeartbeatを使用）
target_pcap = "raw_pcap/heartbeat_philipshue.pcap"

if not os.path.exists(target_pcap):
    print(f"⚠️ 対象ファイルが見つかりません: {target_pcap}")
else:
    print(f"[*] 解析対象PCAPの読み込みとフロー生成: {target_pcap}")
    flows, total_packets = extract_flows_from_pcap(target_pcap)
    print(f"[*] 総パケット数: {total_packets:,} / 生成フロー数: {len(flows):,}")

    if flows:
        # フロー情報をDataFrameに変換して集約・特徴量抽出のテスト
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

        # Src_IP × Dst_IP ペア単位でグルーピングしてエンジンBの特徴量を算出
        print("[*] 仕様書 v2.4 準拠の8ロバスト物理特徴量を算出中...")
        results = []
        for (src, dst), group in df_flows.groupby(["Src_IP", "Dst_IP"]):
            sorted_group = group.sort_values("Start_Time")
            feats = calculate_engine_b_features(sorted_group)
            feats["Src_IP"] = src
            feats["Dst_IP"] = dst
            results.append(feats)

        df_features = pd.DataFrame(results)
        
        print("\n--- 抽出されたエンジンB特徴量 (先頭数行) ---")
        print(df_features[ENGINE_B_8_FEATURES + ["Src_IP", "Dst_IP"]].head())
        print(f"\n[+] 特徴量マトリクス形状: {df_features.shape}")
    else:
        print("⚠️ 有効なフローが検出されませんでした。")