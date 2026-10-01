import numpy as np
import pandas as pd

ENGINE_B_8_FEATURES = [
    'IAT_Median',
    'IAT_MAD',
    'IAT_CV',
    'Periodic_Peak_Ratio',
    'Burst_Active_Ratio',
    'Payload_Bytes_MAD',
    'Byte_Ratio_Median',
    'Connection_Count'
]

def calculate_engine_b_features(flow_group_df):
    """
    Src_IP × Dst_IP ペア単位の時系列フロー群から、
    仕様書 v2.4 に準拠した8つのロバスト物理特徴量を算出する
    """
    if flow_group_df.empty:
        return {feat: 0.0 for feat in ENGINE_B_8_FEATURES}

    # フロー間隔 (IAT) の算出
    # flow_group_df は時間順にソートされている前提
    if 'Start_Time' in flow_group_df.columns and len(flow_group_df) > 1:
        times = flow_group_df['Start_Time'].values
        iats = np.diff(times)
    else:
        iats = np.array([0.0])

    if len(iats) > 0:
        iat_median = float(np.median(iats))
        iat_mad = float(np.median(np.abs(iats - iat_median)))
        iat_mean = float(np.mean(iats))
        iat_std = float(np.std(iats))
        iat_cv = float(iat_std / iat_mean) if iat_mean > 0 else 0.0
    else:
        iat_median, iat_mad, iat_cv = 0.0, 0.0, 0.0

    # 自己相関第1ピーク強度比 (Periodic_Peak_Ratio) の簡易算出
    # ※厳密なFFT/自己相関の計算のベースライン
    if len(iats) > 3 and iat_std > 0:
        norm_iats = (iats - iat_mean) / iat_std
        autocorr = np.correlate(norm_iats, norm_iats, mode='full')
        central_idx = len(autocorr) // 2
        # ラグ1以降の最大値比率などを評価
        if len(autocorr[central_idx + 1:]) > 0:
            periodic_peak_ratio = float(np.max(np.abs(autocorr[central_idx + 1:])) / max(len(iats), 1))
        else:
            periodic_peak_ratio = 0.0
    else:
        periodic_peak_ratio = 0.0

    # Burst / Active 比率の算出（アイドルしきい値を1.0秒と仮定）
    idle_thresh = 1.0
    if len(iats) > 0:
        active_dur = float(np.sum(iats[iats < idle_thresh]))
        idle_dur = float(np.sum(iats[iats >= idle_thresh]))
        total_dur = active_dur + idle_dur
        burst_active_ratio = float(active_dur / total_dur) if total_dur > 0 else 1.0
    else:
        burst_active_ratio = 1.0

    # ペイロードバイト数の中央絶対偏差 (Payload_Bytes_MAD)
    if 'Payload_Length' in flow_group_df.columns:
        lengths = flow_group_df['Payload_Length'].values
        bytes_median = float(np.median(lengths))
        payload_bytes_mad = float(np.median(np.abs(lengths - bytes_median)))
    else:
        payload_bytes_mad = 0.0

    # 上り/下りバイト数の中央値 (Byte_Ratio_Median)
    if 'Fwd_Bytes' in flow_group_df.columns and 'Bwd_Bytes' in flow_group_df.columns:
        fwd_b = flow_group_df['Fwd_Bytes'].values
        bwd_b = flow_group_df['Bwd_Bytes'].values
        total_b = fwd_b + bwd_b
        ratios = np.where(total_b > 0, fwd_b / total_b, 0.0)
        byte_ratio_median = float(np.median(ratios))
    else:
        byte_ratio_median = 0.0

    # 接続セッション総数 (Connection_Count)
    connection_count = int(len(flow_group_df))

    features = {
        'IAT_Median': iat_median,
        'IAT_MAD': iat_mad,
        'IAT_CV': iat_cv,
        'Periodic_Peak_Ratio': periodic_peak_ratio,
        'Burst_Active_Ratio': burst_active_ratio,
        'Payload_Bytes_MAD': payload_bytes_mad,
        'Byte_Ratio_Median': byte_ratio_median,
        'Connection_Count': connection_count
    }

    return features