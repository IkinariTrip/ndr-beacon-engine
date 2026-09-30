import numpy as np
import pandas as pd

DETERMINED_15_FEATURES = [
    'PSH Flag Count',
    'Fwd Header Length',
    'Fwd IAT Std',
    'min_seg_size_forward',
    'Avg Fwd Segment Size',
    'Init_Win_bytes_forward',
    'Fwd Packet Length Mean',
    'Flow IAT Mean',
    'Idle Max',
    'Packet Length Variance',
    'Bwd Packet Length Std',
    'Packet Length Std',
    'Fwd Packet Length Std',
    'Active Min',
    'Init_Win_bytes_backward'
]

def calculate_flow_features(flow):
    pkts = flow.packets
    times = [p[0] for p in pkts]
    lengths = [p[1] for p in pkts]
    directions = [p[2] for p in pkts]
    flags = [p[3] for p in pkts]
    windows = [p[4] for p in pkts]
    headers = [p[5] for p in pkts]

    fwd_lengths = [l for l, d in zip(lengths, directions) if d == 1]
    bwd_lengths = [l for l, d in zip(lengths, directions) if d == -1]
    fwd_times = [t for t, d in zip(times, directions) if d == 1]

    psh_cnt = sum(1 for flg in flags if flg & 0x08)
    fwd_header_len = sum(h for h, d in zip(headers, directions) if d == 1)

    if len(fwd_times) > 1:
        fwd_iats = np.diff(fwd_times) * 1e6
        fwd_iat_std = float(np.std(fwd_iats))
    else:
        fwd_iat_std = 0.0

    fwd_headers = [h for h, d in zip(headers, directions) if d == 1]
    min_seg_fwd = min(fwd_headers) if fwd_headers else 0

    fwd_len_mean = float(np.mean(fwd_lengths)) if fwd_lengths else 0.0
    avg_fwd_seg_size = fwd_len_mean

    init_win_fwd = 0
    init_win_bwd = 0
    for w, d in zip(windows, directions):
        if d == 1 and init_win_fwd == 0:
            init_win_fwd = w
        elif d == -1 and init_win_bwd == 0:
            init_win_bwd = w
        if init_win_fwd != 0 and init_win_bwd != 0:
            break

    if len(times) > 1:
        flow_iats = np.diff(times) * 1e6
        flow_iat_mean = float(np.mean(flow_iats))
        flow_iat_diffs = np.diff(times)
    else:
        flow_iat_mean = 0.0
        flow_iat_diffs = []

    idle_thresh = 1.0
    idles = [dt * 1e6 for dt in flow_iat_diffs if dt >= idle_thresh]
    actives = [dt * 1e6 for dt in flow_iat_diffs if dt < idle_thresh]
    idle_max = float(max(idles)) if idles else 0.0
    active_min = float(min(actives)) if actives else 0.0

    pkt_len_var = float(np.var(lengths)) if lengths else 0.0
    pkt_len_std = float(np.std(lengths)) if lengths else 0.0
    bwd_len_std = float(np.std(bwd_lengths)) if len(bwd_lengths) > 1 else 0.0
    fwd_len_std = float(np.std(fwd_lengths)) if len(fwd_lengths) > 1 else 0.0

    features = {
        'PSH Flag Count': psh_cnt,
        'Fwd Header Length': fwd_header_len,
        'Fwd IAT Std': fwd_iat_std,
        'min_seg_size_forward': min_seg_fwd,
        'Avg Fwd Segment Size': avg_fwd_seg_size,
        'Init_Win_bytes_forward': init_win_fwd,
        'Fwd Packet Length Mean': fwd_len_mean,
        'Flow IAT Mean': flow_iat_mean,
        'Idle Max': idle_max,
        'Packet Length Variance': pkt_len_var,
        'Bwd Packet Length Std': bwd_len_std,
        'Packet Length Std': pkt_len_std,
        'Fwd Packet Length Std': fwd_len_std,
        'Active Min': active_min,
        'Init_Win_bytes_backward': init_win_bwd
    }

    metadata = {
        'Start_Time': flow.start_time,
        'Src_IP': flow.src_ip,
        'Dst_IP': flow.dst_ip,
        'Src_Port': flow.src_port,
        'Dst_Port': flow.dst_port,
        'Protocol': flow.protocol,
        'Total_Packets': len(pkts)
    }

    return features, metadata

def build_feature_dataframe(flows):
    feat_list = []
    meta_list = []

    for f in flows:
        feat, meta = calculate_flow_features(f)
        feat_list.append(feat)
        meta_list.append(meta)

    df_features = pd.DataFrame(feat_list)
    df_meta = pd.DataFrame(meta_list)

    if df_features.empty:
        return pd.DataFrame(columns=DETERMINED_15_FEATURES), df_meta

    df_features = df_features[DETERMINED_15_FEATURES]
    df_features = df_features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df_features, df_meta