"""
stage2_biflow.py
パケット単位のParquetから、双方向フロー（Biflow）をベクトル化処理で生成する
（仕様書v2.5 3.4節準拠）。

規則:
  - 5-tuple を方向正規化したキーでグルーピングする
  - TCPは120秒無通信、UDPは30秒無通信でフローを区切る
  - TCPはFINまたはRSTパケットの直後で区切る（簡略化した規則。詳細はREADME参照）
  - 開始側（Src）は、そのフロー最初のパケットの送信元とする

出力: biflows_<capture>.parquet
"""
import os
import sys

import numpy as np
import pandas as pd

from config import CAPTURES, INTERMEDIATE_DIR, TCP_TIMEOUT_SEC, UDP_TIMEOUT_SEC


def build_biflows(name):
    in_path = os.path.join(INTERMEDIATE_DIR, f"packets_{name}.parquet")
    out_path = os.path.join(INTERMEDIATE_DIR, f"biflows_{name}.parquet")
    print(f"\n===== 段階2（Biflow生成）: {name} =====")

    df = pd.read_parquet(in_path)
    print(f"  読み込み完了: {len(df):,} パケット")
    if df.empty:
        pd.DataFrame(columns=[
            "flow_uid", "src_ip", "src_port", "dst_ip", "dst_port", "proto",
            "start_ts", "end_ts", "duration", "fwd_pkts", "bwd_pkts",
            "fwd_bytes", "bwd_bytes",
        ]).to_parquet(out_path, index=False)
        print("[!] パケットが0件のため、空のBiflowファイルを出力しました")
        return

    ep_a = df["src_ip"] + ":" + df["src_port"].astype(str)
    ep_b = df["dst_ip"] + ":" + df["dst_port"].astype(str)
    lo = np.where(ep_a < ep_b, ep_a, ep_b)
    hi = np.where(ep_a < ep_b, ep_b, ep_a)
    df["_key"] = df["proto"].astype(str) + "|" + lo + "|" + hi

    df = df.sort_values(["_key", "ts"], kind="mergesort").reset_index(drop=True)

    timeout = np.where(df["proto"].values == "tcp", TCP_TIMEOUT_SEC, UDP_TIMEOUT_SEC)
    same_key = df["_key"].values[1:] == df["_key"].values[:-1]
    gap = df["ts"].values[1:] - df["ts"].values[:-1]
    prev_closed = (
        (df["proto"].values[:-1] == "tcp")
        & ((df["fin"].values[:-1] == 1) | (df["rst"].values[:-1] == 1))
    )
    new_flow = np.ones(len(df), dtype=bool)
    new_flow[1:] = (~same_key) | (gap > timeout[1:]) | prev_closed
    df["_local_flow_no"] = new_flow.cumsum()

    first_rows = df.groupby("_local_flow_no", sort=False).head(1)
    initiator = first_rows.set_index("_local_flow_no")[
        ["src_ip", "src_port", "dst_ip", "dst_port"]
    ].rename(columns={
        "src_ip": "_init_src_ip", "src_port": "_init_src_port",
        "dst_ip": "_init_dst_ip", "dst_port": "_init_dst_port",
    })
    df = df.join(initiator, on="_local_flow_no")

    is_fwd = (df["src_ip"] == df["_init_src_ip"]) & (df["src_port"] == df["_init_src_port"])
    df["_fwd_bytes"] = np.where(is_fwd, df["ip_len"], 0.0)
    df["_bwd_bytes"] = np.where(is_fwd, 0.0, df["ip_len"])
    df["_fwd_pkt"] = is_fwd.astype("int32")
    df["_bwd_pkt"] = (~is_fwd).astype("int32")

    agg = df.groupby("_local_flow_no", sort=False).agg(
        src_ip=("_init_src_ip", "first"), src_port=("_init_src_port", "first"),
        dst_ip=("_init_dst_ip", "first"), dst_port=("_init_dst_port", "first"),
        proto=("proto", "first"),
        start_ts=("ts", "min"), end_ts=("ts", "max"),
        fwd_pkts=("_fwd_pkt", "sum"), bwd_pkts=("_bwd_pkt", "sum"),
        fwd_bytes=("_fwd_bytes", "sum"), bwd_bytes=("_bwd_bytes", "sum"),
    ).reset_index(drop=True)
    agg["duration"] = agg["end_ts"] - agg["start_ts"]
    agg["flow_uid"] = [f"{name}_{i}" for i in range(len(agg))]
    agg = agg.sort_values("start_ts").reset_index(drop=True)

    agg.to_parquet(out_path, index=False)
    print(f"[+] 完了: {out_path}  Biflow数={len(agg):,}")
    return agg


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = list(CAPTURES) if "--all" in args else [a for a in args if not a.startswith("--")]
    if not targets:
        print("使い方: python stage2_biflow.py <capture名> [<capture名> ...] | --all")
        sys.exit(1)
    for name in targets:
        build_biflows(name)
