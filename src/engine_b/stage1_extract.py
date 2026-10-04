"""
stage1_extract.py
生PCAPから tshark でパケット情報を抽出し、キャプチャごとに
Parquetファイル（packets_<capture>.parquet）を作成する（仕様書v2.5 3.3節）。
"""
import os
import sys
import shutil
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CAPTURES, INTERMEDIATE_DIR
from common import find_pcap

FIELDS = [
    "frame.time_epoch", "ip.src", "ip.dst", "ip.proto",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "ip.len", "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
]
NUMERIC_FIELDS = [
    "frame.time_epoch", "ip.proto", "tcp.srcport", "tcp.dstport",
    "udp.srcport", "udp.dstport", "ip.len",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
]


def run_tshark(pcap_path, tsv_path):
    if shutil.which("tshark") is None:
        raise RuntimeError("tshark が見つかりません。`brew install wireshark` 等で導入してください。")
    cmd = [
        "tshark", "-r", pcap_path, "-Y", "ip && (tcp || udp)",
        "-T", "fields", "-E", "separator=/t", "-E", "header=y", "-E", "occurrence=f",
    ]
    for f in FIELDS:
        cmd += ["-e", f]
    print("[*] tshark 実行中:", pcap_path)
    with open(tsv_path, "w") as out:
        subprocess.run(cmd, stdout=out, check=True)


def normalize_chunk(chunk):
    for c in NUMERIC_FIELDS:
        if c in chunk.columns:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    chunk = chunk.rename(columns={
        "frame.time_epoch": "ts", "ip.src": "src_ip", "ip.dst": "dst_ip",
        "ip.len": "ip_len",
        "tcp.flags.syn": "syn", "tcp.flags.ack": "ack",
        "tcp.flags.fin": "fin", "tcp.flags.reset": "rst",
    })
    chunk["src_port"] = chunk["tcp.srcport"].fillna(chunk["udp.srcport"])
    chunk["dst_port"] = chunk["tcp.dstport"].fillna(chunk["udp.dstport"])
    chunk["proto"] = chunk["ip.proto"].map({6: "tcp", 17: "udp"})

    for c in ("syn", "ack", "fin", "rst"):
        chunk[c] = chunk[c].fillna(0).astype("int8")
    chunk["ip_len"] = chunk["ip_len"].fillna(0).astype("float64")

    chunk = chunk.dropna(subset=["src_ip", "dst_ip", "src_port", "dst_port", "proto", "ts"])
    chunk["src_port"] = chunk["src_port"].astype("int32")
    chunk["dst_port"] = chunk["dst_port"].astype("int32")

    keep = ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
            "ip_len", "syn", "ack", "fin", "rst"]
    chunk = chunk[keep].sort_values("ts")
    chunk = chunk.drop_duplicates(
        subset=["ts", "src_ip", "dst_ip", "src_port", "dst_port", "ip_len"]
    )
    return chunk


def extract_one(name):
    cap = CAPTURES[name]
    pcap_path = find_pcap(cap["dir"])
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    tsv_path = os.path.join(INTERMEDIATE_DIR, f"packets_{name}.tsv")
    out_path = os.path.join(INTERMEDIATE_DIR, f"packets_{name}.parquet")

    print(f"\n===== 段階1（パケット抽出）: {name} =====")
    print(f"  対象PCAP: {pcap_path}")
    run_tshark(pcap_path, tsv_path)

    writer = None
    n_rows = 0
    reader = pd.read_csv(tsv_path, sep="\t", dtype=str, chunksize=500_000, low_memory=False)
    for chunk in reader:
        chunk = normalize_chunk(chunk)
        if chunk.empty:
            continue
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_rows += len(chunk)
        print(f"  ... {n_rows:,} 件処理済み")
    if writer is not None:
        writer.close()

    os.remove(tsv_path)
    print(f"[+] 完了: {out_path}（{n_rows:,} パケット）")
    return n_rows


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = list(CAPTURES) if "--all" in args else [a for a in args if not a.startswith("--")]
    if not targets:
        print("使い方: python stage1_extract.py <capture名> [<capture名> ...] | --all")
        sys.exit(1)
    for name in targets:
        extract_one(name)
