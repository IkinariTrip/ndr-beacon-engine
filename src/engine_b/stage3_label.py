"""
stage3_label.py
BiflowにZeekラベルを照合し、Label(1=C2 / 0=正常 / 除外)を確定する
（仕様書v2.5 2.4節・3.5節準拠）。
出力: labeled_<capture>.parquet
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from config import (
    CAPTURES, INTERMEDIATE_DIR, LABEL_MATCH_TOLERANCE_SEC,
    POSITIVE_DETAILED_LABELS, POSITIVE_REQUIRES_C2_PEER,
    EXCLUDE_DETAILED_LABELS, REFERENCE_ONLY_DETAILED_LABELS,
)
from common import find_labeled, is_external, is_known_c2

# 正常な背景通信（プロトコル名がそのままdetailed-labelに入っているケース）
BENIGN_PROTOCOL_LABELS = {"ntp", "dhcp", "mdns", "dns", "ssdp", "http", "https", "ssl", "ssh"}


def read_zeek_labeled(path):
    with open(path, errors="ignore") as f:
        cols = None
        for line in f:
            if line.startswith("#fields"):
                raw = line.rstrip("\n")
                cols = re.split(r"\t+| {2,}", raw)[1:]
                cols = [c for c in cols if c != ""]
                break
    if cols is None:
        raise ValueError(f"#fields 行が見つかりません: {path}")
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, names=cols,
        low_memory=False, na_values=["-"],
    )
    df.columns = [str(c).strip() for c in df.columns]
    detail_col = None
    for c in df.columns:
        if c.replace("-", "").replace("_", "").lower() == "detailedlabel":
            detail_col = c
    if "label" not in df.columns or detail_col is None:
        raise ValueError(f"label/detailed-label 列が見つかりません: {path}（列: {list(df.columns)}）")

    df["label"] = df["label"].astype(str).str.lower()
    df["detailed_label"] = df[detail_col].astype(str).str.lower()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    if "conn_state" not in df.columns:
        df["conn_state"] = None

    keep = ["ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "conn_state", "label", "detailed_label"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["ts", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p"])
    df["id.orig_p"] = df["id.orig_p"].astype(int)
    df["id.resp_p"] = df["id.resp_p"].astype(int)
    return df


def make_key(ip_a, port_a, ip_b, port_b, proto):
    a = ip_a.astype(str) + ":" + port_a.astype(int).astype(str)
    b = ip_b.astype(str) + ":" + port_b.astype(int).astype(str)
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    return proto.astype(str) + "|" + lo + "|" + hi


def classify(name, detailed_label, peer_ip, c2_ips, zeek_label=None):
    pos_set = POSITIVE_DETAILED_LABELS.get(name, set())
    if detailed_label in pos_set:
        if name in POSITIVE_REQUIRES_C2_PEER:
            return 1 if peer_ip in c2_ips else None
        return 1
    if detailed_label in REFERENCE_ONLY_DETAILED_LABELS:
        return "reference"
    if detailed_label in EXCLUDE_DETAILED_LABELS:
        return None
    if detailed_label in ("-", "nan", "", "none"):
        return "benign_candidate"
    if "c&c" in detailed_label or "command-and-control" in detailed_label:
        return None
    # label列がbenignで、detailed-labelがプロトコル名そのものの場合は正常候補
    if zeek_label == "benign" and detailed_label in BENIGN_PROTOCOL_LABELS:
        return "benign_candidate"
    return "unknown"


def label_one(name):
    cap = CAPTURES[name]
    zeek_path = find_labeled(cap["dir"])
    bi_path = os.path.join(INTERMEDIATE_DIR, f"biflows_{name}.parquet")
    out_path = os.path.join(INTERMEDIATE_DIR, f"labeled_{name}.parquet")
    print(f"\n===== 段階3（ラベル照合）: {name} =====")
    print(f"  対象ログ: {zeek_path}")
    zeek = read_zeek_labeled(zeek_path)
    bi = pd.read_parquet(bi_path)
    host_ip = cap["host_ip"]
    if bi.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("[!] Biflowが0件のため、空のラベル付きファイルを出力しました")
        return

    bi["_key"] = make_key(bi["src_ip"], bi["src_port"], bi["dst_ip"], bi["dst_port"], bi["proto"])
    zeek["_key"] = make_key(
        zeek["id.orig_h"], zeek["id.orig_p"], zeek["id.resp_h"], zeek["id.resp_p"], zeek["proto"]
    )
    tol = pd.Timedelta(seconds=LABEL_MATCH_TOLERANCE_SEC)

    left = bi.copy()
    left["_t"] = pd.to_datetime(left["start_ts"], unit="s")
    left = left.sort_values("_t").reset_index(drop=True)
    right = zeek[["_key", "ts", "label", "detailed_label", "conn_state"]].copy()
    right["_t"] = pd.to_datetime(right["ts"], unit="s")
    right = right.sort_values("_t").reset_index(drop=True)

    out = pd.merge_asof(
        left, right[["_t", "_key", "label", "detailed_label", "conn_state"]],
        on="_t", by="_key", direction="nearest", tolerance=tol,
    )
    out = out.rename(columns={"label": "zeek_label", "detailed_label": "zeek_detailed"})
    out = out.drop(columns=["_t"])
    out["matched"] = out["zeek_label"].notna()
    print(f"  Biflow数={len(out):,}  Zeek照合成功={int(out['matched'].sum()):,}  "
          f"未照合={int((~out['matched']).sum()):,}")

    out["peer_ip"] = np.where(out["src_ip"] == host_ip, out["dst_ip"], out["src_ip"])
    out["peer_port"] = np.where(out["src_ip"] == host_ip, out["dst_port"], out["src_port"])
    c2_ips = set(cap["c2_server_ips"])

    n = len(out)
    label_final = np.full(n, np.nan)
    exclude_reason = np.array([None] * n, dtype=object)

    zeek_label_arr = out["zeek_label"].fillna("-").values
    zeek_det_arr = out["zeek_detailed"].fillna("-").values
    conn_state_arr = out["conn_state"].values
    proto_arr = out["proto"].values
    peer_ip_arr = out["peer_ip"].values
    matched_arr = out["matched"].values

    for i in range(n):
        if not matched_arr[i]:
            if not is_external(peer_ip_arr[i]):
                exclude_reason[i] = "not_north_south"
            else:
                exclude_reason[i] = "unmatched"
            continue

        det = zeek_det_arr[i]
        kind = classify(name, det, peer_ip_arr[i], c2_ips, zeek_label=zeek_label_arr[i])

        if kind == 1:
            label_final[i] = 1
            continue
        if kind == "reference":
            exclude_reason[i] = "reference_only"
            continue
        if kind == "unknown":
            exclude_reason[i] = f"unknown_label:{det}"
            continue
        if kind is None:
            exclude_reason[i] = f"excluded_label:{det}"
            continue

        if not is_external(peer_ip_arr[i]):
            exclude_reason[i] = "not_north_south"
        elif is_known_c2(peer_ip_arr[i]):
            exclude_reason[i] = "known_c2_peer"
        elif proto_arr[i] == "tcp" and conn_state_arr[i] in ("REJ", "S0"):
            exclude_reason[i] = "tcp_no_response"
        else:
            label_final[i] = 0

    out["label_final"] = label_final
    out["exclude_reason"] = exclude_reason

    unknown_mask = pd.Series(exclude_reason).astype(str).str.startswith("unknown_label")
    if unknown_mask.any():
        print("[警告] 未知の detailed-label が見つかりました。config.py の対応表を見直してください:")
        print(out.loc[unknown_mask.values, "zeek_detailed"].value_counts())

    out.to_parquet(out_path, index=False)
    print("  ---- 内訳 ----")
    print(f"  Label=1: {int((out['label_final'] == 1).sum())}")
    print(f"  Label=0: {int((out['label_final'] == 0).sum())}")
    for reason, cnt in pd.Series(exclude_reason).dropna().value_counts().items():
        print(f"  除外({reason}): {cnt}")
    print(f"[+] 完了: {out_path}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    targets = list(CAPTURES) if "--all" in args else [a for a in args if not a.startswith("--")]
    if not targets:
        print("使い方: python stage3_label.py <capture名> [<capture名> ...] | --all")
        sys.exit(1)
    for name in targets:
        label_one(name)