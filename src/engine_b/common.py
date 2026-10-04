"""common.py — 各段階で共通して使うユーティリティ関数"""
import glob
import os
import ipaddress

from config import BASE_DIR, KNOWN_C2_NETWORKS


def _find_all(capture_dir, pattern):
    return glob.glob(os.path.join(BASE_DIR, capture_dir, "**", pattern), recursive=True)


def _pick_one(paths, pattern, capture_dir):
    if len(paths) == 0:
        raise FileNotFoundError(f"{pattern} が見つかりません: {capture_dir}")
    if len(paths) > 1:
        paths = sorted(paths, key=os.path.getsize, reverse=True)
        print(f"[警告] {pattern} が複数見つかりました。最大サイズのファイルを使用します: {paths[0]}")
    return paths[0]


def find_pcap(capture_dir):
    return _pick_one(_find_all(capture_dir, "*.pcap"), "*.pcap", capture_dir)


def find_labeled(capture_dir):
    for pat in ("*zeek-conn-log.labeled", "*zeek-conn.log.labeled", "*.labeled"):
        paths = _find_all(capture_dir, pat)
        if paths:
            return _pick_one(paths, pat, capture_dir)
    raise FileNotFoundError(f".labeled ファイルが見つかりません: {capture_dir}")


_KNOWN_NETS = [ipaddress.ip_network(n) for n in KNOWN_C2_NETWORKS]


def is_external(ip_str):
    try:
        a = ipaddress.ip_address(str(ip_str))
    except ValueError:
        return False
    if a.is_private or a.is_multicast or a.is_link_local or a.is_loopback or a.is_unspecified:
        return False
    if str(a) == "255.255.255.255":
        return False
    return True


def is_known_c2(ip_str):
    try:
        a = ipaddress.ip_address(str(ip_str))
    except ValueError:
        return False
    return any(a in net for net in _KNOWN_NETS)


def canon_key(ip_a, port_a, ip_b, port_b, proto):
    ep1, ep2 = (ip_a, port_a), (ip_b, port_b)
    if ep2 < ep1:
        ep1, ep2 = ep2, ep1
    return (proto, ep1, ep2)
