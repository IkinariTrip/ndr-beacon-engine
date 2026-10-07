"""
src/flow_generator.py（v5.4）
【v5.4 変更点】
1. 層構造の厳密チェック
   旧実装は、IPアドレスを最初のIP層から、ポートをパケット内のどこかにある
   TCP/UDP層から、別々に取っていた。GREなどでIPが入れ子になったパケットや、
   ICMPエラーに埋め込まれたヘッダーでは、IPとポートが別々の層から取られ、
   組み合わせが崩れるおそれがある。
   新実装では「最も外側のIP層の直下がTCPまたはUDPのパケット」だけを使い、
   それ以外（GRE・ICMP・IPの断片・入れ子のIPなど）は理由ごとに数えて除外する。
   エンジンB（tshark）と同じく、外側のIPv4のTCP/UDPだけを見る方針にそろえる。
2. DNSの解析を無効化
   フロー生成にDNSの中身は使わない。壊れたDNSを解析しようとして出る
   「DNS decompression loop detected」の警告を止め、処理も軽くする。
3. タイムアウト処理の高速化（結果は変えない）
   旧実装は、パケットを1つ読むたびに進行中の全フローを調べていた。
   最終更新が古い順に並べた辞書（OrderedDict）を使い、
   先頭から期限切れの分だけ取り出す方式に変えた。
戻り値は旧実装と同じ (flows, total_packets)。除外の内訳は LAST_STATS で確認できる。
"""
import operator
from collections import OrderedDict
import scapy.all as scapy
from scapy.packet import split_layers

# ---- DNS（mDNS・LLMNRを含む）の解析を無効化する ----
DNS_PORT_RULES = (
    {"dport": 53}, {"sport": 53},
    {"dport": 5353}, {"sport": 5353},
    {"dport": 5355}, {"sport": 5355},
)
for rule in DNS_PORT_RULES:
    for l4_layer in (scapy.UDP, scapy.TCP):
        try:
            split_layers(l4_layer, scapy.DNS, **rule)
        except Exception:
            pass

# 直前に実行した extract_flows_from_pcap の除外内訳（確認用）
LAST_STATS = {}

class PacketInfo:
    """1パケットの情報。valid が False のパケットはフロー生成に使わない。"""
    def __init__(self, pkt):
        self.time = float(pkt.time)
        self.length = 0
        self.valid = False
        self.reason = None
        self.is_tcp = False
        self.is_udp = False
        self.src_ip = None
        self.dst_ip = None
        self.src_port = None
        self.dst_port = None
        self.protocol = None
        self.flags = 0
        self.window = 0
        self.header_length = 0

        # 最も外側のIPv4層（getlayerは先頭から探すので、最初に見つかるのが外側）
        ip = pkt.getlayer(scapy.IP)

        # ICMPエラーに埋め込まれたヘッダー（IPerror）はIPの派生型なので、型を厳密に確認する
        if ip is None or type(ip) is not scapy.IP:
            self.reason = "not_ipv4"
            return

        # IPの断片（2つ目以降、または続きがあるもの）は除外する
        if int(ip.frag) != 0 or (int(ip.flags) & 0x1):
            self.reason = "ip_fragment"
            return

        # 外側IPの直下の層だけを見る
        l4 = ip.payload
        if type(l4) is scapy.TCP:
            self.is_tcp = True
            self.flags = int(l4.flags)
            self.window = int(l4.window)
            self.header_length = int(l4.dataofs) * 4
        elif type(l4) is scapy.UDP:
            self.is_udp = True
            self.header_length = 8
        else:
            # GRE・ICMP・入れ子のIPなど。どの層だったかを理由として残す
            self.reason = "outer_ip_payload_" + type(l4).__name__
            return

        self.src_ip = ip.src
        self.dst_ip = ip.dst
        self.protocol = int(ip.proto)
        self.src_port = int(l4.sport)
        self.dst_port = int(l4.dport)

        # ヘッダを除いた実データ長（旧実装と同じ定義：TCP/UDPのペイロード長）
        self.length = len(l4.payload)
        self.valid = True

class Flow:
    def __init__(self, first_pkt, forward_key):
        self.forward_key = forward_key
        self.src_ip = first_pkt.src_ip
        self.dst_ip = first_pkt.dst_ip
        self.src_port = first_pkt.src_port
        self.dst_port = first_pkt.dst_port
        self.protocol = first_pkt.protocol
        self.start_time = first_pkt.time
        self.last_time = first_pkt.time
        self.packets = []
        self.add_packet(first_pkt, is_fwd=True)

    def add_packet(self, pkt, is_fwd):
        self.last_time = pkt.time
        direction = 1 if is_fwd else -1
        self.packets.append(
            (pkt.time, pkt.length, direction, pkt.flags, pkt.window, pkt.header_length)
        )

def _is_expired(current_time, last_time, idle_timeout):
    """経過時間がタイムアウトを超えていれば True（比較記号の文字化けを避けるため operator.gt を使用）。"""
    return operator.gt(current_time - last_time, idle_timeout)

def extract_flows_from_pcap(pcap_path, idle_timeout=120.0):
    # キーからフローへの対応表。最終更新が古い順に並ぶよう、更新のたびに末尾へ移す
    active_flows = OrderedDict()
    finished_flows = []
    total_packets = 0
    skipped = {}

    reader = scapy.PcapReader(pcap_path)
    try:
        for raw_pkt in reader:
            total_packets += 1
            pkt = PacketInfo(raw_pkt)
            if not pkt.valid:
                skipped[pkt.reason] = skipped.get(pkt.reason, 0) + 1
                continue

            current_time = pkt.time

            # 期限切れのフローを、古い順に先頭から取り出す（旧実装の全件確認と同じ結果）
            while active_flows:
                oldest_key = next(iter(active_flows))
                oldest_flow = active_flows[oldest_key]
                if _is_expired(current_time, oldest_flow.last_time, idle_timeout):
                    finished_flows.append(oldest_flow)
                    del active_flows[oldest_key]
                else:
                    break

            fwd_key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
            bwd_key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)
            fin_or_rst = pkt.is_tcp and bool(pkt.flags & 0x01 or pkt.flags & 0x04)

            if fwd_key in active_flows:
                flow = active_flows[fwd_key]
                flow.add_packet(pkt, is_fwd=True)
                if fin_or_rst:
                    finished_flows.append(flow)
                    del active_flows[fwd_key]
                else:
                    active_flows.move_to_end(fwd_key)
            elif bwd_key in active_flows:
                flow = active_flows[bwd_key]
                flow.add_packet(pkt, is_fwd=False)
                if fin_or_rst:
                    finished_flows.append(flow)
                    del active_flows[bwd_key]
                else:
                    active_flows.move_to_end(bwd_key)
            else:
                active_flows[fwd_key] = Flow(pkt, forward_key=fwd_key)
    finally:
        reader.close()

    finished_flows.extend(active_flows.values())

    used_packets = total_packets - sum(skipped.values())
    skipped_sorted = dict(sorted(skipped.items(), key=lambda item: -item[1]))

    LAST_STATS.clear()
    LAST_STATS.update(
        total_packets=total_packets,
        used_packets=used_packets,
        skipped=skipped_sorted,
        flows=len(finished_flows),
    )
    return finished_flows, total_packets
