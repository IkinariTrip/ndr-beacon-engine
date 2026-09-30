import os
import scapy.all as scapy
from collections import defaultdict

class PacketInfo:
    def __init__(self, pkt):
        self.time = float(pkt.time)
        # ヘッダを除いた実データ長（ペイロードサイズ）を取得
        if pkt.haslayer(scapy.TCP):
            self.length = len(pkt[scapy.TCP].payload)
        elif pkt.haslayer(scapy.UDP):
            self.length = len(pkt[scapy.UDP].payload)
        else:
            self.length = 0
        self.is_ip = False
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

        if pkt.haslayer(scapy.IP):
            self.is_ip = True
            ip = pkt[scapy.IP]
            self.src_ip = ip.src
            self.dst_ip = ip.dst
            self.protocol = ip.proto

            if pkt.haslayer(scapy.TCP):
                self.is_tcp = True
                tcp = pkt[scapy.TCP]
                self.src_port = tcp.sport
                self.dst_port = tcp.dport
                self.flags = int(tcp.flags)
                self.window = tcp.window
                self.header_length = tcp.dataofs * 4
            elif pkt.haslayer(scapy.UDP):
                self.is_udp = True
                udp = pkt[scapy.UDP]
                self.src_port = udp.sport
                self.dst_port = udp.dport
                self.header_length = 8

class Flow:
    def __init__(self, first_pkt: PacketInfo, forward_key):
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

    def add_packet(self, pkt: PacketInfo, is_fwd: bool):
        self.last_time = pkt.time
        direction = 1 if is_fwd else -1
        self.packets.append((pkt.time, pkt.length, direction, pkt.flags, pkt.window, pkt.header_length))

def extract_flows_from_pcap(pcap_path: str, idle_timeout: float = 120.0):
    active_flows = {}
    finished_flows = []
    total_packets = 0

    reader = scapy.PcapReader(pcap_path)
    for raw_pkt in reader:
        total_packets += 1
        pkt = PacketInfo(raw_pkt)
        if not pkt.is_ip or not (pkt.is_tcp or pkt.is_udp):
            continue

        fwd_key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        bwd_key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)

        current_time = pkt.time
        keys_to_remove = []
        for k, flow in active_flows.items():
            if current_time - flow.last_time > idle_timeout:
                finished_flows.append(flow)
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del active_flows[k]

        if fwd_key in active_flows:
            flow = active_flows[fwd_key]
            flow.add_packet(pkt, is_fwd=True)
            if pkt.is_tcp and (pkt.flags & 0x01 or pkt.flags & 0x04):
                finished_flows.append(flow)
                del active_flows[fwd_key]
        elif bwd_key in active_flows:
            flow = active_flows[bwd_key]
            flow.add_packet(pkt, is_fwd=False)
            if pkt.is_tcp and (pkt.flags & 0x01 or pkt.flags & 0x04):
                finished_flows.append(flow)
                del active_flows[bwd_key]
        else:
            new_flow = Flow(pkt, forward_key=fwd_key)
            active_flows[fwd_key] = new_flow

    reader.close()
    finished_flows.extend(active_flows.values())
    return finished_flows, total_packets
