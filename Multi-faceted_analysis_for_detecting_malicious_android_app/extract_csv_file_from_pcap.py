import os
import pyshark
import csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

UNCOMMON_PORTS = {1337, 4444, 9001}

def extract_features_from_pcap(pcap_file):
    try:
        cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=False)
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
        return None

    total_packets = 0
    total_bytes = 0
    protocols = Counter()
    dest_ports = set()
    dest_ips = set()
    domain_resolved = False
    dns_query_count = 0
    user_agents = set()
    packet_times = []
    packet_sizes = []
    src_dst_pairs = set()
    flow_count = defaultdict(int)

    for packet in cap:
        try:
            total_packets += 1
            packet_length = int(packet.length)
            total_bytes += packet_length
            packet_sizes.append(packet_length)
            packet_times.append(float(packet.sniff_timestamp))

            protocols[packet.highest_layer] += 1

            if 'ip' in packet:
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                dest_ips.add(dst_ip)
                flow_key = (src_ip, dst_ip)
                src_dst_pairs.add(flow_key)
                flow_count[flow_key] += 1

            if hasattr(packet, 'tcp'):
                dest_ports.add(int(packet.tcp.dstport))
            elif hasattr(packet, 'udp'):
                dest_ports.add(int(packet.udp.dstport))

            if hasattr(packet, 'dns') and hasattr(packet.dns, 'qry_name'):
                dns_query_count += 1
                domain_resolved = True

            if hasattr(packet, 'http') and hasattr(packet.http, 'user_agent'):
                user_agents.add(str(packet.http.user_agent))

        except Exception:
            continue

    cap.close()

    frequent_beaconing = False
    if len(packet_times) >= 2:
        packet_times.sort()
        intervals = [t2 - t1 for t1, t2 in zip(packet_times, packet_times[1:])]
        short_beacons = sum(1 for x in intervals if 5 <= x <= 10)
        frequent_beaconing = short_beacons >= 3

    suspicious_ua = any(len(ua.strip()) == 0 or 'python' in ua.lower() for ua in user_agents)

    avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0
    max_packet_size = np.max(packet_sizes) if packet_sizes else 0
    min_packet_size = np.min(packet_sizes) if packet_sizes else 0
    std_packet_size = np.std(packet_sizes) if packet_sizes else 0
    skew_packet_size = (np.mean(packet_sizes) - np.median(packet_sizes)) / (std_packet_size + 1e-5) if packet_sizes else 0

    unique_flows = len(src_dst_pairs)
    flow_durations = np.diff(packet_times)
    avg_flow_duration = np.mean(flow_durations) if len(flow_durations) > 0 else 0

    flow_values = np.array(list(flow_count.values()))
    prob = flow_values / np.sum(flow_values) if len(flow_values) > 0 else [1]
    flow_entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0

    inter_arrival_times = np.diff(packet_times)
    avg_inter_arrival = np.mean(inter_arrival_times) if len(inter_arrival_times) > 0 else 0
    max_inter_arrival = np.max(inter_arrival_times) if len(inter_arrival_times) > 0 else 0
    min_inter_arrival = np.min(inter_arrival_times) if len(inter_arrival_times) > 0 else 0

    return {
        'app_name': os.path.splitext(os.path.basename(pcap_file))[0],
        'total_packets': total_packets,
        'total_bytes': total_bytes,
        'unique_dest_ips': len(dest_ips),
        'dns_queries': dns_query_count,
        'uncommon_ports_used': int(any(p in UNCOMMON_PORTS for p in dest_ports)),
        'many_unique_ips': int(len(dest_ips) > 20),
        'high_dns_queries': int(dns_query_count > 50),
        'ip_only_traffic': int(not domain_resolved),
        'frequent_beaconing_detected': int(frequent_beaconing),
        'suspicious_user_agent': int(suspicious_ua),
        'avg_packet_size': avg_packet_size,
        'max_packet_size': max_packet_size,
        'min_packet_size': min_packet_size,
        'std_packet_size': std_packet_size,
        'skew_packet_size': skew_packet_size,
        'unique_flows': unique_flows,
        'avg_flow_duration': avg_flow_duration,
        'flow_entropy': flow_entropy,
        'avg_inter_arrival_time': avg_inter_arrival,
        'max_inter_arrival_time': max_inter_arrival,
        'min_inter_arrival_time': min_inter_arrival,
    }

def process_and_normalize(pcap_root_dir, output_csv):
    all_features = []
    for root, dirs, files in os.walk(pcap_root_dir):
        if not files:
            continue
        label = 1 if 'malicious' in root.lower() else 0
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_path = os.path.join(root, file)
                print(f"📦 Processing: {pcap_path}")
                features = extract_features_from_pcap(pcap_path)
                if features:
                    features['label'] = label
                    all_features.append(features)

    if not all_features:
        print("❌ No features extracted. Check your .pcap files and folder structure.")
        return

    df = pd.DataFrame(all_features)

    exclude_columns = [
        'app_name', 'label',
        'uncommon_ports_used', 'many_unique_ips',
        'high_dns_queries', 'ip_only_traffic',
        'frequent_beaconing_detected', 'suspicious_user_agent'
    ]
    numeric_cols = [col for col in df.columns if col not in exclude_columns]

    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: np.log1p(x))

    if 'label' in df.columns:
        label_col = df.pop('label')
        df['label'] = label_col
    else:
        print("⚠️ 'label' column missing. Skipping label move.")

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Final dataset saved to: {output_csv}")

if __name__ == "__main__":
    process_and_normalize('pcap', 'androZoo_dataset_analysis/combined_pcap_feature_extraction.csv')
