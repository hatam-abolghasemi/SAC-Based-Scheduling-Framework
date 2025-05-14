import socket
import yaml
import subprocess
import time
from datetime import datetime
import os

OUTPUT_TMP = '/tmp/prometheus.yaml'
OUTPUT_FINAL = '/etc/prometheus/prometheus.yml'

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(('localhost', port)) == 0

def find_open_ports_above_11000(max_port=65000):
    return sorted([port for port in range(11001, max_port + 1) if is_port_open(port)])

def generate_prometheus_config(open_ports, output_file=OUTPUT_TMP):
    config = {
        'global': {
            'scrape_interval': '15s'
        },
        'scrape_configs': [
            {
                'job_name': 'dynamic_11xxx',
                'static_configs': [
                    {
                        'targets': [f'localhost:{port}' for port in open_ports]
                    }
                ]
            },
            {
                'job_name': 'static_9904',
                'static_configs': [
                    {
                        'targets': ['localhost:9904']
                    }
                ]
            }
        ]
    }
    with open(output_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def move_config_to_final_location():
    try:
        subprocess.run(['sudo', 'mv', OUTPUT_TMP, OUTPUT_FINAL], check=True)
        print(f"📦 Moved config to {OUTPUT_FINAL}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to move config file: {e}")

def reload_prometheus():
    try:
        subprocess.run(['sudo', 'systemctl', 'reload', 'prometheus'], check=True)
        print("✅ Prometheus reloaded.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to reload Prometheus: {e}")

def main_loop():
    last_ports = []
    print("👀 Monitoring for changes in open ports (11XXX)...")
    while True:
        current_ports = find_open_ports_above_11000()
        if current_ports != last_ports:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n🔄 [{timestamp}] Change detected.")
            print(f"   ➕ New config: {current_ports}")
            generate_prometheus_config(current_ports)
            move_config_to_final_location()
            reload_prometheus()
            last_ports = current_ports
        time.sleep(1)

if __name__ == "__main__":
    main_loop()

