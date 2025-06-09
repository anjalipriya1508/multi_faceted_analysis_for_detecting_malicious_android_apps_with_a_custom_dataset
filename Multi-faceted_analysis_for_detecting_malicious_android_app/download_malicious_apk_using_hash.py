import os
import requests

API_KEY = '03210732c0a6c1d6863fa96e03ee35affbced6d75d8a1c3832f1a30bcfc47041' # Replace with your real key
HASH_FILE = 'hashes.txt'
DOWNLOAD_DIR = 'malicious_androzoo_apks'

def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def download_apks():
    ensure_folder(DOWNLOAD_DIR)

    with open(HASH_FILE, 'r') as f:
        hashes = [line.strip() for line in f if line.strip()]

    for sha256 in hashes:
        print(f"[+] Downloading {sha256}...")
        url = f"https://androzoo.uni.lu/api/download?apikey={API_KEY}&sha256={sha256}"
        response = requests.get(url)
        
        if response.status_code == 200:
            apk_path = os.path.join(DOWNLOAD_DIR, f"{sha256}.apk")
            with open(apk_path, 'wb') as apk_file:
                apk_file.write(response.content)
            print(f"[✔] Saved: {apk_path}")
        else:
            print(f"[✘] Failed: {sha256} - HTTP {response.status_code}")

if __name__ == "__main__":
    download_apks()
