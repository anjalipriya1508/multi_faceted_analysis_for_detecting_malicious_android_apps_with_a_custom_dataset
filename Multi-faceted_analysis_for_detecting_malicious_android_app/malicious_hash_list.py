import csv
import os
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from datetime import datetime

API_KEY = '03210732c0a6c1d6863fa96e03ee35affbced6d75d8a1c3832f1a30bcfc47041'
CSV_FILE = os.path.expanduser('~/Downloads/latest.csv')
DOWNLOAD_DIR = 'malicious_arm64'
MAX_VALID_APKS = 500
MAX_THREADS = 3
LOG_FILE = 'final_malicious_arm64_apk_download_log.txt'
VALID_HASHES_FILE = 'final_valid_arm64_hashes.txt'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

valid_apks = []
lock = Lock()
log_lock = Lock()

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}\n"
    print(full_msg.strip())
    with log_lock:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(full_msg)

def save_valid_hashes():
    with lock:
        with open(VALID_HASHES_FILE, 'w', encoding='utf-8') as f:
            for sha in valid_apks:
                f.write(sha + '\n')

def load_valid_hashes():
    if os.path.exists(VALID_HASHES_FILE):
        with open(VALID_HASHES_FILE, 'r', encoding='utf-8') as f:
            hashes = [line.strip() for line in f.readlines()]
        return hashes
    return []

def download_and_check(row, line_number, pbar):
    sha256 = row['sha256']
    vt_detection = row.get('vt_detection', '0')

    if int(vt_detection) < 1:
        pbar.update(1)
        return False

    apk_path = os.path.join(DOWNLOAD_DIR, f'{sha256}.apk')
    url = f'https://androzoo.uni.lu/api/download?apikey={API_KEY}&sha256={sha256}'

    try:
        log(f"[{line_number}] ⬇️ Downloading: {sha256}")
        response = requests.get(url, stream=True, timeout=120)

        if response.status_code == 200:
            with open(apk_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            log(f"[{line_number}] ✅ Download complete: {sha256}")

            result = subprocess.run(
                ['aapt2', 'dump', 'badging', apk_path],
                capture_output=True, text=True
            )
            if "'arm64-v8a'" in result.stdout:
                log(f"[{line_number}] ✅ arm64-v8a supported: {sha256}")
                with lock:
                    if sha256 not in valid_apks:
                        valid_apks.append(sha256)
                        save_valid_hashes()
                return True
            else:
                log(f"[{line_number}] ❌ Not arm64-v8a compatible: {sha256}")
                os.remove(apk_path)
        else:
            log(f"[{line_number}] ❌ HTTP {response.status_code} for {sha256}")
    except Exception as e:
        log(f"[{line_number}] ⚠️ Error: {sha256} - {e}")
    finally:
        pbar.update(1)
    return False

def main():
    global valid_apks
    valid_apks = load_valid_hashes()

    total_attempts = 0
    max_attempts = 100

    start_line = 20400  # Hardcoded start row in CSV

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Skip lines before start_line
        for _ in range(start_line - 2):
            next(reader)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            pbar = tqdm(total=max_attempts, desc="Processing Malicious ARM64 APKs", ncols=100)

            for line_number, row in enumerate(reader, start=start_line):
                if len(valid_apks) >= MAX_VALID_APKS or total_attempts >= max_attempts:
                    break
                if 'sha256' in row and 'vt_detection' in row:
                    futures.append(executor.submit(download_and_check, row, line_number, pbar))
                    total_attempts += 1

            for future in as_completed(futures):
                if len(valid_apks) >= MAX_VALID_APKS:
                    break

            pbar.close()

    log(f"\n✅ Total malicious APKs with arm64-v8a support downloaded: {len(valid_apks)}")
    for sha in valid_apks:
        log(f"  - {sha}")

if __name__ == '__main__':
    main()
