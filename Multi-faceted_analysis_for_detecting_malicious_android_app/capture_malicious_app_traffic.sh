#!/bin/bash

# === Set Paths ===
AAPT_PATH="aapt"  # Provide full path if needed
ADB_PATH="adb"
APK_PARENT_FOLDER="malicious_androzoo_apks"
PCAP_FOLDER="pcap/malicious"
TCPDUMP_PATH="/Users/anjalipriya/Downloads/tcpdump"
TCPDUMP_REMOTE="/data/local/tmp/tcpdump"
PYTHON_SCRIPT="get_package_name.py"

# === Create output folder if it doesn't exist ===
mkdir -p "$PCAP_FOLDER"

# === Debug print folder paths ===
echo "[DEBUG] APK Parent Folder: $APK_PARENT_FOLDER"
echo "[DEBUG] PCAP Output Folder: $PCAP_FOLDER"

# === Push tcpdump binary to device (once) ===
echo "[*] Pushing tcpdump to device..."
"$ADB_PATH" push "$TCPDUMP_PATH" "$TCPDUMP_REMOTE"
"$ADB_PATH" shell chmod 755 "$TCPDUMP_REMOTE"

# === Handle Ctrl+C gracefully ===
trap 'echo "Interrupted. Killing tcpdump and exiting."; "$ADB_PATH" shell pkill tcpdump; exit 1' INT TERM

# === Function: Get Package Name ===
get_package_name() {
    local apk_file="$1"
    PACKAGE=$(python3 "$PYTHON_SCRIPT" "$apk_file" "$AAPT_PATH")
    echo "$PACKAGE"
}

# === Process each .apk file in the folder ===
shopt -s nullglob

echo "[DEBUG] Searching for .apk files in: $APK_PARENT_FOLDER"
apk_files=("$APK_PARENT_FOLDER"/*.apk)

echo "[DEBUG] Found ${#apk_files[@]} APK(s)"
if [[ ${#apk_files[@]} -eq 0 ]]; then
    echo "❌ No APK files found in $APK_PARENT_FOLDER."
    echo "📂 Current working directory: $(pwd)"
    echo "📄 Files in APK_PARENT_FOLDER:"
    ls -l "$APK_PARENT_FOLDER"
    exit 1
fi

for apk_file in "${apk_files[@]}"; do
    app_name=$(basename "$apk_file" .apk)
    echo
    echo "========================================"
    echo " Processing APK: $app_name.apk"
    echo " [DEBUG] Full path: $apk_file"

    echo " [*] Starting tcpdump..."
    "$ADB_PATH" shell "nohup $TCPDUMP_REMOTE -i any -s 0 -w /data/local/tmp/${app_name}.pcap >/dev/null 2>&1 &"
    sleep 2

    echo " [*] Installing APK"
    "$ADB_PATH" install -r "$apk_file"

    echo " [*] Simulating app usage (30s)..."
    sleep 30

    echo " [*] Stopping tcpdump"
    "$ADB_PATH" shell pkill tcpdump

    echo " [*] Pulling pcap file..."
    "$ADB_PATH" pull "/data/local/tmp/${app_name}.pcap" "$PCAP_FOLDER/${app_name}.pcap"

    echo " [*] Detecting package name..."
    PACKAGE=$(get_package_name "$apk_file")
    echo " [*] Package Name: $PACKAGE"

    if [[ -n "$PACKAGE" ]]; then
        echo " [*] Uninstalling $PACKAGE"
        "$ADB_PATH" uninstall "$PACKAGE"
    else
        echo " [!] Failed to detect package name. Skipping uninstall."
    fi

    # Clean up pcap from device
    "$ADB_PATH" shell rm "/data/local/tmp/${app_name}.pcap"
done

echo
echo "✅ Done capturing traffic for all direct APKs in $APK_PARENT_FOLDER."
