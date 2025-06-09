#!/bin/bash

# === Set Paths ===
AAPT_PATH="aapt"  # Provide full path if needed
ADB_PATH="adb"
APK_PARENT_FOLDER="non_malicious_multiple_apk/demo_begnin_apk"
PCAP_FOLDER="pcap/begnin"
TCPDUMP_PATH="/Users/anjalipriya/Downloads/tcpdump"
TCPDUMP_REMOTE="/data/local/tmp/tcpdump"

# === Create output folder if it doesn't exist ===
mkdir -p "$PCAP_FOLDER"

# === Push tcpdump binary to device (once) ===
echo "[*] Pushing tcpdump to device..."
"$ADB_PATH" push "$TCPDUMP_PATH" "$TCPDUMP_REMOTE"
"$ADB_PATH" shell chmod 755 "$TCPDUMP_REMOTE"

# === Handle Ctrl+C gracefully ===
trap 'echo "Interrupted. Killing tcpdump and exiting."; "$ADB_PATH" shell pkill tcpdump; exit 1' INT TERM

# === Loop through all subfolders ===
for app_folder in "$APK_PARENT_FOLDER"/*/; do
    app_name=$(basename "$app_folder")
    echo
    echo "========================================"
    echo " Processing App Folder: $app_name"

    # === Build install-multiple command ===
    apk_files=("$app_folder"*.apk)
    if [[ ${#apk_files[@]} -eq 0 ]]; then
        echo " [!] No APKs found in $app_folder. Skipping..."
        continue
    fi

    echo " [*] Starting tcpdump..."
    "$ADB_PATH" shell "nohup $TCPDUMP_REMOTE -i any -s 0 -w /data/local/tmp/${app_name}.pcap >/dev/null 2>&1 &"
    sleep 2

    echo " [*] Installing APKs from $app_folder"
    "$ADB_PATH" install-multiple -r "${apk_files[@]}"

    echo " [*] Simulating app usage (60s)..."
    sleep 60

    echo " [*] Stopping tcpdump"
    "$ADB_PATH" shell pkill tcpdump

    echo " [*] Pulling pcap file..."
    "$ADB_PATH" pull "/data/local/tmp/${app_name}.pcap" "$PCAP_FOLDER/${app_name}.pcap"

    # === Detect package name using Python helper ===
    echo " [*] Detecting package name from base APK..."
    base_apk="${apk_files[0]}"
    PACKAGE=$(python3 "/Users/anjalipriya/Downloads/malicious_app_detectionUsing_traffic_analysis/get_package_name.py" "$base_apk" "$AAPT_PATH")
    
    echo " [*] Package Name: $PACKAGE"

    if [[ -n "$PACKAGE" ]]; then
        echo " [*] Uninstalling $PACKAGE"
        "$ADB_PATH" uninstall "$PACKAGE"
    else
        echo " [!] Failed to detect package name. Skipping uninstall."
    fi

    # === Clean up remote pcap ===
    "$ADB_PATH" shell rm "/data/local/tmp/${app_name}.pcap"
done

echo
echo "✅ Done capturing traffic for all APKs."
