#!/bin/bash

# === Set Paths ===
AAPT_PATH="aapt"   # Or full path if not in PATH
ADB_PATH="adb"
APK_FOLDER="/Users/anjalipriya/Downloads/malicious_app_detectionUsing_traffic_analysis/malicious_androzoo_apks"
LOG_FOLDER="/Users/anjalipriya/Downloads/malicious_app_detectionUsing_traffic_analysis/malicious_apk_logcat_logs"

# === Parameters ===
MONKEY_EVENTS=300
WAIT_TIME=60

# === Create output folder ===
mkdir -p "$LOG_FOLDER"

# === Loop through APK files directly ===
for apk_path in "$APK_FOLDER"/*.apk; do
    apk_file=$(basename "$apk_path")
    app_name="${apk_file%.apk}"

    echo "----------------------------------------"
    echo " Processing APK: $apk_file"

    # === Install the APK ===
    echo " Installing APK: $apk_file"
    "$ADB_PATH" install -r "$apk_path" > /dev/null
    if [[ $? -ne 0 ]]; then
        echo " [!] Failed to install $apk_file. Skipping."
        continue
    fi

    # === Extract package name using aapt ===
    echo " Getting package name..."
    PACKAGE=$("$AAPT_PATH" dump badging "$apk_path" 2>/dev/null | grep -m 1 "package: name=" | sed -E "s/.*name='([^']+)'.*/\1/")

    if [[ -z "$PACKAGE" ]]; then
        echo " [!] Failed to detect package name for $apk_file. Skipping."
        "$ADB_PATH" uninstall "$PACKAGE" > /dev/null 2>&1
        continue
    fi

    echo " PACKAGE: $PACKAGE"

    # === Clear previous logs ===
    "$ADB_PATH" logcat -c

    # === Launch app using monkey ===
    echo " Launching app using monkey..."
    "$ADB_PATH" shell monkey -p "$PACKAGE" -v "$MONKEY_EVENTS" > /dev/null 2>&1

    echo " Waiting for $WAIT_TIME seconds..."
    sleep "$WAIT_TIME"

    # === Dump logcat ===
    LOGFILE="$LOG_FOLDER/${app_name}_logcat.txt"
    echo " Saving logcat to $LOGFILE"
    "$ADB_PATH" logcat -d > "$LOGFILE"

    # === Uninstall app ===
    echo " Uninstalling $PACKAGE"
    "$ADB_PATH" uninstall "$PACKAGE" > /dev/null

done

echo
echo "✓ Dynamic behavior logcat capture completed for all APKs."
