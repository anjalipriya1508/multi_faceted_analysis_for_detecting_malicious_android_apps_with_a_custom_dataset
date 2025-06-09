#!/bin/bash

# === Set Paths ===
AAPT_PATH="aapt"  # Provide full path if needed
ADB_PATH="adb"
APK_PARENT_FOLDER="non_malicious_multiple_apk/demo_begnin_apk"
LOG_FOLDER="begnin_logcat_folder"

# === Create output folder if it doesn't exist ===
mkdir -p "$LOG_FOLDER"

# === Handle Ctrl+C gracefully ===
trap 'echo "Interrupted. Clearing logcat and exiting."; "$ADB_PATH" shell logcat -c; exit 1' INT TERM

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

    echo " [*] Clearing previous logcat..."
    "$ADB_PATH" shell logcat -c

    echo " [*] Installing APKs from $app_folder"
    "$ADB_PATH" install-multiple -r "${apk_files[@]}"

    echo " [*] Capturing logcat in background..."
    "$ADB_PATH" logcat -v time > "$LOG_FOLDER/${app_name}.log" &
    LOGCAT_PID=$!

    echo " [*] Simulating app usage (60s)..."
    sleep 60

    echo " [*] Stopping logcat capture..."
    kill "$LOGCAT_PID"
    wait "$LOGCAT_PID" 2>/dev/null

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
done

echo
echo "✅ Done capturing logcat logs for all APKs."
