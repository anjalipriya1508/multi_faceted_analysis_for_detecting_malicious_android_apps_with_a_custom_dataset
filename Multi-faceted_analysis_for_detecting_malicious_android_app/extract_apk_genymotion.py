import subprocess
import os

# --- Configuration ---
package_file = 'genymotion_package.txt'  # Text file with package names (one per line)
output_directory = 'non_malicious_multiple_apk/demo_begnin_apk'  # Output base folder

# --- Ensure output directory exists ---
os.makedirs(output_directory, exist_ok=True)

# --- Enable adb root access ---
print("🔐 Enabling adb root (needed for full access)...")
subprocess.run(['adb', 'root'])

# --- Helper: Run adb command ---
def adb_command(cmd_list):
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()

# --- Read package names ---
with open(package_file, 'r') as f:
    packages = [line.strip() for line in f if line.strip()]

# --- Process each package ---
for package in packages:
    print(f"\n📦 Extracting APKs for: {package}")
    raw_output = adb_command(['adb', 'shell', 'pm', 'path', package])

    if not raw_output:
        print(f"❌ No APK paths found for {package}")
        continue

    apk_paths = [line.replace("package:", "").strip() for line in raw_output.splitlines() if line.startswith("package:")]
    
    if not apk_paths:
        print(f"❌ No valid APK paths for {package}")
        continue

    # Create folder for this app
    package_folder = os.path.join(output_directory, package.replace('.', '_'))
    os.makedirs(package_folder, exist_ok=True)

    success = False
    for idx, apk_path in enumerate(apk_paths):
        apk_filename = os.path.basename(apk_path)
        local_path = os.path.join(package_folder, apk_filename)

        print(f"📥 Pulling APK part {idx+1}: {apk_path}")
        result = subprocess.run(['adb', 'pull', apk_path, local_path], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Pulled: {apk_filename} → {package_folder}")
            success = True
        else:
            print(f"⚠️ Failed to pull: {apk_filename} — {result.stderr.strip()}")

    if not success:
        print(f"❌ Failed to extract any APKs for {package}")

print("\n🎯 Done: All APKs extracted and grouped in folders.")
