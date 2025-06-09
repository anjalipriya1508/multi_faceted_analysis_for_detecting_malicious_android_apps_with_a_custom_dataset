import subprocess
import os

# Path to your adb executable
ADB_PATH = "adb"

# Path to the package list file
PACKAGE_LIST_FILE = 'genymotion_package.txt'

# Check if file exists
if not os.path.isfile(PACKAGE_LIST_FILE):
    print(f"❌ Package list file not found: {PACKAGE_LIST_FILE}")
    exit(1)

# Read package names
with open(PACKAGE_LIST_FILE, "r", encoding="utf-8") as f:
    packages = [line.strip() for line in f if line.strip()]

print("📦 Starting to uninstall packages...\n")

for package in packages:
    print(f"🔄 Uninstalling: {package}")
    try:
        result = subprocess.run(
            [ADB_PATH, "uninstall", package],
            capture_output=True, text=True
        )
        if "Success" in result.stdout:
            print(f"✅ Successfully uninstalled {package}")
        else:
            print(f"⚠️ Failed to uninstall {package}: {result.stdout.strip()} {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ Error uninstalling {package}: {e}")

print("\n✅ Done.")
