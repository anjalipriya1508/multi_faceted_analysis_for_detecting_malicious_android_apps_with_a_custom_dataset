import subprocess

output_file = 'genymotion_package.txt'

# Run adb shell command to list installed packages
print("📦 Fetching installed packages from connected Android device/emulator...")
result = subprocess.run(['adb', 'shell', 'pm', 'list', 'packages','-3'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if result.returncode != 0:
    print("❌ Failed to list packages. Is your emulator running and connected via ADB?")
    print(result.stderr.decode('utf-8'))
else:
    raw_output = result.stdout.decode('utf-8')
    # Extract package names and write to file
    packages = [line.replace('package:', '').strip() for line in raw_output.splitlines()]

    with open(output_file, 'w') as f:
        for pkg in packages:
            f.write(pkg + '\n')

    print(f"✅ Saved {len(packages)} package names to {output_file}")
