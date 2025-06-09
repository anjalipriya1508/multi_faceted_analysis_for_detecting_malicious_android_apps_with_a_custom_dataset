import subprocess
import re
import sys
import os

def get_package_name(apk_path, aapt_path):
    if not os.path.isfile(apk_path):
        return None

    try:
        output = subprocess.check_output(
            [aapt_path, 'dump', 'badging', apk_path],
            stderr=subprocess.STDOUT,
            encoding='utf-8'
        )
        match = re.search(r"package: name='(\S+)'", output)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)  # Wrong usage
    apk_path = sys.argv[1]
    aapt_path = sys.argv[2]
    pkg = get_package_name(apk_path, aapt_path)
    if pkg:
        print(pkg)  # Output ONLY the package name
