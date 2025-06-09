import os
import pandas as pd
from collections import defaultdict
from androguard.misc import AnalyzeAPK

# Folder containing APK files directly
APK_FOLDER = "malicious_androzoo_apks"

# Global sets of unique features
all_permissions = set()
all_activities = set()
all_services = set()
all_receivers = set()
all_providers = set()

# Dictionary: apk file name → combined features
app_features = defaultdict(lambda: {
    "permissions": set(),
    "activities": set(),
    "services": set(),
    "receivers": set(),
    "providers": set()
})

# === Traverse APK files directly ===
for apk_file in os.listdir(APK_FOLDER):
    if not apk_file.endswith('.apk'):
        continue

    apk_path = os.path.join(APK_FOLDER, apk_file)
    app_name = os.path.splitext(apk_file)[0]  # Strip .apk extension

    try:
        a, d, dx = AnalyzeAPK(apk_path)

        # Extract features
        perms = set(a.get_permissions())
        acts = set(a.get_activities())
        srvs = set(a.get_services())
        rcvs = set(a.get_receivers())
        prvs = set(a.get_providers())

        # Store in app_features
        app_features[app_name]["permissions"].update(perms)
        app_features[app_name]["activities"].update(acts)
        app_features[app_name]["services"].update(srvs)
        app_features[app_name]["receivers"].update(rcvs)
        app_features[app_name]["providers"].update(prvs)

        # Update global sets
        all_permissions.update(perms)
        all_activities.update(acts)
        all_services.update(srvs)
        all_receivers.update(rcvs)
        all_providers.update(prvs)

    except Exception as e:
        print(f"❌ Failed to analyze {apk_file}: {e}")

# === Prepare column names
permission_list = sorted(all_permissions)
activity_list = sorted(all_activities)
service_list = sorted(all_services)
receiver_list = sorted(all_receivers)
provider_list = sorted(all_providers)

columns = (
    ["app_name"] +
    ["perm_" + p for p in permission_list] +
    ["act_" + a for a in activity_list] +
    ["srv_" + s for s in service_list] +
    ["rcv_" + r for r in receiver_list] +
    ["prv_" + p for p in provider_list]
)

# === Create feature matrix
data = []
for app_name, feats in app_features.items():
    row = [app_name]
    row += [1 if p in feats["permissions"] else 0 for p in permission_list]
    row += [1 if a in feats["activities"] else 0 for a in activity_list]
    row += [1 if s in feats["services"] else 0 for s in service_list]
    row += [1 if r in feats["receivers"] else 0 for r in receiver_list]
    row += [1 if p in feats["providers"] else 0 for p in provider_list]
    data.append(row)

# === Export to CSV
df = pd.DataFrame(data, columns=columns)
output_dir = "androZoo_dataset_analysis"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "malicious_static_feature_extraction.csv")
df.to_csv(output_path, index=False)

print(f"\n✅ App-level features saved to {output_path}")
