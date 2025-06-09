import os
import pandas as pd
from collections import defaultdict
from androguard.misc import AnalyzeAPK

# Path to parent folder with subfolders for each app
APK_ROOT_FOLDER = "non_malicious_multiple_apk/demo_begnin_apk"

# Global sets of unique features
all_permissions = set()
all_activities = set()
all_services = set()
all_receivers = set()
all_providers = set()

# Dictionary: app folder name → combined features
app_features = defaultdict(lambda: {
    "permissions": set(),
    "activities": set(),
    "services": set(),
    "receivers": set(),
    "providers": set()
})

# === Traverse app folders ===
for app_folder in os.listdir(APK_ROOT_FOLDER):
    full_app_path = os.path.join(APK_ROOT_FOLDER, app_folder)
    if not os.path.isdir(full_app_path):
        continue  # Skip files

    for apk_file in os.listdir(full_app_path):
        if not apk_file.endswith('.apk'):
            continue

        apk_path = os.path.join(full_app_path, apk_file)
        try:
            a, d, dx = AnalyzeAPK(apk_path)

            # Extract features
            perms = set(a.get_permissions())
            acts = set(a.get_activities())
            srvs = set(a.get_services())
            rcvs = set(a.get_receivers())
            prvs = set(a.get_providers())

            # Merge into app feature set
            app_features[app_folder]["permissions"].update(perms)
            app_features[app_folder]["activities"].update(acts)
            app_features[app_folder]["services"].update(srvs)
            app_features[app_folder]["receivers"].update(rcvs)
            app_features[app_folder]["providers"].update(prvs)

            # Add to global lists
            all_permissions.update(perms)
            all_activities.update(acts)
            all_services.update(srvs)
            all_receivers.update(rcvs)
            all_providers.update(prvs)

        except Exception as e:
            print(f"❌ Failed to analyze {apk_file} in {app_folder}: {e}")

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
df.to_csv("androZoo_dataset_analysis/begnin_static_feature_extraction.csv", index=False)

print("\n✅ App-level features saved to begnin_static_feature_extraction.csv")
