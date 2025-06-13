import os
import re
import pandas as pd

def extract_logcat_features(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    clean_name = re.sub(r'_logcat$', '', base_name)  # Removes _logcat at end

    features = {
        'file_name': clean_name,
        'used_sendTextMessage': 0,
        'used_requestNetwork': 0,
        'has_fatal_exception': 0,
        'activity_started': 0,
        'read_contacts': 0,
        'write_file': 0,
        'access_location': 0,
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.lower()
                if 'sendtextmessage' in line:
                    features['used_sendTextMessage'] = 1
                if 'requestnetwork' in line or 'connectivityservice' in line:
                    features['used_requestNetwork'] = 1
                if 'fatal exception' in line:
                    features['has_fatal_exception'] = 1
                if 'start proc' in line:
                    features['activity_started'] = 1
                if 'contacts' in line:
                    features['read_contacts'] = 1
                if re.search(r'open|write|create file', line):
                    features['write_file'] = 1
                if 'location' in line:
                    features['access_location'] = 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return features



def process_logcat_folder(folder_path, output_csv):
    all_features = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):  # or use "logcat.txt" for strict filtering
                file_path = os.path.join(root, file)
                features = extract_logcat_features(file_path)
                all_features.append(features)
    
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"Feature CSV saved to: {output_csv}")

# Usage
folder_path = "begnin_logcat_folder"         # Change this to your logcat files folder
output_csv = "androZoo_dataset_analysis/begnin_dynamic_feature_extraction.csv"   # Desired output file
process_logcat_folder(folder_path, output_csv)
