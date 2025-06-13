# About the Dataset
This script used to develops a customized and comprehensive Android malware dataset that integrates three layers of features:

**Static Features:** Extracted directly from APK files using tools like Apktool, androguard, etc.

**Dynamic Features:** Collected by executing each app in a Genymotion emulator with 60 seconds of UI simulation while capturing logcat logs.

**Network Features:** Captured as PCAP (packet capture) files during app runtime using tools like tcpdump.
The dataset includes:

**Malicious samples:** Collected from AndroZoo, with malware confirmed using VirusTotal detections.

**Benign apps:** Extracted from third-party apps installed on the emulator, ensuring ARM64-v8a compatibility and feature completeness.

This hybrid feature dataset supports advanced malware detection research using both traditional machine learning and few-shot learning models.

# Need Help?
If you have any questions or run into any issues while using the scripts, feel free to open an issue on this repository or drop a message — I'm happy to help!

Alternatively, you can reach out via **[anjali1508priya@gmail.com]** or via **https://www.linkedin.com/in/anjali-priya-622241196**
