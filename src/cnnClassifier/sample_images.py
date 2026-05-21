import os
import urllib.request

# 1. Create the directories safely
os.makedirs("static/samples", exist_ok=True)

# 2. Download a public sample image for "Normal"
print("Downloading sample 1...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-1.jpg", 
    "static/samples/normal.jpg"
)

# 3. Download a public sample image for "Anomaly"
print("Downloading sample 2...")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-2.jpg", 
    "static/samples/anomaly.jpg"
)

print("Done! Check your static/samples folder.")