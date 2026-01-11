import requests
import time
import subprocess
import sys

# Start the FastAPI app in a separate process
process = subprocess.Popen([sys.executable, "fastapi_app.py"])
print("Starting FastAPI app...")
time.sleep(5) # Wait for the app to start

patient = {
 'N_Days': 3839,
 'Drug': 'D-penicillamine',
 'Age': 29724,
 'Sex': 'F',
 'Ascites': 'N',
 'Hepatomegaly': 'Y',
 'Spiders': 'N',
 'Edema': 'N',
 'Bilirubin': 2.2,
 'Cholesterol': 546.0,
 'Albumin': 3.37,
 'Copper': 65.0,
 'Alk_Phos': 1636.0,
 'SGOT': 151.9,
 'Tryglicerides': 90.0,
 'Platelets': 430.0,
 'Prothrombin': 10.6,
 'Stage': 3.0}

url = "http://localhost:9696/predict"

try:
    response = requests.post(url, json=patient)
    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print(f"Failed with status code {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
finally:
    # Kill the process
    process.terminate()
    print("FastAPI app terminated.")
