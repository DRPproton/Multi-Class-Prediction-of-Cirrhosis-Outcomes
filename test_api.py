import requests

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

response = requests.post(url, json=patient).json()

print(response)
