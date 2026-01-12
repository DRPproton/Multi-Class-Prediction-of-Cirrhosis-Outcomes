from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from utils import *
import uvicorn

# Loading the models
with open('xgb_model.bin', 'rb') as f_in:
    xgb_model = pickle.load(f_in)

with open('dv_model.bin', 'rb') as f_in:
    dv_model = pickle.load(f_in)

with open('scaler.bin', 'rb') as f_in:
    scaler_model = pickle.load(f_in)

app = FastAPI()

class Patient(BaseModel):
    N_Days: int
    Drug: str
    Age: int
    Sex: str
    Ascites: str
    Hepatomegaly: str
    Spiders: str
    Edema: str
    Bilirubin: float
    Cholesterol: float
    Albumin: float
    Copper: float
    Alk_Phos: float
    SGOT: float
    Tryglicerides: float
    Platelets: float
    Prothrombin: float
    Stage: float

@app.post('/predict')
def predict(patient: Patient):
    client = patient.model_dump()

    X = pd.DataFrame([client])

    df_test = process_day_columns(X)
    df_test.Stage = df_test.Stage.astype('category')
    df_test = feature_eng_num_to_cat(df_test)

    df_test_dict = df_test.to_dict(orient='records')

    df_test = dv_model.transform(df_test_dict)
    df_test = scaler_model.transform(df_test)


    prediction = xgb_model.predict(df_test)
    proba = xgb_model.predict_proba(df_test)

    pred = None

    #  0 = D (death), 1 = C (censored), 2 = CL (censored due to liver transplantation)
    if prediction[0] == 0:
        pred = 'D (death)'
    elif prediction[0] == 1:
        pred = 'C (censored)'
    else:
        pred = 'CL (censored due to liver transplantation)'

    result = {
        "Probability of Status_D": round(float(proba[0][0]), 3),
        "Probability of Status_C": round(float(proba[0][1]), 3),
        "Probability of Status_CL": round(float(proba[0][2]), 3),
        "Prediction of the patient": str(pred)
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9696)
