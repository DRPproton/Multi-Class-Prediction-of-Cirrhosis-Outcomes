"""
Capstone Project 1
Multi-Class Prediction of Cirrhosis Outcomes
Dashel Ruiz Perez 12/10/2023
"""

# Importig libraries
import pandas as pd
import pickle
import time
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction import DictVectorizer
from utils import *

# Supressing warnigns
import warnings
warnings.filterwarnings("ignore")

# Importing the datasets
print('Importing datasets...')
df_train = pd.read_csv('data/train.csv')   # Syntetic data
df_original = pd.read_csv('data/cirrhosis.csv')  # original dataset
time.sleep(1)


print('Preparing the data...')
# Preparing datasets
df = pd.concat([df_train, df_original], axis =0)
# drop the "id" column
df.drop(columns=['id', 'ID'], inplace=True)
df.reset_index(drop=True, inplace=True)
# Dropping ows with null values
df = df.dropna()

# Processing day columns
df_proc = process_day_columns(df)

# Converting 'Stage' to category
df_proc.Stage = df_proc.Stage.astype('category')

# Feature eng
df_new_features = feature_eng_num_to_cat(df_proc)

# Converting target to numeric
df_new_features.Status = df_new_features.Status.map({"D": 0, "C": 1, "CL": 2})


# Spliting the data
X = df_new_features.drop(columns='Status')
y = df_new_features.Status

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)

# DictVectorizer
dv = DictVectorizer(sparse=False)

train_dict = X_train.to_dict(orient='records')
test_dict = X_test.to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
X_test = dv.transform(test_dict)

# Scaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Model
best_params = {'booster': 'gbtree', 
               'max_depth': 11, 
               'learning_rate': 0.04216709720284281, 
               'n_estimators': 504, 
               'min_child_weight': 1, 
               'subsample': 0.3437341948452076, 
               'colsample_bylevel': 0.9831950290382536, 
               'colsample_bytree': 0.1720049167961935, 
               'colsample_bynode': 0.7420031757532206, 
               'reg_alpha': 0.9242740053249385, 
               'reg_lambda': 0.9419384081768526, 
               'eval_metric': 'mlogloss'}

xgb = XGBClassifier(**best_params)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_test)


print('Making prediciions...')
print(f'Logloss score => {log_loss(y_test, xgb_pred)}')
print(f"roc_auc score = >{roc_auc_score(y_test, xgb_pred, multi_class='ovr')}")
print(xgb_pred[0])

time.sleep(2)

# Exporting model
print('Exporting model...')
with open('xgb_model.bin', 'wb') as f_out:
    pickle.dump(xgb, f_out)

with open('dv_model.bin', 'wb') as f_out:
    pickle.dump(dv, f_out)

with open('scaler.bin', 'wb') as f_out:
    pickle.dump(sc, f_out)

# Imprting models for testing
time.sleep(1)
print('Importing saved model...')
with open('xgb_model.bin', 'rb') as f_in:
    xgb_model = pickle.load(f_in)

with open('dv_model.bin', 'rb') as f_in:
    dv_model = pickle.load(f_in)

with open('scaler.bin', 'rb') as f_in:
    scaler_model = pickle.load(f_in)


# Testing model
patient = {
 'N_Days': 3839,
 'Drug': 'D-penicillamine',
 'Age': 19724,
 'Sex': 'F',
 'Ascites': 'N',
 'Hepatomegaly': 'Y',
 'Spiders': 'N',
 'Edema': 'N',
 'Bilirubin': 1.2,
 'Cholesterol': 546.0,
 'Albumin': 3.37,
 'Copper': 65.0,
 'Alk_Phos': 1636.0,
 'SGOT': 151.9,
 'Tryglicerides': 90.0,
 'Platelets': 430.0,
 'Prothrombin': 10.6,
 'Stage': 2.0}

patiente_df = pd.DataFrame([patient])

df_test = process_day_columns(patiente_df)
df_test.Stage = df_test.Stage.astype('category')
df_test = feature_eng_num_to_cat(df_test)

df_test_dict = df_test.to_dict(orient='records')

df_test = dv_model.transform(df_test_dict)
df_test = scaler_model.transform(df_test)


prediction = xgb_model.predict(df_test)
proba = xgb_model.predict_proba(df_test)


print(f'Prediction with saved model => {prediction[0]}')
print(f'Prediction probability with saved model => {proba}')

print(f'Probability of Status_C => {proba[0][1]}')
print(f'Probability of Status_CL => {proba[0][2]}')
print(f'Probability of Status_D => {proba[0][0]}')
