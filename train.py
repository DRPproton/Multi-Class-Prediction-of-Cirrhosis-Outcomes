"""
Capstone Project 1
Multi-Class Prediction of Cirrhosis Outcomes
Dashel Ruiz Perez 12/10/2023
"""

# Importig libraries
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder

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


# Function to process days columns
def process_day_columns(input_df):
    from datetime import datetime
    # Define the starting date
    df = input_df.copy()
    start_date = datetime(1986, 7, 1)

    # Convert days to months and years using pandas
    df['date'] = start_date + pd.to_timedelta(df['N_Days'], unit='D')
    df['N_Months'] = (df['date'].dt.year - start_date.year) * 12 + df['date'].dt.month - start_date.month
    # Age column to year
    df['Age'] = df['Age'] // 365.25 

    df.drop(columns='date', inplace=True)

    return df

df_proc = process_day_columns(df)

# Converting 'Stage' to category
df_proc.Stage = df_proc.Stage.astype('category')


# Function to make new features
def feature_eng_num_to_cat(df):
    df_temp = df.copy()
    df_temp['normal_patelets'] = df_temp.Platelets.map(lambda x: "N" if x < 150 else "Y") # Pateletes
    df_temp['normal_cholesterol'] = df_temp.Cholesterol.map(lambda x: "Y" if x < 201 else "N") # Cholesterol
    df_temp['normal_tryglicerides'] = df_temp.Tryglicerides.map(lambda x: "Y" if x < 151 else "N") # Tryglicerides
    df_temp['normal_SGOT'] = df_temp.SGOT.map(lambda x: "Y" if x < 41 else "N") # SGOT
    df_temp['normal_copper'] = df_temp.Copper.map(lambda x: "Y" if x >= 62 and x <= 140 else "N") # Copper
    df_temp['normal_bilirubin'] = df_temp.Bilirubin.map(lambda x: "Y" if x >= 0.2 and x <= 1.2 else "N") # Bilirubin
    df_temp['normal_albumin'] = df_temp.Albumin.map(lambda x: "Y" if x >= 3.4 and x <= 5.4 else "N") # Albumin

    return df_temp

df_new_features = feature_eng_num_to_cat(df_proc)

# Converting target to numeric
df_new_features.Status = df_new_features.Status.map({"D": 0, "C": 1, "CL": 2})


cols_to_encode = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Stage', 
                'normal_patelets', 'normal_cholesterol', 'normal_tryglicerides', 
                'normal_SGOT', 'normal_copper', 'normal_bilirubin', 'normal_albumin']
# Making the encoder
encoder = OrdinalEncoder()

# Function with encode categorical variable
def encode_cols(df, bi_cols, dummy_nom_list, encoder):
    df_temp = df.copy()
    df_temp[bi_cols] = encoder.fit_transform(df_temp[bi_cols])

    for col in  dummy_nom_list:
        df_temp = pd.concat([df_temp.drop(col, axis = 1),
                            pd.get_dummies(df_temp[col], prefix = col, prefix_sep = '_', drop_first = True, dummy_na = False, dtype='int32')],
                        axis=1)

    return df_temp

df_final = encode_cols(df_new_features, cols_to_encode, ['Edema'],encoder)

# Split the data and Standarize
X = df_final.drop(columns='Status')
y = df_final.Status

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2, stratify=y)


sc = RobustScaler()
sc_df = sc.fit_transform(X_train)

X_train = pd.DataFrame(sc_df, columns=X_train.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)

print(X_test.head(3))
print(X_train.head(3))