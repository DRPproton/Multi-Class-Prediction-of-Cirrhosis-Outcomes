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

# Supressing warnigns
import warnings
warnings.filterwarnings("ignore")

# Importing the datasets
print('Importing datasets...')
df_train = pd.read_csv('data/train.csv')   # Syntetic data
df_original = pd.read_csv('data/cirrhosis.csv')  # original dataset
time.sleep(1)


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

print(df_proc.head())


