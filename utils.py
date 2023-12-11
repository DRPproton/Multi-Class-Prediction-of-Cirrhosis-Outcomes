# Importig libraries
import pandas as pd


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