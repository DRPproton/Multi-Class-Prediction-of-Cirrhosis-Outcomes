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
df_train = pd.read_csv('data/train.csv')   # Syntetic data
df_original = pd.read_csv('data/cirrhosis.csv')  # original dataset