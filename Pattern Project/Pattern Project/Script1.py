import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

def hourPreProcessing(df, columnName):
    for i in range(len(df)):
        oldValue = df.loc[i, columnName]
        if ':' in oldValue:
            hour,minute = oldValue.split(':')
            hour=float(hour)*60
            minute = float(minute)
            newValue = hour+minute
            newValue /=60
        else:
            newValue = float(oldValue)
        df.loc[i, columnName] = newValue

def columnDropper(df, columnName):
    df.drop(columnName, axis=1, inplace=True)

def ColumnSeparator(df, columnName):
    for i in range(len(df)):
        value = df.loc[i, columnName]
        items = value.split(', ')
        for item in items:
            key, val = item.split(': ')
            key=key.strip("{'")
            val=val.strip("'}")
            df.loc[i, key] = val
    df.drop(columns=[columnName], inplace=True)

def encodingData(CateData):
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    for column in CateData.columns:
        for i in range(len(CateData[column])):
            if CateData[column].iloc[i] not in label_encoders[column].classes_:
                CateData[column].iloc[i] = "Others"
        CateData[column] = label_encoders[column].transform(CateData[column])
    return CateData

def regressionScriptTester():
    data= pd.read_csv("parkinsons_disease_data_reg.csv")
    hourPreProcessing(data,'WeeklyPhysicalActivity (hr)')
    columnDropper(data, 'PatientID')
    columnDropper(data, 'DoctorInCharge')
    numerical_columns = [
        'UPDRS', 'CholesterolHDL', 'BMI', 'MoCA',
        'CholesterolTotal', 'DiastolicBP', 'AlcoholConsumption',
        'CholesterolTriglycerides', 'SystolicBP', 'Age'
    ]    
    numeric_Data = data[numerical_columns]
    with open('mean.pk1', 'rb') as f:
        mean = pickle.load(f)
        
    numeric_Data= numeric_Data.fillna(mean)

    with open('scaler.pk1', 'rb') as f:
        loaded_scaler = pickle.load(f)
    numeric_Data = loaded_scaler.transform(numeric_Data)
    
    ColumnSeparator(data,'MedicalHistory')
    ColumnSeparator(data,'Symptoms')
    categorical_columns = ['PosturalInstability', 'Depression', 'Gender', 'Hypertension', 
                           'Bradykinesia', 'FamilyHistoryParkinsons', 'Diabetes', 'Stroke', 'SleepDisorders', 'Tremor']
    
    category_Data = data[categorical_columns]
    with open('mode.pk1', 'rb') as f:
        mode = pickle.load(f)
    category_Data = category_Data.fillna(mode)
    
    encoded_df= encodingData(category_Data)
    encoded_df['Disease Symptoms'] =  (encoded_df['Tremor'] + encoded_df['Bradykinesia'] + encoded_df['SleepDisorders'] + encoded_df['PosturalInstability']) / 4
    encoded_df['ChronicDiseasesScore'] = (encoded_df['Hypertension'] + encoded_df['Diabetes']) / 2
    encoded_df.drop(['Tremor', 'Bradykinesia','SleepDisorders', 'PosturalInstability','Hypertension', 'Diabetes'], axis=1, inplace=True)
regressionScriptTester()