import pandas as pd
from pandas_profiling import ProfileReport #https://github.com/ydataai/pandas-profiling
from sklearn import preprocessing


def clean_data(df, nulls=False):
    #identify null values
    if nulls:
        df.replace('?', None, inplace=True)
    
    #switch to correct data type
    df['admission_type_id'] = df['admission_type_id'].astype(str)
    df['discharge_disposition_id'] = df['discharge_disposition_id'].astype(str)
    df['admission_source_id'] = df['admission_source_id'].astype(str)
    
    #drop id columns
    df.drop('encounter_id', axis=1, inplace=True)
    df.drop('patient_nbr', axis=1, inplace=True) #TODO try combining entries?
    
    #drop columns with only one value
    df.drop('examide', axis=1, inplace=True)
    df.drop('citoglipton', axis=1, inplace=True)
    return df


def preprocess(df):
    #TODO try scaling to 0-1
    #TODO try z-score scaling
    #TODO try methods for turning categorial data numeric
    #https://stackoverflow.com/questions/59538006/scaling-data-frame-with-numeric-and-categorical
    return df


def generate_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("data_exploration_org.html")


def get_data():
    df = pd.read_csv('data/diabetic_data.csv')
    df = clean_data(df)
    df = preprocess(df)
    return df