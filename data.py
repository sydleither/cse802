import pandas as pd
from pandas_profiling import ProfileReport #https://github.com/ydataai/pandas-profiling
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA


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


def cat_to_num(df):
    X = df.drop('readmitted', axis=1)
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    X['readmitted'] = df['readmitted']
    return X


def normalize(df, z_score=False):
    if z_score:
        scaler = StandardScaler() #z-score
    else:
        scaler = MinMaxScaler() #0-1
    numerics = df.select_dtypes(exclude=['object'])
    df[numerics.columns] = scaler.fit_transform(numerics)
    return df


def dimensionality_reduction(X):
    pca = PCA(n_components='mle').fit(X)
    return pca


def feature_selection(X, y, model, n_features=None, direction='forward'):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)
    sfs.fit(X, y)
    X = X[X.columns[sfs.get_support()]]
    return X


def generate_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("data_exploration_org.html")


def get_data():
    df = pd.read_csv('data/diabetic_data.csv')
    df = clean_data(df)
    df = normalize(df, False)
    df = cat_to_num(df)
    return df