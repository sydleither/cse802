import pandas as pd
#from pandas_profiling import ProfileReport #https://github.com/ydataai/pandas-profiling
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
    
    #drop columns with only one value
    df.drop('examide', axis=1, inplace=True)
    df.drop('citoglipton', axis=1, inplace=True)
    
    #following Strack et al.
    df.drop('payer_code', axis=1, inplace=True)
    df.drop('weight', axis=1, inplace=True)
    df = df[(df['discharge_disposition_id'] !='11') & \
            (df['discharge_disposition_id'] !='19') & \
            (df['discharge_disposition_id'] !='20') & \
            (df['discharge_disposition_id'] !='20') & \
            (df['discharge_disposition_id'] !='13') & \
            (df['discharge_disposition_id'] !='14')]
    df = df.drop_duplicates(subset=['patient_nbr'])
    df.drop('patient_nbr', axis=1, inplace=True)
    df.drop('glimepiride-pioglitazone', axis=1, inplace=True)
    
    #improving runtime
    remove = {'diag_1':{'?':'other'}, 'diag_2':{'?':'other'}, 'diag_3':{'?':'other'}}
    for i in [1,2,3]:
        uniq = set(df[f'diag_{i}'])
        for item in uniq:
            if len(df.loc[df[f'diag_{i}'] == item]) / len(df) < 0.035:
                remove[f'diag_{i}'][item] = 'other'
    df.replace(remove, inplace=True)
    return df


def cat_to_num(df):
    X = df.drop('readmitted', axis=1)
    X = pd.get_dummies(X, dtype=int)
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
    pca = PCA(n_components=10).fit_transform(X)
    return pca


def feature_selection(X, y, model, n_features=None, direction='forward'):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)
    sfs.fit(X, y)
    return sfs


def generate_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("data_exploration_informed.html")


def get_data():
    df = pd.read_csv('data/diabetic_data.csv')
    df = clean_data(df)
    df = normalize(df, False)
    df = cat_to_num(df)
    return df