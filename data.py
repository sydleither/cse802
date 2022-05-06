import pandas as pd
from pandas_profiling import ProfileReport #https://github.com/ydataai/pandas-profiling
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
    replace_age = {'age': {'[0-10)': '[0-30)', '[10-20)': '[0-30)', '[20-30)':'[0-30)', \
                   '[30-40)': '[30-50)', '[40-50)': '[30-50)'}}
    df.replace(replace_age, inplace=True)
    
    #rebinning
    remove_temp = {}
    all_cat = set(list(df['diag_1'])+list(df['diag_2'])+list(df['diag_3']))
    for x in list(range(390,460))+[785]:
        remove_temp[str(x)] = 'circulatory'
    for x in list(range(460,520))+[786]:
        remove_temp[str(x)] = 'respiratory'
    for x in list(range(520,580))+[787]:
        remove_temp[str(x)] = 'digestive'
    for x in [y for y in all_cat if '250' in y]:
        remove_temp[str(x)] = 'diabetes'
    for x in list(range(800,1000)):
        remove_temp[str(x)] = 'injury'
    for x in list(range(710,740)):
        remove_temp[str(x)] = 'musculoskeletal'
    for x in list(range(580,630))+[788]:
        remove_temp[str(x)] = 'genitourinary'
    neo = list(range(1,280))+[780, 781, 782, 784]+list(range(790,800)) \
        +list(range(290,320))+list(range(680,710))
    neo.remove(250)
    for x in neo:
        remove_temp[str(x)] = 'neoplasms'
    other = list(range(280,290))+list(range(320,390))+list(range(630,680)) \
        +list(range(740,760))+[y for y in all_cat if 'V' in y or 'E' in y] \
        +[783,789,365.44,'?']
    for x in other:
        remove_temp[str(x)] = 'other'
    remove = {'diag_1':remove_temp, 'diag_2':remove_temp, 'diag_3':remove_temp}
    df.replace(remove, inplace=True)
    
    remove_temp = {}
    physician_all = set(df['medical_specialty'])
    keep = ['InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice']
    for x in physician_all:
        if 'Surg' in x:
            remove_temp[x] = 'Surgery'
        elif 'Cardiology' in x:
            remove_temp[x] = 'Cardiology'
        elif x not in keep:
            remove_temp[x] = 'Other'
    remove = {'medical_specialty':remove_temp}
    df.replace(remove, inplace=True)
    
    df.loc[df['discharge_disposition_id'] != '1', 'discharge_disposition_id'] = 'other'
    
    remove_temp = {}
    admission_all = set(df['admission_source_id'])
    for x in admission_all:
        if x in ['1', '2', '3']:
            remove_temp[x] = 'referred'
        elif x == '7':
            remove_temp[x] = 'emergency'
        else:
            remove_temp[x] = 'other'
    remove = {'admission_source_id':remove_temp}
    df.replace(remove, inplace=True)
    
    remove = {}
    for med in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', \
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', \
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', \
                'miglitol', 'troglitazone', 'tolazamide', 'examide', \
                'citoglipton', 'glyburide-metformin', 'glipizide-metformin', \
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', \
                'metformin-pioglitazone']:
        remove[med] = {'Steady':'Yes', 'Up':'Yes', 'Down':'Yes'}
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
    pca = PCA(n_components=50).fit_transform(X)
    return pca


def feature_selection(X, y, model, n_features=None, direction='forward'):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction=direction)
    sfs.fit(X, y)
    return sfs


def generate_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("data_exploration_binned_cat.html")


def get_data(z_score):
    df = pd.read_csv('data/diabetic_data.csv')
    df = clean_data(df)
    df = normalize(df, z_score)
    df = cat_to_num(df)
    return df
