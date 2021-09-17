import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# impute missing data

def fill_null_values_using_most_frequent(dataframe, columns):
    for col in columns:
        dataframe[col] = dataframe[col].fillna((dataframe[col].value_counts().index[0]))
    return dataframe


def fill_null_values_using_median(dataframe, columns):
    for col in columns:
        dataframe[col] = dataframe[col].fillna((dataframe[col].median()))
    return dataframe


def fill_null_values_using_mean(dataframe, columns):
    for col in columns:
        dataframe[col] = dataframe[col].fillna((dataframe[col].mean()))
    return dataframe


# encoding

def label_encoding(dataframe, columns):
    LE = LabelEncoder()
    for col in columns:
        dataframe[col] = LE.fit_transform(dataframe[col])
    return dataframe


def one_hot_encoding(dataframe, column):
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc_df = pd.DataFrame(OH_encoder.fit_transform(dataframe[[column]]))
    enc_df.columns = OH_encoder.get_feature_names([column])
    dataframe.drop([column], axis=1, inplace=True)
    dataframe = dataframe.join(enc_df)
    return dataframe


# normalization

def normalize_min_max(df):
    print("[Normalize - Min Max]")
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def normalize_standard(df):
    print("[Normalize - Std]")
    std_scaler = StandardScaler()
    df_normalized = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
    return df_normalized
