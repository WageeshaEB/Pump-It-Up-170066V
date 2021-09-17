import numpy as np
import pandas as pd


def map_feature_date_recorded(df):
    datetime_obj = pd.to_datetime(df['date_recorded'])
    df["day"] = datetime_obj.dt.day
    df["month"] = datetime_obj.dt.month
    df["year"] = datetime_obj.dt.year
    df = df.drop(['date_recorded'], axis=1)
    return df


def map_feature_construction_year(df):
    df.loc[df.construction_year <= 0, df.columns == 'construction_year'] = 1950
    return df


def map_longitude_latitude(df, min_bound):
    df = df.copy()
    latitude, longitude, region_code = 'latitude', 'longitude', 'region_code'

    # print(df[latitude].value_counts())
    # print(df[longitude].value_counts())

    # min_latitude_df = df[df[latitude] < min_bound]
    min_longitude_df = df[df[longitude] < min_bound]

    # print("latitude & longitude less than the bound -", len(min_latitude_df.index), len(min_longitude_df.index))

    min_longitude_df.iloc[:, df.columns == longitude] = np.nan
    min_longitude_df.iloc[:, df.columns == latitude] = np.nan

    # df[df[latitude] < min_bound] = min_latitude_df
    df[df[longitude] < min_bound] = min_longitude_df

    df[latitude] = df.groupby(region_code).transform(lambda x: x.fillna(x.mean()))[latitude]
    df[longitude] = df.groupby(region_code).transform(lambda x: x.fillna(x.mean()))[longitude]

    return df


def map_gps_height(df, min_bound):
    df = df.copy()
    gps_height, region_code = 'gps_height', 'region_code'
    # print(df[gps_height].value_counts())
    min_gps_height_df = df[df[gps_height] < min_bound]
    min_gps_height_df.iloc[:, df.columns == gps_height] = np.nan
    df[df[gps_height] < min_bound] = min_gps_height_df
    df[gps_height] = df.groupby(region_code).transform(lambda x: x.fillna(x.mean()))[gps_height]
    # print("\nvalue missing columns\n\n", df.isnull().sum())
    df = df.fillna(df.mean())
    return df
