import numpy as np
import pandas as pd

# Columns that had more than 10% of missing values on the training dataset ('air_system_previous_years.csv')
COLS_TO_DROP = ['ed_000', 'cl_000', 'cm_000', 'ec_00', 'dc_000', 'db_000', 'da_000',
       'cz_000', 'cy_000', 'cu_000', 'cv_000', 'ct_000', 'cx_000', 'ad_000',
       'ch_000', 'cg_000', 'co_000', 'cf_000', 'bk_000', 'bl_000', 'bm_000',
       'bn_000', 'ab_000', 'cr_000', 'bo_000', 'bp_000', 'bq_000', 'br_000']

CATEGORICAL_COLUMNS = ['true_class', 'cd_000', 'predicted_class'] # Columns that had less than 10 unique values on the training dataset were consider categorical columns


def clean_dataframe(df: pd.DataFrame):
    return cast_float(drop_columns(replace_na(df)))

def replace_na(df: pd.DataFrame) -> pd.DataFrame:
    for i in df.columns:
        df[i] = df[i].replace('na', np.nan)
    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLS_TO_DROP)

def cast_float(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = [col for col in df.columns if col not in CATEGORICAL_COLUMNS]
    for var in numerical_cols:
        df[var] = df[var].astype(float)
    return df

def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={'class': 'class_'}, inplace=True)
    return df

def dict_to_dataframe(data: dict) -> pd.DataFrame:
    rows = [item.dict() for item in data]
    df = pd.DataFrame(rows)
    return df
