import os
from Skin_Project.params import *
import pandas as pd

# Replace mole type with a number
def categorize(df):
    df['dx'] = df['dx'].replace({'nv': 4, 'mel': 6, 'bkl': 2, 'bcc': 1, 'akiec': 0, 'vasc': 5, 'df': 3})
    return df

# Drop metadata columns for simple models
def drop_columns(df):
    df.drop(columns=['Unnamed: 0', 'lesion_id', 'dx_type', 'age', 'sex', 'localization'])
    return df


# Labels between 1 (malign) and 0 (benign)
def labelize(df):
    benign_classes = [4,2,3]
    for i in range(1,df.shape[0]+1):
        if df['dx'][i-1] in benign_classes:
            df.loc[i-1, "dx"] = 0
            #new_df ['dx'][i-1] = 0
        else:
            df.loc[i-1, "dx"] = 1
    return df

# Splits into a SAMPLE_SIZE% sample
def sampler(df):
    df_sample = df.sample(round(SAMPLE_SIZE*df.shape[0]), random_state = 42)
    return df_sample
