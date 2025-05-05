import os
import re
import pandas as pd
import numpy as np

def fix_columns(dfs):
    
    for i in range(len(dfs)):
        # Split the concatenated column into multiple columns
        split_cols = dfs[i]['# galaxyId haloId haloId_dor ra dec z_app log(m_200)'].str.split(' ', expand=True)

        # Concatenate split_cols with combined_df
        split_df = pd.concat([dfs[i], split_cols], axis=1)

        # Assign appropriate column names
        split_df.columns = ['original_col', 'galaxyId', 'haloId', 'haloId_dor', 'ra', 'dec', 'z_app', 'log(m_200)']

        # Drop the original concatenated column and unnecessary columns
        split_df = split_df.drop(columns=['original_col'])


        numeric_columns = ['galaxyId', 'haloId', 'haloId_dor', 'ra', 'dec', 'z_app', 'log(m_200)']
        split_df[numeric_columns] = split_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        dfs[i] = split_df
    
    for df in dfs:
        df['haloId_dor'] = df['haloId_dor'] - 1
    
    return dfs

def load_data(path):

    files = os.listdir(path)

    # Filter .cat files
    cat_files = [file for file in files if file.endswith('.cat')]

    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)  # Find the number in the filename
        return int(match.group()) if match else 0  # Return the number as an integer

    cat_files_sorted = sorted(cat_files, key=extract_number)

    # Initialize an empty list to hold individual DataFrames
    dfs = []

    # Loop through each .cat file, read into a DataFrame, and append to the list
    for file in cat_files_sorted:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, delimiter='\t')
        dfs.append(df)

    return dfs

def extract_df(data_frame):

    X_data = data_frame.iloc[:,3].values
    Y_data = data_frame.iloc[:,4].values
    Z_data = data_frame.iloc[:,5].values

    #min + (max - min)/2 = (2min + max - min)/2
    x_center = (min(X_data) + max(X_data))/2
    y_center = (min(Y_data) + max(Y_data))/2

    Z_clus = np.mean(Z_data);

    return X_data, Y_data, Z_data, x_center, y_center, Z_clus