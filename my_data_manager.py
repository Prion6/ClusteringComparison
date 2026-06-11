import os
import re
import pandas as pd
import numpy as np
import astro_utils as au
from astropy.stats import biweight_location
import glob
from openpyxl import Workbook
from itertools import zip_longest

def scrub_data(df):
    
    for i in range(len(dfs)):
        # Split the concatenated column into multiple columns
        split_cols = dfs[i]['# galaxyId haloId haloId_new ra dec z_app log(m_200)'].str.split(' ', expand=True)

        # Concatenate split_cols with combined_df
        split_df = pd.concat([dfs[i], split_cols], axis=1)

        # Assign appropriate column names
        split_df.columns = ['original_col', 'galaxyId', 'haloId', 'haloId_new', 'ra', 'dec', 'z_app', 'log(m_200)']

        # Drop the original concatenated column and unnecessary columns
        split_df = split_df.drop(columns=['original_col'])


        numeric_columns = ['galaxyId', 'haloId', 'haloId_new', 'ra', 'dec', 'z_app', 'log(m_200)']
        split_df[numeric_columns] = split_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        dfs[i] = split_df

        counts = dfs[i]['haloId_new'].value_counts()

        # Step 2: Create a dictionary to assign group labels
        group_labels = {}
        current_group_id = 1

        for halo_id, count in counts.items():
            if halo_id == 0.0:
                group_labels[halo_id] = 0  # Keep main halo ID as 0
            elif count < 4:
                group_labels[halo_id] = -1  # Too small to be a group
            else:
                group_labels[halo_id] = current_group_id
                current_group_id += 1

        # Step 3: Map the group labels to the DataFrame
        dfs[i]['group_label'] = dfs[i]['haloId_new'].map(group_labels)
    
    #for df in dfs:
        #df['haloId_dor'] = df['haloId_dor'] - 1
    
    return dfs

def load_cat(cat):
 
    df = pd.read_csv(cat, delim_whitespace=True, header=None)

    df.columns = df.iloc[0]  # Set first row as header
    df = df[1:].reset_index(drop=True)
    
    return df

def separate_halos(df):
    grouped = df.groupby('firstHaloInFOFGroupId')

    # Store each group in a dictionary, where keys are the unique IDs
    dfs = [group_df for _, group_df in grouped]

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

    Z_clus = np.mean(Z_data)

    return X_data, Y_data, Z_data, x_center, y_center, Z_clus

def separate_clusters(dfs, members_threshold = 50, log_mass_threshold = 14, Mp_size_threshold = 1.0):
    clusters = []
    groups = []

    for df in dfs:
        
        main_halo_count = (df['haloId'] == df['firstHaloInFOFGroupId']).sum()

        if main_halo_count < members_threshold:
            groups.append(df)
            continue
        
        halo_mass = float(df['log(m_200)'].iloc[0])

        if halo_mass < log_mass_threshold:
            groups.append(df)
            continue
        
        Z_data = df["z_app"]
    
        clus_z = biweight_location(Z_data)

        r_200 = au.get_r_200(halo_mass, clus_z, H0=67.3, Om0=0.3)

        if r_200 < Mp_size_threshold:
            groups.append(df)
            continue
        
        clusters.append(df)
    
    return clusters, groups

def save_dict_to_excel(results_dict, filename="results.xlsx"):

    directory = os.path.dirname(filename)
    if directory:  # avoid issues if filename has no path
        os.makedirs(directory, exist_ok=True)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for method_name, df in results_dict.items():
            # Excel sheet names cannot exceed 31 chars or contain some special symbols
            safe_name = str(method_name)[:31].replace("/", "_").replace("\\", "_")
            df.to_excel(writer, sheet_name=safe_name, index=False)
    print(f"Results saved to '{filename}'")

def load_results(path):

    results_df = {}

    files = glob.glob(os.path.join(path, "*.xlsx"))

    for f in files:

        xls = pd.ExcelFile(f)
        sheet_names = xls.sheet_names

        key = os.path.splitext(os.path.basename(f))[0]

        df_time = pd.read_excel(xls, sheet_name=sheet_names[0])

        executions = []

        for name in sheet_names[1:]:
            df = pd.read_excel(xls, sheet_name=name)
            df.columns = df.columns.map(str)
            executions.append(df)

        results_df[key] = {
            "time": df_time,
            "executions": executions
        }

    return results_df

def transpose_list_of_dfs(dfs):
    n_exec = len(dfs)
    n_features = dfs[0].shape[1]

    feature_dfs = {}

    for col_idx in range(n_features):
        feature_id = dfs[0].columns[col_idx]

        col_data = [df.iloc[:, col_idx].reset_index(drop=True) for df in dfs]

        df_new = pd.concat(col_data, axis=1)
        df_new.columns = [f"exec_{i+1}" for i in range(n_exec)]

        feature_dfs[feature_id] = df_new

    return feature_dfs

def save_xlsx(results, title):
    wb = Workbook()

    # 1. Setup the Execution Times sheet (Rows = Iterations, Cols = Samples/Runs)
    sheet1 = wb.active
    sheet1.title = "Execution Times"
    
    # Extract just the duration lists from the results
    # results = [( [times], [preds] ), ( [times], [preds] )]
    all_duration_lists = [res[0] for res in results]
    
    # Create Headers: "Iteration", "Sample 1", "Sample 2", etc.
    headers = ["Iteration"] + [f"Run {i+1}" for i in range(len(all_duration_lists))]
    sheet1.append(headers)

    # Use zip_longest to pair up times by iteration index
    # fillvalue="" handles cases where one run has fewer iterations than others
    for idx, row_times in enumerate(zip_longest(*all_duration_lists, fillvalue="")):
        # Append iteration number (idx+1) followed by the times for that iteration
        sheet1.append([idx + 1] + list(row_times))

    # 2. Store the prediction data in separate sheets as before
    for res_idx, (_, data) in enumerate(results):
        sheet = wb.create_sheet(title=f"Result_{res_idx + 1}")
        
        # This keeps your original logic for predictions
        for row in zip_longest(*data, fillvalue=""):
            sheet.append(row)

    wb.save(f"{title}.xlsx")