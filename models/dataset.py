import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_omics(omics_lists, enpoint_dict_path, endpoint, selected_feature_lists=None):
    all_endpoint_df = pd.read_csv(enpoint_dict_path, delimiter='\t', index_col=0)
    # Filter samples treated with chiglitazar
    cg_endpoint_df = all_endpoint_df.loc[all_endpoint_df['Treatment'].isin(['CGZ_32mg', 'CGZ_48mg'])]
    # Select the target endpoint, e.g., y.90, which refers to the therapy's effect on HbA1c.
    endpoint_df = cg_endpoint_df[[endpoint]]
    # Filter out samples without ground truth.
    filter_endpoint_df = endpoint_df.dropna()
    filter_endpoint_df.index = filter_endpoint_df.index.astype(str)

    intersect_sample_lists = filter_endpoint_df.index.to_list()
    print(f"num of samples which has endpoint value: {len(intersect_sample_lists)}")
    all_omics_df = []
    for omics_file in omics_lists:
        file_name = omics_file.split('/')[-1]
        with open(omics_file, 'rb') as file:
            omics_df = pickle.load(file) # shape [feature, sample]
        omics_df.index = omics_df.index.astype(str)
        all_omics_df.append(omics_df)
        
        sample_list = omics_df.columns.to_list()
        intersect_sample_lists = [sample_id for sample_id in sample_list if sample_id in intersect_sample_lists]
    
    selected_omics_df = []
    for index, omics_df in enumerate(all_omics_df):
        omics_df = omics_df.loc[:, intersect_sample_lists]
        if selected_feature_lists is not None and len(selected_feature_lists[index])!= 0:
            select_features = selected_feature_lists[index]
            omics_features = omics_df.index.to_list()
            intersect_feature_lists = set(omics_features).intersection(set(select_features))
            select_features = [feature_id for feature_id in select_features if feature_id in intersect_feature_lists]
            omics_df = omics_df.loc[select_features, :]
  
        omics_df = omics_df.transpose()
        selected_omics_df.append(omics_df)
        
        
    target_y_values_dict = filter_endpoint_df.loc[intersect_sample_lists, endpoint].to_dict()
    y_labels = [target_y_values_dict[sample_id] for sample_id in intersect_sample_lists] 

    return selected_omics_df, np.array(y_labels) # omics_array, ground truth

