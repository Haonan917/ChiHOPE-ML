import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import load_omics
from models.seed import set_global_seed
import argparse
import yaml
import os
from pathlib import Path
import pandas as pd
from models.classfication_models import optimise_all_models, refit_all_models
from sklearn.model_selection import StratifiedKFold
import pickle

train_endpoint_path = "data/train_endpoint_latest.tsv"
test_endpoint_path = "data/validation_endpoint_latest.tsv"
BINARY_ENDPOINTS = ['y.90', 'y.158']

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    seed = args.seed
    set_global_seed(seed)
    # Endpoint
    endpoint = args.endpoint
    # Output
    outpud_dir = args.output_dir
    # Input
    data_config = load_config(args.data_config_path)
    data_type = data_config['data_type']
    omics_parent_dir = data_config['omics_parent_dir']
    discovery_dir_name = data_config['discovery_dir_name']
    
    omics_discovery_path = [os.path.join(omics_parent_dir, discovery_dir_name, omics_file)
                            for omics_file in data_config['omics_list']]
    omics_type_list = [omics_file[:3] for omics_file in data_config['omics_list']]
    
    # Feature selection
    feature_selection_dir = args.feature_selection_dir
    selected_feature_file = args.selected_feature_file
    return_with_mismatch_flag = args.return_with_mismatch
    
    def check_mismatched_omics(selected_feature_lists, omics_type_list):
        ignored_omics_indices = [index for index, omics_type in enumerate(omics_type_list) if len(selected_feature_lists[index]) == 0]
        for index in ignored_omics_indices:
            print(f"You have listed {omics_type_list[index]} in the data config file. However, no features are selected by the feature selection file, so {omics_type_list[index]} will be ignored.")
        return ignored_omics_indices
    
    selected_feature_lists = None # if this variable is none, all features will be used for modeling
    if feature_selection_dir is not None: # otherwise, only include selected features and its omics type
        selected_feature_path = Path(feature_selection_dir) / endpoint / selected_feature_file
        
        if selected_feature_path.exists():
            selected_feature_df = pd.read_csv(selected_feature_path, sep=",", index_col=0)
            selected_feature_lists = [selected_feature_df[selected_feature_df["omics_type"]==omics_type].index.tolist()
                                    for omics_type in omics_type_list]
            
            ignored_omics_indices = check_mismatched_omics(selected_feature_lists=selected_feature_lists, omics_type_list=omics_type_list)

            if ignored_omics_indices and return_with_mismatch_flag:
                print("Return for mismatches in the number of omics types between data configuration and feature selection")
                return

            print("Omics datasets provided: {}".format(', '.join(omics_type_list)))
            # Filter ignored omics type
            omics_discovery_path = [omics_discovery_path[i] for i in range(len(omics_discovery_path)) if i not in ignored_omics_indices]
            omics_type_list = [omics_type_list[i] for i in range(len(omics_type_list)) if i not in ignored_omics_indices]
            selected_feature_lists = [selected_feature_lists[i] for i in range(len(selected_feature_lists)) if i not in ignored_omics_indices]
            
            dataset_num = len(omics_discovery_path)
            print("Omics datasets left: {}".format(', '.join(omics_type_list)))      
    
    # Output
    interim_path = Path(os.path.join(outpud_dir, endpoint, 'result_{}'.format(str(seed))))
    interim_path.mkdir(exist_ok=True, parents=True)
       
    # load omics file
    X_array, y = load_omics(omics_discovery_path, train_endpoint_path, endpoint, selected_feature_lists=selected_feature_lists)
    selected_feature_lists = [omics_data.columns.to_list() for omics_data in X_array]
    X= pd.concat(X_array, axis=1)
    print(f"Shape of original data: {X.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    splits = []
    for (tr,ts) in cv.split(X, y):
        splits.append((tr,ts))
    trained_models = optimise_all_models(X, y, splits)
    refit_models = refit_all_models(X, y, results=trained_models, splits=splits, omicsComb=data_type, save_dir=interim_path, random_state=int(seed))

    with open(interim_path / 'output_models_{}.pkl'.format(data_type), 'wb') as f:
        pickle.dump(trained_models, f)
    with open(interim_path / 'output_refits_{}.pkl'.format(data_type), 'wb') as f:
        pickle.dump(refit_models, f)

if __name__ == '__main__':    
    # data_config = './config/cgz_main_config/cg/select_1/CLN.yaml'
    data_config = './config/cgz_main_config/cg/select_5/CLN_MTB_PRT_MIR_MET.yaml'
    # data_config = './config/cgz_main_config/cg/select_3/CLN_MTB_MET.yaml'

    output_dir = './output_predictivity/'
    endpoint = 'y.90'
    feature_selection_dir = None

    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument('--data_config_path', type=str, default=data_config)
    parser.add_argument('--output_dir', type=str, default=output_dir, help="parent output directory")
    # Task
    parser.add_argument('--endpoint', type=str, default=endpoint, help="predictive endpoint")
    # Reproductibility
    parser.add_argument('--seed', type=int, default=2, help="random seed for initialization")
    # Feature selection
    parser.add_argument('--feature_selection_dir', type=str, default=feature_selection_dir, nargs='?', help="Feature selection results directory")
    parser.add_argument('--selected_feature_file', type=str, default="selected_feature_lists.csv", help="Feature selection file")
    parser.add_argument('--return_with_mismatch', action='store_true', help="Specify whether to proceed when the number of omics types mismatches between data configuration and feature selection.")

    args = parser.parse_args()
    main(args)
