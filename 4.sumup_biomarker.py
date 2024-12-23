import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import re
import os
import yaml
import re

    
ENPOINT_PATTERN = "y.[\d]+"
BINARY_ENDPOINTS = ['y.90', 'y.158']


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def expand_shap_values(shap_df, all_features):
    model_features = shap_df.columns.to_list()
    expanded_shap = pd.DataFrame(0, index=np.arange(shap_df.shape[0]), columns=all_features)
    expanded_shap[model_features] = shap_df.values
    return expanded_shap

def save_SHAP(shap_values, origin_values, feature_list, output_dir, file_prefix):
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_list,
        'feature_importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='feature_importance', key=lambda x: abs(x),
                                                            ascending=False)
    
    shap_df =  pd.DataFrame(shap_values, columns= feature_list)
    origin_df = pd.DataFrame(origin_values, columns= feature_list)
    
    feature_importance_df.to_csv(output_dir / "{}feature_importance.tsv".format(file_prefix), sep='\t', index=False)
    shap_df.to_csv(output_dir / "{}shap_values.tsv".format(file_prefix), sep='\t', index=False)
    origin_df.to_csv(output_dir / "{}origin_values.tsv".format(file_prefix), sep='\t', index=False)


def merge_SHAP(origin_df_list, shap_df_list):
    shap_features_list = [shap_df.columns.to_list() for shap_df in shap_df_list]
    all_equal = all(features == shap_features_list[0] for features in shap_features_list)
    if not all_equal:
        raise("SHAP feature lists differ.")
    
    union_features  = shap_features_list[0]

    ensemble_shap_df = sum(shap_df_list) / len(shap_df_list)
    ensemble_origin_df = origin_df_list[0]

    return union_features, ensemble_origin_df, ensemble_shap_df

def calculate_avg_shap_importance(results, save_dir):
    for omics in results.keys():
        shap_df_list = [shap_df for (origin_df, shap_df) in results[omics]]
        origin_df_list = [origin_df for (origin_df, shap_df) in results[omics]]

        union_features, integrated_origin_df, integrated_shap_df =  merge_SHAP(origin_df_list, shap_df_list)
        save_SHAP(
            shap_values=integrated_shap_df.values, 
            origin_values=integrated_origin_df.values, 
            feature_list=union_features, 
            output_dir=Path(save_dir),
            file_prefix='integrated_avg_{}_'.format(omics)
        )


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    PATTERN = r"(?P<algo>.*?)_(?P<omics_type>.*?)_shap_values.tsv"
    
    # 获取路径下的所有文件夹
    endpoints = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    perfm_df = None
    for endpoint in endpoints:
        if not endpoint.startswith('y.'):
            continue
        print(endpoint)

        results = {}
        endpoint_dir = os.path.join(output_dir, endpoint)
        seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
        for seed in seeds:
            seed_dir = os.path.join(endpoint_dir, seed)
            files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
            file_names = [f for f in files if f.endswith('shap_values.tsv')]
            algos = list(set([re.match(PATTERN, f).group("algo") for f in file_names]))
            algos = [algo for algo in algos if algo != "avg"]
            omics = list(set([re.match(PATTERN, f).group("omics_type") for f in file_names]))

            for omic in omics:  
                shap_list = [os.path.join(seed_dir, "{}_{}_shap_values.tsv".format(algo, omic)) for algo in algos]
                shap_df_list = [pd.read_csv(shap_path, sep='\t') for shap_path in shap_list]
                
                origin_list = [os.path.join(seed_dir, "{}_{}_origin_values.tsv".format(algo, omic)) for algo in algos]
                origin_df_list = [pd.read_csv(origin_path, sep='\t') for origin_path in origin_list]
                
                union_features, ensemble_origin_df, ensemble_shap_df = merge_SHAP(origin_df_list, shap_df_list)

                # Save shap values of ensemble predictor per random seed
                save_SHAP(
                    shap_values=ensemble_shap_df.values, 
                    origin_values=ensemble_origin_df.values, 
                    feature_list=union_features, 
                    output_dir=Path(seed_dir),
                    file_prefix='avg_{}_'.format(omic)
                )

                if omic not in results:
                    results[omic] = []
                results[omic].append((ensemble_origin_df, ensemble_shap_df))

    # Save mean shap values of ensemble predictor across all random seeds
    calculate_avg_shap_importance(results, endpoint_dir)
                
                
if __name__ == '__main__':
    dir_name = './output_predictivity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=dir_name, help="parent output directory")
    parser.add_argument('--output_dir', type=str, default=dir_name, help="parent output directory")
    args = parser.parse_args()
    main(args)
