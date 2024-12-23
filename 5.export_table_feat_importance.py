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
    PATTERN = r"(?P<algo>.*?)_CLN_MTB_PRT_MIR_MET_feature_importance.tsv"
    
    # 获取路径下的所有文件夹
    endpoints = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    perfm_df = None
    for endpoint in endpoints:
        if not endpoint.startswith('y.'):
            continue
        print(endpoint)

        feat_results = {}
        endpoint_dir = os.path.join(output_dir, endpoint)
        seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
        for seed in seeds:
            seed_dir = os.path.join(endpoint_dir, seed)
            files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
            file_names = [f for f in files if f.endswith('CLN_MTB_PRT_MIR_MET_feature_importance.tsv')]
            algos = list(set([re.match(PATTERN, f).group("algo") for f in file_names]))

            for algo in algos:
                feat_path = os.path.join(seed_dir, "{}_CLN_MTB_PRT_MIR_MET_feature_importance.tsv".format(algo))
                feat_df = pd.read_csv(feat_path, sep='\t', index_col=0)
                feat_df = feat_df.rename(columns={'feature_importance': "{}_{}".format(algo, seed.split('_')[-1])})
                if algo not in feat_results:
                    feat_results[algo] = []
                
                feat_results[algo].append(feat_df)

    all_feat_df = None
    for algo in feat_results.keys():
        print(algo)
        feat_df = pd.concat(feat_results[algo], axis=1)
        feat_df = feat_df[sorted(feat_df.columns, key=lambda x: int(re.search(r'(\d+)', x).group()))]
        
        feat_df['{}_mean'.format(algo)] = feat_df.mean(axis=1)
        feat_df['{}_std'.format(algo)] = feat_df.std(axis=1)
        feat_df = feat_df.sort_values(by='{}_mean'.format(algo), ascending=False)
        feat_df['{}_rank'.format(algo)] = range(1, len(feat_df) + 1)
        
        all_feat_df = pd.concat([all_feat_df, feat_df], axis=1)

    all_feat_df = all_feat_df.sort_values(by='avg_mean', axis=0, ascending=False)
    all_feat_df.to_csv(os.path.join(output_dir, 'table_feat.csv'), index=True)
                
if __name__ == '__main__':
    dir_name = './output_predictivity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=dir_name, help="parent output directory")
    parser.add_argument('--output_dir', type=str, default=dir_name, help="parent output directory")
    args = parser.parse_args()
    main(args)
