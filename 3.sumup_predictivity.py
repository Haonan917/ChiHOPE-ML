import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import argparse
from pathlib import Path
import re
import os
import yaml
from sklearn.metrics import  auc, roc_auc_score

    
ENPOINT_PATTERN = "y.[\d]+"
BINARY_ENDPOINTS = ['y.90', 'y.158']


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def calculate_avg_predictions(valid_results, test_results, save_dir, algo):
    for omics in valid_results.keys():
        valid_pred_df_list = valid_results[omics]
        valid_avg_pred_df = sum(valid_pred_df_list) / len(valid_pred_df_list)
        valid_auc = roc_auc_score(valid_avg_pred_df["ytruth"], valid_avg_pred_df["ypred"])

        test_pred_df_list = test_results[omics]
        test_avg_pred_df = sum(test_pred_df_list) / len(test_pred_df_list)
        test_auc = roc_auc_score(test_avg_pred_df["ytruth"], test_avg_pred_df["ypred"])
        
        print(f"omics: {omics}, valid auc: {valid_auc}, test auc: {test_auc}")

        valid_avg_pred_df.to_csv(Path(save_dir) / 'integrated_valid_ypred_{}_{}.csv'.format(algo, omics), index=False)
        test_avg_pred_df.to_csv(Path(save_dir) / 'integrated_test_ypred_{}_{}.csv'.format(algo, omics), index=False)

        perfm_path = Path(save_dir) / 'ensemble_predictivity.csv'
        file_exists = os.path.exists(perfm_path)
        with open(perfm_path, 'a') as f:
            if not file_exists:
                f.write('data_type,algo,train.val,auc\n')
            
            f.write('{},{},{},{}\n'.format(omics,algo,1,valid_auc))
            f.write('{},{},{},{}\n'.format(omics,algo,0,test_auc))


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    endpoints = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    perfm_df = None
    for endpoint in endpoints:
        if not endpoint.startswith('y.'):
            continue
        print(f"endpoint: {endpoint}")

        endpoint_dir = os.path.join(output_dir, endpoint)
        seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
        for seed in seeds:
            seed_dir = os.path.join(endpoint_dir, seed)
            files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
            test_files = [os.path.join(seed_dir, f) for f in files if f.startswith('test_output')]
            valid_files = [os.path.join(seed_dir, f) for f in files if f.startswith('valid_output')]
            
            for valid_file, test_file in zip(valid_files, test_files):
                test_df = pd.read_csv(test_file, sep=',')
                valid_df = pd.read_csv(valid_file, sep=',')

                valid_selected = valid_df[['data_type', 'algo', 'seed', 'train.val', 'mean_auc']].rename(columns={'mean_auc': 'auc'})
                merged_df = pd.concat([test_df, valid_selected], ignore_index=True)
                perfm_df = pd.concat([perfm_df, merged_df], ignore_index=True)

    output_file_path = os.path.join(endpoint_dir, 'integrated_predictivity.csv')
    perfm_df.to_csv(output_file_path, index=False)

    summary_perfm_df = perfm_df.groupby(['data_type', 'algo', 'train.val']).agg(
        mean_auc=('auc', 'mean'),
        median_auc=('auc', 'median'),
        std_auc=('auc', 'std'),
    ).reset_index()
    summary_perfm_df.to_csv(os.path.join(output_dir, 'summarized_predictivity.csv'), index=False)


    algos = summary_perfm_df['algo'].unique().tolist()
    for algo in algos:
        # if algo != "avg":
        #     continue
        print(f"algo: {algo}")
        # calculate mean auc curve for ensemble_algorithms
        PATTERN = "valid_ypred_{}_(?P<omics_type>.*?).csv".format(algo)
        for endpoint in endpoints:
            if not endpoint.startswith('y.'):
                continue
            endpoint_dir = os.path.join(output_dir, endpoint)
            valid_results = {}
            test_results = {}
            seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
            for seed in seeds:
                seed_dir = os.path.join(endpoint_dir, seed)
                files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
                ypred_files = [f for f in files if f.startswith('valid_ypred_{}'.format(algo))]
                omics_list = list(set([re.match(PATTERN, f).group("omics_type") for f in ypred_files]))

                for omics in omics_list:
                    valid_ypred_path =  os.path.join(seed_dir, 'valid_ypred_{}_{}.csv'.format(algo, omics))
                    valid_ypred_df = pd.read_csv(valid_ypred_path, sep=",")

                    test_ypred_path =  os.path.join(seed_dir, 'test_ypred_{}_{}.csv'.format(algo, omics))
                    test_ypred_df = pd.read_csv(test_ypred_path, sep=",")
                    # fpr, tpr, thresholds = roc_curve(ypred_df["ytruth"], ypred_df['ypred'])

                    if omics not in valid_results:
                        valid_results[omics] = []
                    
                    if omics not in test_results:
                        test_results[omics] = []
                    # results[omics].append((fpr, tpr))
                    valid_results[omics].append(valid_ypred_df)
                    test_results[omics].append(test_ypred_df)

            calculate_avg_predictions(valid_results, test_results, endpoint_dir, algo)

                
if __name__ == '__main__':
    dir_name = './output_predictivity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=dir_name, help="parent output directory")
    parser.add_argument('--output_dir', type=str, default=dir_name, help="parent output directory")
    args = parser.parse_args()
    main(args)
