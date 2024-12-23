'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-11-06 03:31:41
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-11-24 23:54:16
FilePath: /CGZMain-Predictor/3.sumup_predictivity.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './biomarker_code/')))
import pandas as pd
import argparse
from pathlib import Path
import re
import os
import yaml
import biomarker_code
from sklearn.metrics import  auc, roc_auc_score
import pickle
    
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

        # valid_avg_pred_df.to_csv(Path(save_dir) / 'integrated_valid_ypred_avg_{}.csv'.format(omics), index=False)
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

    # calculate mean auc curve for ensemble_algorithms
    PATTERN = r"output_refits_(?P<omics_type>.*?).pkl"
    for endpoint in endpoints:
        if not endpoint.startswith('y.'):
            continue
        endpoint_dir = os.path.join(output_dir, endpoint)
        valid_results = {}
        test_results = {}
        seeds = [d for d in os.listdir(endpoint_dir) if os.path.isdir(os.path.join(endpoint_dir, d))]
        
        all_config_df = None
        for seed in seeds:
            seed_dir = os.path.join(endpoint_dir, seed)
            files =  [f for f in os.listdir(seed_dir) if os.path.isfile(os.path.join(seed_dir, f))]
            model_files = [f for f in files if f.startswith('output_refits')]
            omics_list = list(set([re.match(PATTERN, f).group("omics_type") for f in model_files]))

            for omics in omics_list:
                model_path = Path(os.path.join(seed_dir, 'output_refits_{}.pkl'.format(omics)))
                refit_models = pickle.load(open(model_path, 'rb'))
                config_dict = {}
                config_dict["Seed"] = int(seed.split('_')[-1])
                config_dict["Modality"] = omics
                for algo in refit_models.keys():
                    if algo == 'avg':
                        continue

                    pipeline = refit_models[algo][0]
                    model = pipeline.named_steps[algo]
                    params = model.get_params()

                    if algo == "svc":
                        config_dict["svm.C"]= params["C"]

                    elif algo == "lr":
                        config_dict["lr.l1_ratio"]= params["l1_ratio"]
                        config_dict["lr.C"] = params["C"]
                    elif algo == "xgb":
                        config_dict["xgb.subsample"]= params["subsample"]
                        config_dict["xgb.learning_rate"] = params["learning_rate"]
                        config_dict["xgb.n_estimators"]= params["n_estimators"]
                        config_dict["xgb.max_depth"] = params["max_depth"]
                        config_dict["xgb.colsample_bytree"] = params["colsample_bytree"]

 
                config_df = pd.DataFrame(config_dict, index=[0])
                all_config_df = pd.concat([all_config_df, config_df], ignore_index=True)

        sorted_df = all_config_df.sort_values(by=['Modality', 'Seed'], ascending=[True, True])
        sorted_df.to_csv(os.path.join(output_dir, 'table_config.csv'), index=False)

                
if __name__ == '__main__':
    dir_name = './output_predictivity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=dir_name, help="parent output directory")
    parser.add_argument('--output_dir', type=str, default=dir_name, help="parent output directory")
    args = parser.parse_args()
    main(args)
