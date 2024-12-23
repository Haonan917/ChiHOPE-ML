'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-11-05 08:52:36
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-11-18 00:56:38
FilePath: /CGZMain-Predictor/1.build_data_config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import yaml
from itertools import combinations
import os


parent_dir = './data/main_data/cg/'
save_root_dir = './config/cgz_main_config/cg/'
discovery_dir = 'Training'
validation_dir = 'Validation'

original_array = [
    'CLNData_CG.pkl',
    'MTBData_CG.pkl',
    'PRTData_CG.pkl',
    'MIRData_CG.pkl',
    'METData_CG.pkl',
]

all_combinations = []
for r in range(1, len(original_array) + 1):
    all_combinations.extend(combinations(original_array, r))

for combo in all_combinations:
    omics_list = list(combo)
    omics_name ='_'.join([omics[:3] for omics in omics_list])
    print(omics_name)
    data = {
        'omics_parent_dir': parent_dir,
        'discovery_dir_name': discovery_dir,
        'validation_dir_name': validation_dir,
        'data_type': omics_name,
        'omics_list': omics_list
    }
    save_dir = save_root_dir + 'select_{}/'.format(len(omics_list))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    yaml_file = omics_name + '.yaml'
    save_path = save_dir + yaml_file
    
    with open(save_path, 'w') as yaml_file:
         yaml.dump(data, yaml_file, default_flow_style=False)
    print(f'{omics_name} YAML file is saved at: {save_path}')







