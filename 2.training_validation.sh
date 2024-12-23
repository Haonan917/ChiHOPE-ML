#!/bin/bash
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2024-11-24 23:51:33
 # @LastEditors: Haonan Chen haonanchen0815@163.com
 # @LastEditTime: 2024-12-23 01:58:27
 # @FilePath: /CGZMain-Predictor/2.training_validation.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
data_root_dir="./config/cgz_main_config/cg/"
output_dir="./output_predictivity/"

seed_start=1
seed_end=10
seed_interval=1

endpoint_name="y.90"

# 获取指定目录下的所有子目录名
directories=$(find "$data_root_dir" -maxdepth 1 -type d)
for dir in $directories; do
    # 跳过当前目录（"."）和上级目录（".."）
    if [ "$dir" != "$data_root_dir" ] && [ "$dir" != "$data_root_dir/.." ]; then
        # echo "Directory: $dir"
        files=$(find "$dir" -type f)
        for file in $files; do
            for ((seed=$seed_start; seed<=$seed_end; seed+=$seed_interval)); do
                echo "$endpoint_name-$file-$seed"
                python training_code/run_train.py \
                    --data_config_path $file \
                    --endpoint $endpoint_name \
                    --output_dir $output_dir \
                    --seed $seed \
                    --feature_selection_dir $feature_selection_dir \
                    --return_with_mismatch

                python validation_code/run_validation.py \
                    --data_config_path $file \
                    --endpoint $endpoint_name \
                    --output_dir $output_dir \
                    --seed $seed \
                    --feature_selection_dir $feature_selection_dir \
                    --return_with_mismatch

            done
        done
    fi
done