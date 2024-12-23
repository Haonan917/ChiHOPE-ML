import pandas as pd
from pathlib import Path
import time
if __name__ == "__main__":
    data_dir = "data/main_data/cg/"
    train_dir = "Training"
    test_dir = "Validation"
    
    data_dir_path = Path(data_dir)
    discovery_dir_path = data_dir_path / train_dir
    validation_dir_path = data_dir_path / test_dir
    
    omics_file_list = ["CLNData_CG", "MTBData_CG", "PRTData_CG", "MIRData_CG", "METData_CG"]
    from_suffix = ".csv"
    to_suffix = ".pkl"
    
    
    s = time.time()
    for omics_file in omics_file_list:
        all_train_file  = discovery_dir_path / (omics_file + from_suffix)
        all_validation_file  = validation_dir_path / (omics_file + from_suffix)

        # read csv
        all_train_df = pd.read_csv(all_train_file, sep="," ,index_col=0) # (feature, sample)
        all_validation_df = pd.read_csv(all_validation_file, sep=",",index_col=0) # (feature, sample)

        # export pkl
        all_train_df.to_pickle(discovery_dir_path / (omics_file + to_suffix))
        all_validation_df.to_pickle(validation_dir_path / (omics_file + to_suffix))

    e = time.time()
    print(f"Time to run：{e-s} seconds")
    print("Finish converting")