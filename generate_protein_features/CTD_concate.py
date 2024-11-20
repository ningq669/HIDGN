
import pandas as pd 
import numpy as np 
import os 
import argparse

def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the data reading and train-test-splitting
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default=r"C:\Users\HP\Desktop\大学学习资料\信息生物\第一次论文\MULGA-master\Data\DrugBank\protein_features")
    parse.add_argument('-ctdc_path', '--ctdc_path_topofallfeature', type=str, default="CTDC_features.txt")
    parse.add_argument('-ctdt_path', '--ctdt_path_topofallfeature', type=str, default="CTDT_features.txt")
    parse.add_argument('-ctdd_path', '--ctdd_path_topofallfeature', type=str, default="CTDD_features.txt")
    parse.add_argument('-out', '--out_topofallfeature', type=str, default="CTD_features.txt")
    config = parse.parse_args()
    return config


def concate_ctd(root_path,ctdc_path,ctdt_path,ctdd_path,out):
    CTDC_path = os.path.join(root_path,ctdc_path)
    # CTDC = pd.read_csv(CTDC_path)
    CTDC = pd.read_csv(CTDC_path, header=None, sep="\t").values
    CTDC = CTDC[1:, 1:]
    CTDT_path = os.path.join(root_path,ctdt_path)
    # CTDT = pd.read_csv(CTDT_path)
    CTDT = pd.read_csv(CTDT_path, header=None, sep="\t").values
    CTDT = CTDT[1:, 1:]
    CTDD_path = os.path.join(root_path,ctdd_path)
    # CTDD = pd.read_csv(CTDD_path)
    CTDD = pd.read_csv(CTDD_path, header=None, sep="\t").values
    CTDD = CTDD[1:, 1:]
    CTD = np.concatenate([CTDC,CTDT,CTDD],axis=1)
    CTD = pd.DataFrame(CTD)
    out_path = os.path.join(root_path,out)
    CTD.to_csv(out_path)
    
    

if __name__ == "__main__": 
    config = get_config()
    root_path = config.root_path_topofallfeature
    ctdc_path = config.ctdc_path_topofallfeature
    ctdt_path = config.ctdt_path_topofallfeature
    ctdd_path = config.ctdd_path_topofallfeature 
    out = config.out_topofallfeature
    concate_ctd(root_path,ctdc_path,ctdt_path,ctdd_path,out)
    
    
