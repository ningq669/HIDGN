import numpy as np
import pandas as pd

file_path = '../Data/KIBA/'
delimiter = ','

all_data = pd.read_excel(file_path+'targets.xlsx')
data = all_data.values


fasta_str = ""
for i in range(data.shape[0]):

    fasta_str += ">{}\n".format(data[i, 0])

    seq = data[i, 2]
    fasta_str += seq + "\n"

    with open(file_path+'fasta_seqs.txt', 'w') as f:
        f.write(fasta_str)