from sklearn.metrics import roc_auc_score
import numpy as np 
from tqdm import tqdm

from algo import simmat
from DLCS import get_sim
from multiprocessing import Pool

import sys 

mode = sys.argv[2]
filename = sys.argv[1]
print(mode, filename)

true_labels = []
inputs = []
for line in tqdm(open(filename).readlines()):
    if filename.endswith('txt') and filename.startswith('msr'):
        label, _,_,sen1, sen2 = line.strip().split("\t")
        label = label[-1]
    elif filename.endswith('tsv'):
        if filename.startswith('quora') or 'parabank' in filename:
            _,_,_, sen1, sen2, label = line.strip().split('\t')
        else:
            _,sen1, sen2, label = line.strip().split("\t")
    inputs.append((sen1, sen2))
    true_labels.append(int(label))

def worker1(x):
    return simmat(x[0],x[1])

def worker2(x):
    return get_sim(x[0],x[1])

if __name__ == '__main__':
    p = Pool(12)
    if mode == 'simmat':
        pred_probs = p.map(worker1, inputs)
    else:
        pred_probs = p.map(worker2, inputs)
    print(roc_auc_score(np.array(true_labels), np.array(pred_probs)))
