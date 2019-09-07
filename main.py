from sklearn.metrics import roc_auc_score
import numpy as np 

from algo import simmat
from DLCS import get_sim

import sys 

mode = sys.argv[0]
filename = sys.argv[1]
print(mode, filename)

true_labels = []
pred_probs = []
for line in open(filename):
    label, _,_,sen1, sen2 = line.strip().split("\t")
    true_labels.append(int(label))
    if mode == 'simmat':
        pred_probs.append(simmat(sen1, sen2))
    else:
        pred_probs.append(get_sim(sen1, sen2))
print(roc_auc_score(np.array(true_labels), np.array(pred_probs)))