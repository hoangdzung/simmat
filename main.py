from sklearn.metrics import roc_auc_score
import numpy as np 

from algo import simmat

true_labels = []
pred_probs = []
for line in open('data/msr_paraphrase.txt'):
    label, _,_,sen1, sen2 = line.strip().split("\t")
    true_labels.append(int(label))
    pred_probs.append(simmat(sen1, sen2))

print(roc_auc_score(np.array(true_labels), np.array(pred_probs)))