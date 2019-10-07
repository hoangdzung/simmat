from sklearn.metrics import roc_auc_score
import numpy as np 
from tqdm import tqdm

from algo import simmat
from DLCS import get_sim
import argparse
import sys 

parser = argparse.ArgumentParser()
parser.add_argument('--filename')
parser.add_argument('--use_sym',action="store_true")
parser.add_argument('--simmat',action="store_true")
parser.add_argument('--name')
args = parser.parse_args()

filename = args.filename

true_labels = []
pred_probs = []
for line in tqdm(open(filename).readlines()):
    if 'msr' in filename:
        label, _,_,sen1, sen2 = line.strip().split("\t")
        label = label[-1]
    elif 'parabank' in filename:
         _,_,_, sen1, sen2, label = line.strip().split('\t')
    # else:
    #     _,sen1, sen2, label = line.strip().split("\t")

    true_labels.append(int(label))
    
    if args.simmat:
        pred_probs.append(simmat(sen1, sen2))
    else:
        pred_probs.append(get_sim(sen1, sen2, args.use_sym))
        
print(roc_auc_score(np.array(true_labels), np.array(pred_probs)))

from sklearn.model_selection import  KFold
from sklearn.linear_model import LogisticRegression

X = np.expand_dims(pred_probs,-1)
y = np.array(true_labels)

kf = KFold(n_splits=5)
scores = []
for train_index, test_index in kf.split(X): 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

print(np.mean(np.array(scores)))

np.save(args.name+'.npy', np.array(pred_probs))
