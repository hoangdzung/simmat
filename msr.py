from sklearn.metrics import roc_auc_score
import numpy as np 
from tqdm import tqdm

from algo import simmat
from DLCS import get_sim

import sys 


file_train = sys.argv[1]
file_test = sys.argv[2]
mode = sys.argv[3]

X_train = []
y_train = []
for line in tqdm(open(file_train).readlines()):
    label, _,_,sen1, sen2 = line.strip().split("\t")
    label = label[-1]
    y_train.append(int(label))
    
    if mode == 'simmat':
        X_train.append([simmat(sen1, sen2)])
    elif mode == 'use_sym':
        X_train.append([get_sim(sen1, sen2, True)])
    else:
        X_train.append([get_sim(sen1, sen2, False)])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for line in tqdm(open(file_test).readlines()):
    label, _,_,sen1, sen2 = line.strip().split("\t")
    label = label[-1]
    y_test.append(int(label))
    
    if mode == 'simmat':
        X_test.append([simmat(sen1, sen2)])
    elif mode == 'use_sym':
        X_test.append([get_sim(sen1, sen2, True)])
    else:
        X_test.append([get_sim(sen1, sen2, False)])
X_test = np.array(X_test)
y_test = np.array(y_test)

scores = []
for i in range(10):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

print(np.mean(np.array(scores)))

np.save('pred_'+mode+'.npy', np.array(pred_probs))
