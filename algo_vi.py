import numpy as np 
from scipy.optimize import linear_sum_assignment as lsa 
from wordnet import ViWordnet

viwordnet = ViWordnet('/home/trungdunghoang/Documents/NLP/vi-wordnet')
stopwords = set([line.strip().replace(' ','_') for line in open('stopwords.words')])

def match(L1, L2, i, j):
    end1 = i 
    end2 = j
    while (end1 < len(L1) and end2 < len(L2) and L1[end1]==L2[end2]):
        end1+=1
        end2+=1 

    return tuple(L1[i:end1])

def segment(text):
    text = text.lower().split()

    tokens = []
    i = 0
    while i < len(text):
        maxWord = []
        for j in range(i, len(text)):
            tempWord = text[i:j+1]
            if ' '.join(tempWord) in viwordnet.vocab and len(tempWord) > len(maxWord):
                maxWord = tempWord
        i = i+len(maxWord)
        tokens.append(maxWord)
    text = ['_'.join(i) for i in tokens]

    return text

def matchIdenticalPhrases(L1, L2):
    P = set()
    considered1 = set()
    considered2 = set()
    while(True):
        new = ''
        pos = None 
        for i in range(len(L1)):
            for j in range(len(L2)):
                if i not in considered1 and j not in considered2:
                    temp = match(L1, L2, i, j)
                    if len(temp) > len(new):
                        new = temp
                        pos = (i,j)
        if len(new) > 0:
            P.add(new)
            considered1 = considered1.union(set(range(pos[0],pos[0]+len(new))))
            considered2 = considered2.union(set(range(pos[1],pos[1]+len(new))))
            # print(new, considered1,considered2)
        else:
            break
    
    return P, considered1, considered2

def matching(L1, L2):
    sim_matrix = np.zeros((len(L1), len(L2)))
    for i, w1 in enumerate(L1):
        for j, w2 in enumerate(L2):
            sim_matrix[i][j] = viwordnet.path(w1, w2)

    row_ind, col_ind = lsa(-sim_matrix)
    return sim_matrix[row_ind, col_ind].tolist(), row_ind.tolist(), col_ind.tolist()
    
def relmat(N, M, alpha=0.2):
    numerator = sum([len(i) + len(i)**alpha for i in N])
    denominator = numerator

    numerator += sum([i**alpha for i in M])
    denominator += 2*len(M)

    return numerator/max(denominator,1e-20)

def simmat(s1, s2, alpha=0.2):
    ### Phase 1
    seg1 = segment(s1)
    seg2 = segment(s2)
    ### Phase 2
    P, considered1, considered2 = matchIdenticalPhrases(seg1, seg2)
    
    seg1 = [seg1[i] for i in range(len(seg1)) if i not in considered1]
    seg2 = [seg2[i] for i in range(len(seg2)) if i not in considered2]
    ### Phase 3 
    seg1 = [i for i in seg1 if i not in stopwords]
    seg2 = [i for i in seg2 if i not in stopwords]
    ### Phase 4 
    match_pairs, _, _ = matching(seg1, seg2)
    match_pairs = [i for i in match_pairs if i >0]
    ### Phase 5
    simmat = relmat(P, match_pairs, alpha)
    return simmat
    
if __name__ == '__main__':
    import sys
    import string
    from tqdm import tqdm 

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file_label = sys.argv[3]

    sens1 = [line.strip() for line in open(file1)]
    sens2 = [line.strip() for line in open(file2)]
    labels = [int(i.strip()) for i in open(file_label)]

    preds = []

    for sen1, sen2 in tqdm(zip(sens1,sens2)):
        sen1 = sen1.translate(str.maketrans("","", string.punctuation))
        sen2 = sen2.translate(str.maketrans("","", string.punctuation) )

        preds.append(simmat(sen1, sen2))   

    import numpy as np 
    np.save('gt.npy', np.array(labels))
    np.save('pred.npy', np.array(preds))
