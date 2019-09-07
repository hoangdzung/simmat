import numpy as np 
from scipy.optimize import linear_sum_assignment as lsa 
from utils import getLemmas, match, path_similarity

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
            sim_matrix[i][j] = path_similarity(w1, w2)

    row_ind, col_ind = lsa(-sim_matrix)
    return sim_matrix[row_ind, col_ind].tolist(), row_ind.tolist(), col_ind.tolist()
    
def relmat(N, M, alpha=0.2):
    numerator = sum([len(i) + len(i)**alpha for i in N])
    denominator = numerator

    numerator += sum([i**alpha for i in M])
    denominator += 2*len(M)

    return numerator/max(denominator,1e-20)

def p(rew1, rew2):
    p = abs(rew1 - rew2) / max(rew1, rew2)
    p = 0.5*(p**3)
    return  p 

def simmat(s1, s2, alpha=0.2):
    lemma1, pos1 = getLemmas(s1)
    lemma2, pos2 = getLemmas(s2)

    ### Phase 1
    P, considered1, considered2 = matchIdenticalPhrases(lemma1, lemma2)

    lemma1 = [lemma1[i] for i in range(len(lemma1)) if i not in considered1]
    lemma2 = [lemma2[i] for i in range(len(lemma2)) if i not in considered2]

    pos1 = [pos1[i] for i in range(len(pos1)) if i not in considered1]
    pos2 = [pos2[i] for i in range(len(pos2)) if i not in considered2]

    ### Phase 2 
    lemma1 = [lemma1[i] for i in range(len(lemma1)) if pos1[i] not in ['IN', 'PRP$', 'MD', '.']]
    lemma2 = [lemma2[i] for i in range(len(lemma2)) if pos2[i] not in ['IN', 'PRP$', 'MD', '.']]

    ### Phase 3 
    match_pairs, ind1, ind2 = matching(lemma1, lemma2)
    for i, j in zip(ind1, ind2):
        print(lemma1[i], lemma2[j])

    match_pairs = [i for i in match_pairs if i >0]

    ### Phase 4
    rel_mat = relmat(P, match_pairs, alpha)

    simmat = rel_mat * (1-p(len(lemma1), len(lemma2)))

    return simmat
    
if __name__ == '__main__':
    sen1 = "The study is being published today in the journal Science"
    sen2 = "Their findings were published today in Science."
    print(simmat(sen1, sen2))