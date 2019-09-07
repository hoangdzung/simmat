import stanfordnlp 
from nltk.corpus import wordnet
nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos")


def getLemmas(sentence):
    doc = nlp(sentence)
    return [word.lemma for word in doc.sentences[0].words], [word.xpos for word in doc.sentences[0].words] 

def path_similarity(w1, w2):
    sims = [0]
    for i in wordnet.synsets(w1):
        for j in wordnet.synsets(w2):
            sims.append(i.path_similarity(j))

    return max([i for i in sims if i is not None])

def match(L1, L2, i, j):
    end1 = i 
    end2 = j
    while (end1 < len(L1) and end2 < len(L2) and L1[end1]==L2[end2]):
        end1+=1
        end2+=1 

    return tuple(L1[i:end1])
