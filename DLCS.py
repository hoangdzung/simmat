import numpy as np
import math
from nltk.corpus import wordnet
from utils import getLemmas
from utils import path_similarity

def smoothFunct(x):
    return math.log1p(max([1, math.e + 0.5 -0.5*x]))

def is_synonyms(x, y):
    if x ==y:
        return True

    return False

def get_common_substring_matrix(strX, strY,use_sym):
    lenX = len(strX)
    lenY = len(strY)
    
    edit_matrix = [[0] * (lenY) for _ in range(lenX)]
    for i in range(lenX):
        for j in range(lenY):
            # if not is_synonyms(strX[i], strY[j]):
            # # if strX[i] != strY[j]: 
            #     continue
            if use_sym:
                sim = path_similarity(strX[i], strY[j])
                if i==0 or j ==0:
                    edit_matrix[i][j]=sim
                else:
                    edit_matrix[i][j]=sim+edit_matrix[i-1][j-1]
            else:
                if not is_synonyms(strX[i], strY[j]):
                # if strX[i] != strY[j]: 
                    continue
                elif i==0 or j ==0:
                    edit_matrix[i][j]=1
                else:
                    edit_matrix[i][j]=1+edit_matrix[i-1][j-1]
    return np.array(edit_matrix)

def getCommonSubString(srcSen, susSen, srcToken, susToken, use_sym=False):
    edit_matrix = get_common_substring_matrix(srcSen, susSen,use_sym)
    # print(edit_matrix)
    srcSenCopy = srcSen[:]
    susSenCopy = susSen[:]

    startSrcIndexs = []
    startSusIndexs = []

    while(edit_matrix.max()!=0):
        endSrcIndex = np.argmax(edit_matrix)//edit_matrix.shape[1]
        endSusIndex = np.argmax(edit_matrix)%edit_matrix.shape[1]
        commonLength = edit_matrix.max()

        startSrcIndex = endSrcIndex - commonLength + 1
        startSusIndex = endSusIndex - commonLength + 1

        startSrcIndexs.append(startSrcIndex)
        startSusIndexs.append(startSusIndex)
        edit_matrix[:, startSusIndex:endSusIndex+1] = 0
        edit_matrix[startSrcIndex:endSrcIndex+1, :] = 0

        for i in range(startSrcIndex+1,endSrcIndex+1):
            srcSenCopy[startSrcIndex]+='+'+srcSen[i]
            srcToken[startSrcIndex] += srcToken[i]

        for i in range(startSusIndex+1,endSusIndex+1):
            susSenCopy[startSusIndex]+='+'+susSen[i]
            susToken[startSusIndex] += susToken[i]

    squeezedSrcSen = []
    squeezedSusSen = []
    squeezedSrcToken = []
    squeezedSusToken = []
    for i in sorted(startSrcIndexs):
        squeezedSrcSen.append(srcSenCopy[i])
        squeezedSrcToken.append(srcToken[i])
    for i in sorted(startSusIndexs):
        squeezedSusSen.append(susSenCopy[i])
        squeezedSusToken.append(susToken[i])
    zipSqueezedIndexs = []
    for i, j in zip(startSrcIndexs, startSusIndexs):
        zipSqueezedIndexs.append((srcToken[i], susToken[j]))
    return squeezedSrcSen, squeezedSusSen, squeezedSrcToken, squeezedSusToken, zipSqueezedIndexs

def get_sim(srcSen, susSen, use_sym=False):
    srcSen, _ = getLemmas(srcSen)
    susSen, _ = getLemmas(susSen)
    originSrcToken = [[i] for i in range(len(srcSen))]
    originSusToken = [[i] for i in range(len(susSen))]

    result = getCommonSubString(srcSen, susSen, originSrcToken, originSusToken, use_sym)
    if len(result[-1]) ==0:
        return 0
    result = getCommonSubString(*(result[:-1]+[use_sym]))
    # print(result[:2])
    commonLength = 0
    for src, sus in result[-1]:
        assert len(src) == len(sus)
        subCommonLength = len(src)
        srcDistance = max(src) - min(src) 
        susDistance = max(sus) - min(sus)
        meanSpace = max([srcDistance, susDistance]) / (subCommonLength-1+1e-100)

        commonLength += smoothFunct(meanSpace) * subCommonLength

    return commonLength/ min(len(srcSen),len(susSen))
