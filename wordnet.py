import csv
import os
from collections import defaultdict

class ViWordnet():
    def __init__(self, datadir):
        self.vocab = defaultdict(list)
        for csv_file in os.listdir(datadir):
            if not csv_file.endswith('csv'): continue
            if csv_file == 'meta.csv': continue 
            topic = csv_file.replace('.csv','')
            with open(os.path.join(datadir, csv_file)) as f:
                csv_reader = csv.reader(f, delimiter=',')
                for i, row in enumerate(csv_reader):
                    for word in row:
                        word = word.strip()
                        if len(word) == 0: continue
                        self.vocab[word].append({'topic':topic,'subtopic':str(i)})
        with open(os.path.join(datadir, 'meta.csv')) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for _, row in enumerate(csv_reader):
                for word in row[1:]:
                    word = word.strip()
                    if len(word) == 0: continue
                    self.vocab[word].append({'topic':row[0].strip()})


    def path(self, x1, x2):
        x1 = x1.lower().replace('_',' ')
        x2 = x2.lower().replace('_',' ')
        if x1 not in self.vocab or x2 not in self.vocab: return 0
        sim = [0]
        for data1 in self.vocab[x1]:
            for data2 in self.vocab[x2]:
                if data1['topic'] != data2['topic']: sim.append(0)
                else: ### Same topic 
                    if 'subtopic' not in data1 and 'subtopic' not in data2: ### Same parent topic 
                        sim.append(1)
                    elif 'subtopic' in data1 and 'subtopic' in data2:
                        if data1['subtopic'] != data2['subtopic']: sim.append(1/3.0)
                        else: sim.append(1)
                    else: sim.append(0.5)

        return max(sim)

if __name__ == '__main__':
    viwordnet =  ViWordnet('/home/trungdunghoang/Documents/NLP/vi-wordnet')
    import pdb;pdb.set_trace()