import gensim
import pandas as pd
class Keyword_Merge:
    def __init__(self,model_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        self.keyword = {}
    def merge(self,all_keyword,sentence_label):
        keyword = {}
        for i in range(len(all_keyword)):
            s1 = all_keyword[i] 
            if s1 == "":
                continue
            if s1 not in keyword:
                keyword[s1] = [] 
            for j in range(i+1,len(all_keyword)):
                s2 = all_keyword[j]
                if s2 == "":
                    continue
                try:
                    sim = self.model.similarity(s1,s2)
                    if(sim >= 0.5):
                        keyword[s1].append(s2)
                        all_keyword[j]=""
                except:
                    continue
        self.keyword = keyword
        for i in range(len(sentence_label)):
            if sentence_label[i] not in self.keyword:
                for key,values in self.keyword.items():
                    if sentence_label[i]  in values:
                        sentence_label[i]  = key
                        break
        return sentence_label