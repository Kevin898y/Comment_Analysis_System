import gensim
import pandas as pd
class Keyword_Merge:
    def __init__(self,model_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        self.keyword = {}
    def merge(self,data,sentence_label,sentiment_label,evaluation):
        keyword = {}
        for i in range(len(data)):
            s1 = data[i] 
            if s1 == "":
                continue
            if s1 not in keyword:
                keyword[s1] = [] 
            for j in range(i+1,len(data)):
                s2 = data[j]
                if s2 == "":
                    continue
                try:
                    sim = self.model.similarity(s1,s2)
                    if(sim >= 0.5):
                        keyword[s1].append(s2)
                        data[j]=""
                except:
                    continue
        self.keyword = keyword
        for i,label in zip(range(len(sentence_label)),sentiment_label):
            if label == evaluation:
                if sentence_label[i] not in self.keyword:
                    for key,values in self.keyword.items():
                        if sentence_label[i]  in values:
                            sentence_label[i]  = key
                            break
        return keyword,sentence_label    