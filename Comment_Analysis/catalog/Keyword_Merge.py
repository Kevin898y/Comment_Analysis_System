import numpy as np
import gensim
import pandas as pd
import collections
class Keyword_Merge:
    def __init__(self,model_path,original_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path) 
        self.original_model =  gensim.models.KeyedVectors.load_word2vec_format(original_path,binary=True)
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
                    sim2 = self.original_model.similarity(s1,s2)
                    if(sim >= 0.5 or sim2 >=0.6):
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
    def pre_process(self,adj_tuple):
        all_adj = []
        for i in adj_tuple:
            for adj in i:
                all_adj.append(adj)
        sort_adj = collections.Counter(all_adj).most_common()
        all_adj = [i[0] for i in sort_adj]
        
        sentence_adj = []
        for t in adj_tuple:
            temp = []
            for i in t:
                temp.append(i)
            sentence_adj.append(temp)
        return all_adj,sentence_adj    
        
    def adj_merge(self,adj_tuple):
        all_adj,sentence_adj = self.pre_process(adj_tuple)
        
        adj = {}
        for i in range(len(all_adj)):
            s1 = all_adj[i] 
            if s1 == "":
                continue
            if s1 not in adj:
                adj[s1] = [] 
            for j in range(i+1,len(all_adj)):
                s2 = all_adj[j]
                if s2 == "":
                    continue
                try:
                    sim = self.model.similarity(s1,s2)
                    sim2 = self.original_model.similarity(s1,s2)
                    if(sim >= 0.5 or sim2 >=0.6):
                        adj[s1].append(s2)
                        all_adj[j]=""
                except:
                    continue
        self.adj = adj
        for adj_list in sentence_adj:
            for i in range(len(adj_list)):
                if adj_list[i] not in adj:
                    for key,values in adj.items():
                        if adj_list[i]  in values:
                            adj_list[i]  = key
                            break
        adj_tuple = []
        for adj_list in sentence_adj:
            temp = []
            for adj in adj_list:
                temp.append(adj)
            adj_tuple.append(tuple(temp))
       
        return tuple(adj_tuple)