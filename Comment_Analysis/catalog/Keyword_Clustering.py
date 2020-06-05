import json
import numpy as np
import pandas as pd
import pickle
#TODO:

class Keyword_Clustering:
    def __init__(self,data,set_path,Min_len=10):
        self.top_keyword = []
        self.data = data
        self.Min_len = Min_len
        with open(set_path, 'rb') as file:
            self.keyword_set =pickle.load(file)
    def set_top_keyword(self):
        sort_keyword = list(self.data.keyword.value_counts().index)
        self.top_keyword = []
        for i in sort_keyword:
            if i !='' and i==i:
                Filter = self.data ['keyword']==i
                filter_data = self.data[Filter]
                if len(filter_data)>self.Min_len or len(filter_data)>=len(self.data)/self.Min_len:
                    self.top_keyword.append(i)
    def clustering(self):
        self.set_top_keyword()
        cluster_label = []
        for keyword in self.top_keyword:
            Find = False
            for num,keywordset in enumerate(self.keyword_set):
                if keyword in keywordset:
                    cluster_label.append(num)
                    Find = True
            if not Find:
                cluster_label.append(-1)
                
        keyword_dict = {}
        index = {}
        for keyword,label in zip(self.top_keyword,cluster_label):
            if label == -1:
                keyword_dict[keyword]= []
            else:
                try:
                    temp = index[label]
                    keyword_dict[temp].append(keyword)
                except:  
                    keyword_dict[keyword]= []
                    index[label] = keyword
        return keyword_dict