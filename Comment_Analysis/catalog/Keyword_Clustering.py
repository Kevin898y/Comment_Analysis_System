import json
import numpy as np
import pandas as pd
#TODO:
class Keyword_Clustering:
    def __init__(self,data,tree_path,Min_len=10):
        self.top_keyword = []
        self.data = data
        self.Min_len = Min_len
        with open(tree_path,'r',encoding = 'utf8') as f:
            self.keyword_tree = json.load(f)
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
        top = {}
        for keyword in self.top_keyword:
            Find = False
            if keyword in self.keyword_tree:
                Find = True
                if keyword not in top:
                    top[keyword] = []
            else:
                for key,values in self.keyword_tree.items():
                    if keyword in values:
                        Find = True
                        try:
                            top[key].append(keyword)
                        except:
                            top[key] = []
                            top[key].append(keyword)
            if not Find:
                top[keyword] = []
        return top