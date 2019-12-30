import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam,RMSprop,Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os import listdir
from os.path import isfile, isdir, join
import pickle
import tensorflow as tf
import re
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from nltk.corpus import stopwords


# In[2]:


class Keyword_Extraction:
    def __init__(self,w2v_path,keyword_path,text_path):
        self.key_embedding = []
        self.keyword = pd.read_csv(keyword_path)
        self.comm_class ={}
        if text_path != '1':
            self.review = pd.read_csv(text_path)
        else:
            self.review = []
        self.texts = []
        self.evaluate ={}
        self.final_comm ={}
        self.model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,binary=True)
        self.cal_key_embedding()
    def remove_emoji(self,text):
        emoji_pattern = re.compile(
            "[\s+\.\!\/_,\-$%^*()+\"]+|[+——！，。？、~@#￥%……&*（）“”＆、‘’；：]+|"
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F917"
                               "]+"
           , flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text).strip()
    def set_comm_class(self):
        for i in self.keyword['keyword']:
            self.comm_class[i] = []
            self.final_comm[i] = []
    def cal_key_embedding(self):
        for word in self.keyword['keyword']:
            try:
                embedding_vector= self.model.get_vector(word.lower())
                self.key_embedding.append(embedding_vector)
            except:
                print("except"+word.lower())
        self.key_embedding = np.array(self.key_embedding)
    def set_texts(self):
        self.texts = []
        for label,comm in zip(self.review['label'],self.review['comm']) :
            temp = self.remove_emoji(str(comm).lower())
            self.texts.append( (label,temp) )
    def cal_sim(self):
        self.set_comm_class()
        self.set_texts()
        for comm in self.texts:
            comm_embedding = []
            for word in comm[1].split():
                try:
                    embedding_vector= self.model.get_vector(word)
                except:
                    embedding_vector = np.zeros(300)
                comm_embedding.append(embedding_vector)

            comm_embedding = np.array(comm_embedding) 
            dis = cosine_similarity(comm_embedding, self.key_embedding)
            max_sim = np.max(dis)
            if max_sim>0.5:
                temp = np.argmax(dis)
                # row = int(temp/21)
                col = temp%22
                try:
                    self.comm_class[ self.keyword['keyword'][col] ].append(comm)
                except:
                    print(comm_embedding.shape)
        self.top5()
    def top5(self):
        for key,comm in self.comm_class.items():
            neg =0
            pos =0
            comment =[]
            for i in comm:
                comment.append(i[1])
                if i[0]==0:
                    neg +=1
                else:
                    pos +=1
            self.final_comm[key].append(comment)
            if comm and pos>neg:
                self.evaluate[key]= 1
            elif comm:
                self.evaluate[key]= 0
            else:
                self.evaluate[key] = 'No Keyword'





        




# ke = Keyword_Extraction('wordtovector/GoogleNews-vectors-negative300.bin','data/keyword.csv','data/all_data.csv')
# ke.set_comm_class()
# ke.cal_key_embedding()``
# ke.set_texts()
# ke.cal_sim()
# ke.comm_class






