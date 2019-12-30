
import pandas as pd
import numpy as np
from keras.models import  load_model
# from keras.layers import  CuDNNLSTM,Bidirectional, GlobalMaxPool1D,Input,Embedding,Dense, Activation, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle


# In[8]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.Session(config=config)
KTF.set_session(session)
# In[9]:


class Review_Sentiment:
    def __init__(self,tokenizer_path,model_path):
        self.tokenizer = pickle.load(open(tokenizer_path,'rb'))
        self.MAX_SEQUENCE_LENGTH = 1000
        self.model = load_model(model_path)
        self.dict = {}
        print(self.Sentiment('Apple'))

    def Sentiment(self,texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        seq = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
        return self.model.predict(seq)
    def to_csv(self,rs,review):
        raw_data= {'label':[],'comm':[]}
        for label,comm in zip(rs,review['comm']):
            raw_data['label'].append( int(label>0.5) )
            raw_data['comm'].append(comm)

        df = pd.DataFrame(raw_data, columns = ['label','comm'])
        Filter_advantage = df['label']==1
        Filter_disadvantage = df['label']==0
        disadvantage = df[Filter_disadvantage]
        advantage = df[Filter_advantage]
        df.to_csv('review_label.csv',index = False)
        disadvantage.to_csv('disadvantage.csv',index = False)
        advantage.to_csv('advantage.csv',index = False)
        self.data = df

        label = df['label']
        comm = df['comm']
        
        for i,j in zip(label,comm):
            self.dict[j]=i
        
        return raw_data


# In[ ]:


# rs = Review_Sentiment('model/tokenizer.p','model/tokenizer.p')

