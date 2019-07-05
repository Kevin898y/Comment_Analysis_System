
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


# In[9]:


class Review_Sentiment:
    def __init__(self,tokenizer_path,model_path):
        self.tokenizer = pickle.load(open(tokenizer_path,'rb'))
        self.MAX_SEQUENCE_LENGTH = 1000
        self.model = load_model(model_path)
        self.Sentiment('Apple')
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
        df.to_csv('review_label.csv',index = False)
        
        return df


# In[ ]:


# rs = Review_Sentiment('model/tokenizer.p','model/tokenizer.p')

