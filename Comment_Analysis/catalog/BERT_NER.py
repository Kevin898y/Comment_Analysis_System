# In[0]:
import torch
from transformers import *
import torch.nn.functional as F 
import pandas as pd
import numpy as np
torch.cuda.set_device(0)
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import join
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import math
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from .Keyword_Merge import Keyword_Merge
from .Sentence_Similarity import Sentence_Simlarity
from .Keyword_Clustering import Keyword_Clustering
import json
import pandas as pd

keyword_merge = Keyword_Merge('model/booking_word2vec3.model',"model/GoogleNews-vectors-negative300.bin")
keyword_clustering = Keyword_Clustering([],'model/merge.json')
# sentence_sim = Sentence_Simlarity()
wordnet.ensure_loaded() 
# In[1]:
class Bert_NER:  
    def __init__(self,model_path,MAX_LEN=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LEN = 100
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.data=''
        self.size = []#
        self.training_data = {'label':[],'sentence':[]}#
        self.inputs = ''#
    def to_ids(self,):
        inputs = []
        masks = []
        self.training_data['label'].clear()
        self.training_data['sentence'].clear()
        self.size.clear()
        for label,sentence in zip(self.data['label'],self.data['comm']):
            if (sentence == sentence):#pd.dropna() 
                data = self.tokenizer.encode(sentence, add_special_tokens=True)
                if len(data) <= self.MAX_LEN:
                    self.training_data['label'].append(label)
                    self.training_data['sentence'].append(sentence)
                
                    self.size.append(len(data))
                    data = pad_sequences([data], maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
                    attention_masks = [[float(i>0) for i in ii] for ii in data]
                    inputs.append(data)
                    masks.append(attention_masks)
        self.inputs= np.array(inputs).reshape(len(inputs),self.MAX_LEN)
        self.masks = np.array(masks).reshape(len(inputs),self.MAX_LEN)
    def prediction(self,hotel_name,batch_size=300):
        self.to_ids()
        self.model.cuda()
        self.model.eval()
        inputs = torch.tensor(self.inputs)
        masks = torch.tensor(self.masks)
        inputs = inputs.cuda()
        masks = masks.cuda()
        valid_data = TensorDataset(inputs, masks)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data,  shuffle=False, sampler=valid_sampler, batch_size=batch_size)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
                prediction = torch.max(F.softmax(outputs[0],dim=2), 2)[1]
                prediction = prediction.cpu()
                pred.extend(prediction.numpy().tolist())

        ## for cache
        # temp = np.array(pred)
        # np.save('cache/'+hotel_name+"_pred.npy", temp)
        # temp = np.array(self.size)
        # np.save(hotel_name+"_size.npy", temp)
        # np.save(hotel_name+"_inputs.npy", self.inputs)
        # with open(hotel_name+'training_data.json', 'w') as f:
        #     json.dump(self.training_data, f)


        return pred
    def to_csv(self,pred):
        output = {'sentence':[],'label':[]}
        id2tags = {0:'B-KEY', 1:'B-ADJ', 2:'O'}
        for sentence,size,raw_data in zip(pred,self.size, self.inputs):
            label = []
            token = []
            for index in range(size-2):
                label.append(id2tags.get(int(sentence[index+1]) ))
                token.append(raw_data[index+1])
                
            output['sentence'].append(self.tokenizer.convert_ids_to_tokens(token)) 
            output['label'].append(label)
        
        df = pd.DataFrame(output, columns = ['sentence','label'])
        df.to_csv('BERT_output.csv',index = False)
        return df
    def convert_to_original(self,sentence,tags):
        sentence=self.tokenizer.convert_ids_to_tokens(sentence)
        r = []
        r_tags = []
        for index, token in enumerate(sentence):
            if token.startswith("##"):
                if r:
                    r[-1] = f"{r[-1]}{token[2:]}"
            else:
                r.append(token)
                r_tags.append(tags[index])
        return r,r_tags
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    def lemmatize_sentence(self,sentence):
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(sentence):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

        return res
    def to_CoNLL(self,pred,hotel_name):
        id2tags = {0:'B-KEY', 1:'B-ADJ', 2:'O'}
        sentence_label = []
        review = pd.read_csv('cache/'+hotel_name+'ground_truth.csv', dtype={'comm': str})
        data = {'uid':[],'keyword':[],'adj':[],'sentiment':[],'sentence':[],'ground_truth':[]}
        if os.path.isfile('cache/'+hotel_name+'_to_CoNLL.csv'):
            data = pd.read_csv('cache/'+hotel_name+'_to_CoNLL.csv')
            data['adj'] = data['adj'].map(lambda x: eval(x))
            data['sentence'] = data['sentence'].map(lambda x: eval(x))
        else:
            #輸出文字是處理過的，可用self.training_data['sentence']取代
            uid = 0
            for sentence_pred,size,raw_data,sentiment,truth in zip(pred, self.size, self.inputs,self.training_data['label'],review['label']):
                
                token_list = []
                comm_label = []#sentence_label
                adj_label = []
                #delete [ClS],[SEP]
                sentence = raw_data[1:size-1] 
                sentence_pred = sentence_pred[1:size-1]
                sentence,tags = self.convert_to_original(sentence,sentence_pred)
                sentence = self.lemmatize_sentence(sentence) ###????

                for index in range(len(sentence)):
                    label = id2tags.get(int(tags[index]))
                    token = sentence[index]
                    token_list.append((token,label))
                    if label == 'B-KEY':
                        comm_label.append(token)
                        
                    elif label == 'B-ADJ' : 
                        adj_label.append(token)

                #1NF(First normal form)
                for i in comm_label:
                    data['uid'].append(uid)
                    data['keyword'].append(i)
                    data['adj'].append(tuple(adj_label))
                    data['sentiment'].append(sentiment)
                    data['sentence'].append(tuple(token_list))
                    data['ground_truth'].append(truth)
                if len(comm_label) == 0:
                    data['uid'].append(uid)
                    data['keyword'].append('')
                    data['adj'].append(tuple(adj_label))
                    data['sentiment'].append(sentiment)
                    data['sentence'].append(tuple(token_list))
                    data['ground_truth'].append(truth)
                uid+=1
                
            data = pd.DataFrame(data, columns = ['uid','keyword','adj','sentiment','sentence','ground_truth'])
            keyword_sorted = list(data.keyword.value_counts().index)
            sentence_label = keyword_merge.merge(keyword_sorted, data['keyword'])
            data['keyword'] = sentence_label
            data['adj'] = keyword_merge.adj_merge(data['adj'])
            data = data.drop_duplicates()
            data.to_csv('cache/'+hotel_name+'_to_CoNLL.csv',index = False)

        Filter_Good = data['sentiment']==1
        Filter_Bad = data['sentiment'] == 0

        good_sentence = data[Filter_Good]
        bad_sentence = data[Filter_Bad] 

        keyword_clustering.data = good_sentence
        top_good = keyword_clustering.clustering()
        with open('cache/top_good.json', 'w') as f:
            json.dump(top_good, f)

        keyword_clustering.data = bad_sentence
        top_bad = keyword_clustering.clustering()
        with open('cache/top_bad.json', 'w') as f:
            json.dump(top_bad, f)


        # self.good_keyword_top5 = list(data[Filter_Good].keyword.value_counts().index)
        # self.bad_keyword_top5 = list(data[Filter_Bad].keyword.value_counts().index)
        # good_keyword_top5 = []
        # for i in self.good_keyword_top5:
        #     if i !='':
        #         Filter = self.good_sentence ['keyword']==i
        #         clean_data = self.good_sentence[Filter]
        #         if len(clean_data)>10 or len(clean_data)>=len(self.good_keyword_top5)/10:
        #             good_keyword_top5.append(i)
        # temp = np.array(good_keyword_top5)
        # np.save('cache/'+hotel_name+"_good_keyword_top5.npy", temp)       

        # bad_keyword_top5 = []
        # for i in self.bad_keyword_top5:
        #     if i != '':
        #         Filter = self.bad_sentence ['keyword']==i
        #         clean_data = self.bad_sentence[Filter]
        #         if len(clean_data)>10 or len(clean_data)>len(self.bad_keyword_top5)/10:
        #             bad_keyword_top5.append(i)
        # temp = np.array(bad_keyword_top5)
        # np.save('cache/'+hotel_name+"_bad_keyword_top5.npy", temp)       

