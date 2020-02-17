# In[0]:
import torch
from transformers import *
import torch.nn.functional as F 
import pandas as pd
import numpy as np
torch.cuda.set_device(0)
from tqdm import tqdm, trange
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

keyword_merge = Keyword_Merge('model/booking_word2vec.model')


# In[1]:
class Bert_NER:  
    def __init__(self,model_path,data_path,MAX_LEN=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LEN = 100
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.data = pd.read_csv(data_path)
        self.size = []
        self.training_data = {'label':[],'sentence':[]}
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
    def prediction(self,batch_size=300):
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
    def to_CoNLL(self,pred):
        output = []
        id2tags = {0:'B-KEY', 1:'O'}
        keyword= {}
        adj = {}
        sentence_label = []
        sentiment_label = []
        review = pd.read_csv('ground_truth.csv', dtype={'comm': str})
        ground_truth = []
        #輸出文字是處理過的，可用self.training_data['sentence']取代
        for sentence_pred,size,raw_data,sentiment,truth in zip(pred, self.size, self.inputs,self.training_data['label'],review['label']):
            token_list = []
            comm_label = []#sentence_label
            #delete [ClS],[SEP]
            sentence = raw_data[1:size-1] 
            sentence_pred = sentence_pred[1:size-1]
            sentence,tags = self.convert_to_original(sentence,sentence_pred)
            sentence = self.lemmatize_sentence(sentence)

            for index in range(len(sentence)):
                label = id2tags.get(int(tags[index]))
                token = sentence[index]
                token_list.append((token,label))
                if label == 'B-KEY':
      
                    comm_label.append(token)
                    if token in keyword:
                        keyword[token] += 1
                    else:
                        keyword[token] = 0
                    
            #1NF(First normal form)
            for i in comm_label:
                ground_truth.append(truth)
                sentence_label.append(i)
                output.append(tuple(token_list))
                sentiment_label.append(sentiment)
#                 elif label == 'B-ADJ' or label == 'I-ADJ': #沒再用
#                     if token in adj:
#                         adj[token] += 1
#                     else:
#                         adj[token] = 1

        self.output = tuple(output)

        keyword_sorted = sorted(keyword.items(), key=lambda d: d[1],reverse=True)
        sentence_label = keyword_merge.merge(keyword_sorted, sentence_label)

        good_keyword= {}
        bad_keyword ={}

        for label,sentiment in zip(sentence_label,sentiment_label):
            if sentiment == 0:
                if label in bad_keyword:
                    bad_keyword[label] += 1
                else:
                    bad_keyword[label] = 0
            else:
                if label in good_keyword:
                    good_keyword[label] += 1
                else:
                    good_keyword[label] = 0


        good_keyword_top5 = sorted(good_keyword.items(), key=lambda d: d[1],reverse=True)
        self.good_keyword_top5 = [i[0] for i in good_keyword_top5]

    
        bad_keyword_top5 = sorted(bad_keyword.items(), key=lambda d: d[1],reverse=True)
        self.bad_keyword_top5 = [i[0] for i in bad_keyword_top5]

    
        # adj_top5 = sorted(adj.items(), key=lambda d: d[1], reverse=True)[0:5]
        # adj_top5 = [i[0] for i in adj_top5]

        
        # self.adj_top5 = adj_top5

        good = {'label':[],'comm':[]}
        bad = {'label':[],'comm':[]} 
        All = {'label':[],'comm':[],'sen_label':[]} #for test
        for label, comm, sentiment in zip(sentence_label, output, sentiment_label):
            All['label'].append(label)
            All['comm'].append(comm)
            All['sen_label'].append(sentiment)
            if sentiment == 0:
                bad['label'].append(label)
                bad['comm'].append(comm)
            else:
                good['label'].append(label)
                good['comm'].append(comm)
    
        good_sentence = pd.DataFrame(good, columns = ['label','comm'])
        bad_sentence = pd.DataFrame(bad, columns = ['label','comm'])   
        all_sentence = pd.DataFrame(All, columns = ['label','comm','sen_label']) 
         
        self.all_sentence = all_sentence.drop_duplicates()
        self.bad_sentence = bad_sentence.drop_duplicates()
        self.good_sentence = good_sentence.drop_duplicates()
        Filter_duplicated = ~good_sentence.duplicated() 
        self.ground_truth = list(np.array(ground_truth)[Filter_duplicated])
 
        self.bad_sentence.to_csv('bad_sentence.csv',index = False)
        self.all_sentence.to_csv('all_sentence.csv',index = False)
        
        self.good = list(zip( [1]*len(self.good_sentence),self.good_sentence['comm'].tolist()))
        self.bad = list(zip( [0]*len(self.bad_sentence),self.bad_sentence['comm'].tolist()))

        self.all = list(zip(sentiment_label,output))

        return self.all 
