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
from .Sentence_Similarity import Sentence_Simlarity

keyword_merge = Keyword_Merge('model/booking_word2vec.model')
# sentence_sim = Sentence_Simlarity()

# In[1]:
class Bert_NER:  
    def __init__(self,model_path,MAX_LEN=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LEN = 100
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.data=''
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
        id2tags = {0:'B-KEY', 1:'B-ADJ', 2:'O'}
        sentence_label = []
        review = pd.read_csv('ground_truth.csv', dtype={'comm': str})
        data = {'keyword':[],'adj':[],'sentiment':[],'sentence':[],'ground_truth':[]}

        #輸出文字是處理過的，可用self.training_data['sentence']取代
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
                data['keyword'].append(i)
                data['adj'].append(tuple(adj_label))
                data['sentiment'].append(sentiment)
                data['sentence'].append(tuple(token_list))
                data['ground_truth'].append(truth)

        data = pd.DataFrame(data, columns = ['keyword','adj','sentiment','sentence','ground_truth'])
        self.temp = data
        print(data.keyword.value_counts()[0:5])

        self.output = tuple(output)

        keyword_sorted = list(data.keyword.value_counts().index)
        sentence_label = keyword_merge.merge(keyword_sorted, data['keyword'])
        data['keyword'] = sentence_label
        
        Filter_Good = data['sentiment']==1
        Filter_Bad = data['sentiment'] == 0
        good_keyword= {}
        bad_keyword ={}
        self.good_keyword_top5 = list(data[Filter_Good].keyword.value_counts().index[0:5])
        self.bad_keyword_top5 = list(data[Filter_Bad].keyword.value_counts().index[0:5])

        self.good_sentence = data[Filter_Good].drop_duplicates()##same comm????
        self.bad_sentence = data[Filter_Bad].drop_duplicates()
        self.all_sentence = data.drop_duplicates() 
        self.ground_truth = self.all_sentence['ground_truth'].to_list()
        self.all_sentence.to_csv('all_sentence.csv',index = False)

#         #去除同句子不同label
        self.good = list(zip( [1]*len(self.good_sentence['sentence'].drop_duplicates()),self.good_sentence['sentence'].drop_duplicates().tolist()))
        self.bad = list(zip( [0]*len(self.bad_sentence['sentence'].drop_duplicates()),self.bad_sentence['sentence'].drop_duplicates().tolist()))
        self.all = list(zip(self.all_sentence['sentiment'].tolist(),self.all_sentence['sentence'].tolist()))

        return self.all
