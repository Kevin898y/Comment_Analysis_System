# In[0]:
import torch
from pytorch_transformers import *
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
        self.model = torch.load(model_path,map_location='cuda:0')
        self.data = pd.read_csv(data_path)
        self.size = []
    def to_ids(self,):
        inputs = []
        masks = []
        for sentence in self.data['comm']:
            if (sentence == sentence):
                data = self.tokenizer.encode(sentence, add_special_tokens=True)
                if len(data) < self.MAX_LEN:
                    self.size.append(len(data))
                    data = pad_sequences([data], maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
                    attention_masks = [[float(i>0) for i in ii] for ii in data]
                    inputs.append(data)
                    masks.append(attention_masks)
        self.inputs= np.array(inputs).reshape(len(inputs),self.MAX_LEN)
        self.masks = np.array(masks).reshape(len(inputs),self.MAX_LEN)
    def prediction(self,batch_size=300):
        self.to_ids()
        self.model.eval()
        inputs = torch.tensor(self.inputs)
        masks = torch.tensor(self.masks)
        inputs = inputs.cuda()
        masks = masks.cuda()
        valid_data = TensorDataset(inputs, masks)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)
        
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
        keyword= {}
        adj = {}
        sentence_label = []
        for sentence_pred,size,raw_data in zip(pred,self.size, self.inputs):
            temp = {}
            token_label = ''#sentence_label
            sentence = raw_data[0:size]
            sentence,tags = self.convert_to_original(sentence,sentence_pred)
            sentence = self.lemmatize_sentence(sentence)
            
            for index in range(len(sentence)-2):
                label = id2tags.get(int(tags[index+1]))
                token = sentence[index+1]
                temp[token] = label
                
                if label == 'B-KEY' or label=='I-KEY':
                    if token in keyword:
                        keyword[token] += 1
                        token_label = token
                        break #假設只有一個關鍵字
                    else:
                        keyword[token] = 0
                        token_label = token
                        break #假設只有一個關鍵字
                    
                elif label == 'B-ADJ' or label == 'I-ADJ':
                    if token in adj:
                        adj[token] += 1
                    else:
                        adj[token] = 1
                        
            sentence_label.append(token_label)
            output.append(temp)
            
        
        keyword_top5 = sorted(keyword.items(), key=lambda d: d[1],reverse=True)[0:5]
        keyword_top5 = [i[0] for i in keyword_top5]
        keyword_top5,sentence_label= keyword_merge.merge(keyword_top5,sentence_label)
        keyword_top5 =[i for i in keyword_top5.keys()][0:5]
        # adj_top5 = sorted(adj.items(), key=lambda d: d[1], reverse=True)[0:5]
        # adj_top5 = [i[0] for i in adj_top5]

        self.output = output
        self.keyword_top5 = keyword_top5
        # self.adj_top5 = adj_top5
        self.label = sentence_label  

        test = {'label':[],'comm':[]}
        for label,comm in zip(self.label,self.data['comm']):
            test['label'].append(label)
            test['comm'].append(comm)
        self.sentence_label = pd.DataFrame(test, columns = ['label','comm'])  


        return output,keyword,adj,label
#%%

# ner = Bert_NER('model/NER3.pkl','data/test_data.csv')
# pred = ner.prediction()
# data = ner.to_csv(pred)




