# In[1]:
import torch
from transformers import *
torch.cuda.set_device(0)
import tensorflow as tf
import torch.nn.functional as F 
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# In[6]:

class Bert_Split:
    def __init__(self,model_path,data_path,MAX_LEN=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LEN = 129
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
                prediction = torch.max(F.softmax(outputs[0]),  dim=1)[1]
                prediction = prediction.cpu()
                pred.extend(prediction.numpy().tolist())
        return pred
    def split(self,pred):
        raw_data= {'label':[],'comm':[]}
        for p,sentence,label in zip(pred,self.data['comm'],self.data['label']):
            if sentence == sentence and len(sentence) >0:# not nan
                if p == 1:
                    s = sentence.split(',')
                    for i in s:
                        temp = str(i).strip()
                        if len(temp)>0:
                            raw_data['comm'].append(temp)
                            raw_data['label'].append(label)
                else:
                    raw_data['comm'].append(str(sentence))
                    raw_data['label'].append(label)
        df = pd.DataFrame(raw_data, columns = ['label','comm'])
        return df


# In[7]:

# split = Bert_Split('split2.pkl','test_data.csv')
# p = split.prediction()





