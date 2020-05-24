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
        self.MAX_LEN = 100
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.data = pd.read_csv(data_path)
        self.size = []
        self.original_comm = []
    def to_ids(self,):
        inputs = [] 
        masks = []
        self.original_comm = []
        for sentence,original in zip(self.data['comm'],self.data['original_comm']):
            if (sentence == sentence):
                data = self.tokenizer.encode(sentence, add_special_tokens=True)
                if len(data) < self.MAX_LEN:
                    self.original_comm.append(original)
                    self.size.append(len(data))
                    data = pad_sequences([data], maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
                    attention_masks = [[float(i>0) for i in ii] for ii in data]
                    inputs.append(data)
                    masks.append(attention_masks)
        self.inputs= np.array(inputs).reshape(len(inputs),self.MAX_LEN)
        self.masks = np.array(masks).reshape(len(inputs),self.MAX_LEN)
        return self.inputs,self.masks
    def prediction(self,batch_size=300):
        inputs,masks = self.to_ids()
        self.model.cuda()
        self.model.eval()
        inputs = torch.tensor(inputs)
        masks = torch.tensor(masks)
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
    def convert_to_original(self, sentences):
        all_sen = []
        for sentence in sentences:
            r = []
            for index, token in enumerate(sentence):
                if token.startswith("##"):
                    if r:
                        r[-1] = f"{r[-1]}{token[2:]}"
                else:
                    r.append(token)
            all_sen.append(" ".join(r))

        return all_sen
    def split_sentence(self, all_sentence,all_pred,all_label):
        data = {'comm':[],'label':[],'original_comm':[]}
        for sentence,pred,sen_label,original in zip(all_sentence,all_pred,all_label,self.original_comm):
            split_sen = []
            sen = []
            index = 0
    
            for token,label in zip(sentence,pred) :
                if label == 'B-sent' and index!=0:
                    split_sen.append(sen)
                    sen = [token]
                else:
                    sen.append(token)
                index += 1
            split_sen.append(sen)
            for i in self.convert_to_original(split_sen):
                data['comm'].append(i)
                data['label'].append(sen_label)
                data['original_comm'].append(original)
        return data
    def to_csv(self,pred):#TODO:
        output = {'label':[],'comm':[]}
        id2tags = {0:'B-sent', 1:'O'}
        for sent_label,size,raw_data in zip(pred,self.size, self.inputs):
            label = []
            token = []
            for index in range(size-2):
                label.append(id2tags.get(int(sent_label[index+1]) ))
                token.append(raw_data[index+1])
                
            output['comm'].append(self.tokenizer.convert_ids_to_tokens(token)) 
            output['label'].append(label)
            
        # df = pd.DataFrame(output, columns = ['comm','label'])
        data = self.split_sentence(output['comm'],output['label'],self.data['label'])
        data = pd.DataFrame(data,columns = ['label','comm','original_comm'])
        #         df.to_csv('BERT_output.csv',index = False)

        return data

# In[7]:

# split = Bert_Split('split2.pkl','test_data.csv')
# p = split.prediction()





