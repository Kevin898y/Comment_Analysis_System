import torch
from transformers import *
torch.cuda.set_device(0)
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences

#不應該有ground_truth
class Review_Sentiment:
    def __init__(self,model_path,MAX_LEN=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LEN = 100
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.data = []
        self.ground_truth = {'label':[],'comm':[]}
    def to_ids(self):
        inputs = [] 
        masks = []
        self.ground_truth['label'].clear()
        self.ground_truth['comm'].clear()
        for sentence,label in zip(self.data['comm'],self.data['label']):
            if (sentence == sentence):
                data = self.tokenizer.encode(sentence, add_special_tokens=True)
                if len(data) <= self.MAX_LEN:
                    self.ground_truth['label'].append(label)
                    self.ground_truth['comm'].append(sentence)

                    data = pad_sequences([data], maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
                    attention_masks = [[float(i>0) for i in ii] for ii in data]
                    inputs.append(data)
                    masks.append(attention_masks)
        self.inputs= np.array(inputs).reshape(len(inputs),self.MAX_LEN)
        self.masks = np.array(masks).reshape(len(inputs),self.MAX_LEN)
    def prediction(self,data,batch_size=300):
        self.data = data
        self.to_ids()
        self.model.cuda()
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
    def to_csv(self,pred,hotel_name):
        raw_data= {'label':[],'comm':[]}
        for label,comm in zip(pred,self.ground_truth['comm']):
            raw_data['label'].append( int(label>0.5) )
            raw_data['comm'].append(comm)

        df = pd.DataFrame(raw_data, columns = ['label','comm'])
        df.to_csv('cache/'+hotel_name+'_label.csv',index = False)

        #### for test
        df = pd.DataFrame(self.ground_truth, columns = ['label','comm'])
        df.to_csv('cache/'+hotel_name+'ground_truth.csv',index = False)

        return raw_data





