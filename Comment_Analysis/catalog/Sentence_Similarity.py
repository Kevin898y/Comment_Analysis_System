from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
class Sentence_Simlarity:  
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.data = ""
    def sentence_similarity(self,model,sentences):
        sentence_embeddings = model.encode(sentences)
        sim = cosine_similarity(sentence_embeddings[0].reshape(1, -1),sentence_embeddings[1].reshape(1, -1))
        return sim
    def get_sentence(self,sentence):
        output = eval(sentence)
        g  = [i[0] for i in output]
        return [" ".join(g)]
    def Preporcessiong(self):
        unlabel = {'ner':[],'comm':[],'label':[]}
        labeled = {'ner':[],'comm':[],'label':[]}

        for l,comm in zip(self.data['label'],self.data['comm']):
            if l != l:
                unlabel['ner'].append(comm)
                unlabel['comm'].append(self.get_sentence(comm))
                unlabel['label'].append('')
            else:
                labeled['ner'].append(comm)
                labeled['comm'].append(self.get_sentence(comm))
                labeled['label'].append(l)
        return unlabel,labeled
    def similarity(self):
        unlabel,labeled = self.Preporcessiong()
        output = []
        for unlabel_comm,index in zip(unlabel['comm'],range(len(unlabel['label']))):
            embedding_i = self.model.encode(unlabel_comm)
            max_sim = 0 
            sim_label = ''
            for label_comm,l in zip(labeled['comm'],labeled['label']):
                embedding_j = self.model.encode(label_comm)
                sim = cosine_similarity(embedding_i,embedding_j)
                if sim > max_sim:
                    max_sim = sim
                    sim_label = l
            if max_sim > 0.5:
                unlabel['label'][index] = sim_label
        unlabel = pd.DataFrame(unlabel, columns = ['ner','comm','label'])
        labeled = pd.DataFrame(labeled, columns = ['ner','comm','label'])
        return pd.concat([unlabel, labeled], axis=0)   