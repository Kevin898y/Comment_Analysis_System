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
        output = sentence
        g  = [i[0] for i in output]
        return [" ".join(g)]
    def Preporcessiong(self):
        unlabel = {'index':[],'comm':[],'label':[]}
        labeled = {'comm':[],'label':[]}

        for l,comm,index in zip(self.data['label'],self.data['comm'],range(len(self.data['comm']))):
            if l != l:
                unlabel['index'].append(index)
                unlabel['comm'].append(self.get_sentence(comm))
                unlabel['label'].append('')
            else:
                labeled['comm'].append(self.get_sentence(comm))
                labeled['label'].append(l)
        return unlabel,labeled
    def similarity(self):
        unlabel,labeled = self.Preporcessiong()
        for unlabel_comm,index in zip(unlabel['comm'],range(len(unlabel['label']))):
            embedding_i = self.model.encode(unlabel_comm)
            max_sim = 0 
            sim_label = ''
            for label_comm,l in zip(labeled['comm'],labeled['label']):
                embedding_j = self.model.encode(label_comm)
                sim = cosine_similarity(embedding_i,embedding_j)
                if sim > max_sim: ##相同句子有多個label
                    max_sim = sim
                    sim_label = l
            if max_sim > 0.5:
                unlabel['label'][index] = sim_label
        return unlabel      