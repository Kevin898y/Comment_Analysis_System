from .Booking_crawler import Booking_crawler
from .Review_Sentiment import Review_Sentiment
from .Keyword_Extraction import Keyword_Extraction
from .BERT_NER import Bert_NER
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
# import pke
# from nltk.corpus import stopwords
import numpy as np
from catalog.forms import SearchForm
from catalog.forms import CheckForm
from catalog.forms import UploadFileForm
# from pip._vendor.html5lib.filters.sanitizer import Filter
import os
import json

# ke = Keyword_Extraction('wordtovector/GoogleNews-vectors-negative300.bin','data/keyword.csv','1')
# ner = Bert_NER('model/NER3/')
ner = Bert_NER('model/NER5/') 
sen = Review_Sentiment('model/sentiment/')
filename= '' 


def Search(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['booking_url']
            crawler = Booking_crawler(url)
            isfile = crawler.Scrapy_Review()
            s = []
            if isfile:
                hotel_name = crawler.pagename
                review = pd.read_csv('cache/'+hotel_name+'.csv')
                s = review['comm']
            else:
                review,hotel_name =crawler.ToCSV()
                s = review['comm']
            context = {
                's': s,
                'hotel_name':hotel_name,
            }
            return render(request,'simple_crawl.html',context)
    else:
        form = SearchForm(initial={'booking_url': ''})

    context = {
        'form': form,
    }
    return render(request,'home.html', context)
    
def top_keyowrd(request,hotel_name, id, keyowrd): #to do
    hotel_name = hotel_name
    data = pd.read_csv('cache/'+hotel_name+'_to_CoNLL.csv')
    data['adj'] = data['adj'].map(lambda x: eval(x))
    data['sentence'] = data['sentence'].map(lambda x: eval(x))

    Filter = ~data['uid'].duplicated() ##UID
    sentence = data[Filter]

    all_sentence = data
    Filter_Good = all_sentence['sentiment']==1
    Filter_Bad = all_sentence['sentiment'] == 0

    good_num = np.load('cache/'+hotel_name+'good_num.npy',allow_pickle = True)
    good_center = np.load('cache/'+hotel_name+'good_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_good.json',encoding = 'utf8') as f:
        top_good = json.load(f)
    bad_num = np.load('cache/'+hotel_name+'bad_num.npy',allow_pickle = True)
    bad_center = np.load('cache/'+hotel_name+'bad_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_bad.json',encoding = 'utf8') as f:
        top_bad = json.load(f)

    if id >len(top_good):
        Filter = all_sentence[Filter_Bad] ['keyword']==keyowrd
        clean_data = all_sentence[Filter_Bad][Filter]
        data = list(zip(clean_data['ground_truth'], 
                        [0]*len(clean_data['keyword']),
                        clean_data['sentence'],
                        clean_data['original'],
                        clean_data["cluster"] ))
        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:####
                adj_top.append(i[0])
        adjtop = adj_top
        
    else :
        Filter = all_sentence[Filter_Good] ['keyword']==keyowrd
        clean_data = all_sentence[Filter_Good][Filter]
        data = list(zip(clean_data['ground_truth'],
                        [1]*len(clean_data['keyword']),
                        clean_data['sentence'],
                        clean_data['original'],
                        clean_data["cluster"] ))
        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:###
                adj_top.append(i[0])
        adjtop = adj_top
     
    context = {
        'all_num':len(sentence),
        'advantage_num':len(sentence[sentence['sentiment']== 1]),
        'disadvantage_num':len(sentence[sentence['sentiment']== 0]),
        'top_good': top_good,
        'top_bad': top_bad,
        'good_num':good_num,
        'good_center':good_center,
        'bad_num':bad_num,
        'bad_center':bad_center,
        'good_keyword_size':len(top_good),
        'bad_keyword_size': len(top_bad)+1,
        'adj_top':adjtop,
        'data':data,
        'id':id,
        'hotel_name':hotel_name,
        'keyowrd':keyowrd,
    } 

    return render(request,'NER.html',context)

def top_adj(request,hotel_name,id,keyowrd,adj_num ): #to do
    temp = id
    hotel_name = hotel_name
    data = pd.read_csv('cache/'+hotel_name+'_to_CoNLL.csv')
    data['adj'] = data['adj'].map(lambda x: eval(x))
    data['sentence'] = data['sentence'].map(lambda x: eval(x))
    
    Filter = ~data['uid'].duplicated() ##UID
    sentence = data[Filter]
    
    all_sentence = data
    Filter_Good = all_sentence['sentiment']==1
    Filter_Bad = all_sentence['sentiment'] == 0

    good_num = np.load('cache/'+hotel_name+'good_num.npy',allow_pickle = True)
    good_center = np.load('cache/'+hotel_name+'good_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_good.json',encoding = 'utf8') as f:
        top_good = json.load(f)
    bad_num = np.load('cache/'+hotel_name+'bad_num.npy',allow_pickle = True)
    bad_center = np.load('cache/'+hotel_name+'bad_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_bad.json',encoding = 'utf8') as f:
        top_bad = json.load(f)

    if id >len(top_good):
        Filter = all_sentence[Filter_Bad] ['keyword']==keyowrd
        clean_data = all_sentence[Filter_Bad] [Filter]
        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:
                adj_top.append(i[0])

        data = []
        for ground_truth,keyword,sentence,adj,original,cluster in zip(clean_data['ground_truth'], [0]*len(clean_data['keyword']),\
                                                                     clean_data['sentence'],clean_data['adj'],clean_data['original'],clean_data["cluster"]):
            if adj_top[adj_num] in adj:
                data.append((ground_truth,keyword,sentence,original,cluster))

    else :
        Filter = all_sentence[Filter_Good]['keyword']==keyowrd
        clean_data = all_sentence[Filter_Good][Filter]
        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:
                adj_top.append(i[0])


        data = []
        for ground_truth,keyword,sentence,adj,original,cluster in zip(clean_data['ground_truth'], [1]*len(clean_data['keyword']),\
                                                                    clean_data['sentence'],clean_data['adj'],clean_data['original'],clean_data["cluster"] ):
            if adj_top[adj_num] in adj:
                data.append((ground_truth,keyword,sentence,original,cluster))

    
    context = {
        'all_num':len(sentence),
        'advantage_num':len(sentence[sentence['sentiment']== 1]),
        'disadvantage_num':len(sentence[sentence['sentiment']== 0]),
        'top_good': top_good,
        'top_bad': top_bad,
        'good_num':good_num,
        'good_center':good_center,
        'bad_num':bad_num,
        'bad_center':bad_center,
        'good_keyword_size':len(top_good),
        'bad_keyword_size': len(top_bad)+1,
        'adj_top':adj_top,
        'data':data,
        'id':temp,
        'hotel_name':hotel_name,
        'keyowrd':keyowrd,
    }
    return render(request,'NER.html',context)

def sidebar(request, slug,id): #to do
    temp = id
    hotel_name = slug
    # for test
    data = pd.read_csv('cache/'+hotel_name+'_to_CoNLL.csv')
    data['sentence'] = data['sentence'].map(lambda x: eval(x))
    Filter = ~data['uid'].duplicated() ##UID
    all_sentence = data[Filter]

    good_num = np.load('cache/'+hotel_name+'good_num.npy',allow_pickle = True)
    good_center = np.load('cache/'+hotel_name+'good_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_good.json',encoding = 'utf8') as f:
        top_good = json.load(f)
    bad_num = np.load('cache/'+hotel_name+'bad_num.npy',allow_pickle = True)
    bad_center = np.load('cache/'+hotel_name+'bad_center.npy',allow_pickle = True)
    with open('cache/'+hotel_name+'top_bad.json',encoding = 'utf8') as f:
        top_bad = json.load(f)

    if id == 0: #disadvantage
        temp = len(top_good)+len(top_bad)+1
        Filter2 = all_sentence['sentiment']== 0
        data = list(zip(all_sentence[Filter2]['ground_truth'].to_list(),
                        all_sentence[Filter2]['sentiment'].tolist(),
                        all_sentence[Filter2]['sentence'].tolist(),
                        all_sentence[Filter2]['original'],
                        all_sentence[Filter2]['cluster']))  

    elif id == 1: #advantage
        temp =  len(top_good)
        Filter2 = all_sentence['sentiment']== 1
        data = list(zip(all_sentence[Filter2]['ground_truth'].to_list(),
                        all_sentence[Filter2]['sentiment'].tolist(),
                        all_sentence[Filter2]['sentence'].tolist(),
                        all_sentence[Filter2]['original'],
                        all_sentence[Filter2]['cluster']))  

    else: #all
        temp = len(top_good)+len(top_bad)+2
        data = list(zip(all_sentence['ground_truth'].to_list(),
                        all_sentence['sentiment'].tolist(),
                        all_sentence['sentence'].tolist(),
                        all_sentence['original'],
                        all_sentence['cluster']))

    context = {
        'all_num':len(all_sentence),
        'advantage_num':len(all_sentence[all_sentence['sentiment']== 1]),
        'disadvantage_num':len(all_sentence[all_sentence['sentiment']== 0]),
        'top_good': top_good,
        'top_bad': top_bad,
        'good_num':good_num,
        'good_center':good_center,
        'bad_num':bad_num,
        'bad_center':bad_center,
        'good_keyword_size':len(top_good),
        'bad_keyword_size': len(top_bad)+1,
        'data':data,
        'id':temp,
        'hotel_name':slug,
    } 

    return render(request,'NER.html',context)

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES['file']
            global filename
            filename = f.name
            print(filename)
            review = pd.read_csv(f)
            review.to_csv('split.csv',index = False)
           
            return HttpResponseRedirect('/catalog/split_label')
    else:
        form = UploadFileForm()
       
    return render(request, 'Upload.html', {'form': form})
def Split_Label(request):
    if request.method == 'POST':
        form = CheckForm(request.POST)

        if form.is_valid():
            c = request.POST.getlist('check')
            review = pd.read_csv('split.csv')
            raw_data= {'label':[], 'comm':[]}
            print(c)

            for split,review in zip(c,review['comm']):
                if(split=='false'):
                    raw_data['label'].append(0)
                    raw_data['comm'].append(review)
                elif(split=='true'):
                    raw_data['label'].append(1)
                    raw_data['comm'].append(review)
            global filename
            df = pd.DataFrame(raw_data, columns = ['label','comm'])
            df.to_csv(filename[0:-4]+'_label.csv',index =False)

            return HttpResponseRedirect('/catalog/upload')
    else:
        form = CheckForm()

    review = pd.read_csv('split.csv')
    s = review['comm']
    context = {
        'form': form,
        's':s,
    }
    return render(request, 'Split_Label.html', context)

def simple_crawl(request,slug):  #not use
    review = pd.read_csv('cache/'+slug+'.csv')
    s = review['comm']
    context = {
        's': s,
        'hotel_name':slug,
    }
    return render(request,'simple_crawl.html',context)

def sentiment(request):
    # review = pd.read_csv('ground_truth.csv')
    # rs = sen.Sentiment(review['comm'])
    # df = sen.to_csv(rs,review)
    
    # label = df['label']
    # comm = df['comm']
    # Dict = {}
    
    # for i,j in zip(label,comm):
    #     Dict[j]=i
 
    return render(request,'sentiment.html',locals())

def Ner(request,slug):
    hotel_name = slug
    data =0

    if not os.path.isfile('cache/'+hotel_name+'_to_CoNLL.csv'):
        review = pd.read_csv('cache/'+hotel_name+'.csv', dtype={'comm': str})
        pred = sen.prediction(review)
        df = sen.to_csv(pred,hotel_name)
        ner_data = pd.read_csv('cache/'+hotel_name+'_label.csv')
        ner_data['original'] = review['original_comm']
        ner_data['ground_truth'] = df['label']
        ner.data = ner_data
        pred = ner.prediction()
        ner.to_CoNLL(pred,hotel_name)
    
    data = pd.read_csv('cache/'+hotel_name+'_to_CoNLL.csv')
    data['sentence'] = data['sentence'].map(lambda x: eval(x))
    Filter = ~data['uid'].duplicated() ##UID
    all_sentence = data[Filter]

    data = list(zip(all_sentence['ground_truth'].to_list(),
                    all_sentence['sentiment'].tolist(),
                    all_sentence['sentence'].tolist(),
                    all_sentence['original'],
                    all_sentence[Filter]["cluster"]))
    
    context = {
        'hotel_name':hotel_name,
        'all_num':len(all_sentence),
        'advantage_num':len(all_sentence[all_sentence['sentiment']== 1]),
        'disadvantage_num':len(all_sentence[all_sentence['sentiment']== 0]),
        'data':data,
    }
    return render(request,'NER.html',context= context)

#def keyword_list(request):
    # extractor = pke.unsupervised.YAKE()
    # extractor.load_document(input='review.csv', language='en',normalization=None)
    # stoplist = stopwords.words('english')
    # extractor.candidate_selection(n=3, stoplist=stoplist)
    # window = 2
    # use_stems = False # use stems instead of words for weighting
    # extractor.candidate_weighting(window=window,
    #                           stoplist=stoplist,
    #                           use_stems=use_stems)
    # threshold = 0.8
    # keyphrases = extractor.get_n_best(n=10, threshold=threshold)
    # # raw_data= {'key':[]}
    # # for i in keyphrases:
    # #     raw_data['key'].append(i[0])
    
    # context = {
    #     'keyphrases': keyphrases,
    # }
    # return render(request,'Keyword_List.html',context)
    
def keyword(request):
    
    # ke.final_comm = {}
    # ke.review = pd.read_csv('review_label.csv')
    # ke.cal_sim()
    # key = ke.final_comm

    return render(request,'Keyword_Extraction.html',locals())