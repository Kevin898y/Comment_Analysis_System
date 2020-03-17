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
from nltk.corpus import stopwords
import numpy as np
from catalog.forms import SearchForm
from catalog.forms import CheckForm
from catalog.forms import UploadFileForm
# from pip._vendor.html5lib.filters.sanitizer import Filter
import os
import json

# ke = Keyword_Extraction('wordtovector/GoogleNews-vectors-negative300.bin','data/keyword.csv','1')
# ner = Bert_NER('model/NER3/')
ner = Bert_NER('model/NER3/') 
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
        form = SearchForm(initial={'booking_url': 'url'})

    context = {
        'form': form,
    }
    return render(request,'home.html', context)
    
def top1(request,slug, id): #to do
    temp = id
    if id >len(ner.good_keyword_top5):
        id = id-len(ner.good_keyword_top5)-1

        Filter = ner.bad_sentence ['keyword']==ner.bad_keyword_top5[id]
        clean_data = ner.bad_sentence[Filter]
        # data = list(zip( [0]*len(clean_data['keyword']),clean_data['sentence'] ))
        data = list(zip(clean_data['ground_truth'], [0]*len(clean_data['keyword']),clean_data['sentence'] ))
        # adjtop = clean_data.adj.value_counts().index

        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        # adjtop = [i[0] for i in adjtop]
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:
                adj_top.append(i[0])
        adjtop = adj_top
        

    else :
        Filter = ner.good_sentence ['keyword']==ner.good_keyword_top5[id]
        clean_data = ner.good_sentence[Filter]
        # data = list(zip( [1]*len(clean_data['keyword']),clean_data['sentence'] ))
        data = list(zip(clean_data['ground_truth'], [1]*len(clean_data['keyword']),clean_data['sentence'] ))
        all_adj = {}
        for i in clean_data['adj']:
            for adj in i:
                if adj not in all_adj:
                    all_adj[adj] = 1
                else:
                    all_adj[adj]+=1
        adjtop = sorted(all_adj.items(), key=lambda d: d[1],reverse=True)
        # adjtop = [i[0] for i in adjtop]
        adj_top = []
        for i in adjtop:
            if i[1]>len(clean_data['keyword'])/12 and i[1]>=3:
                adj_top.append(i[0])
        adjtop = adj_top
        
          
    
    context = {
        'good_keyword_top5': ner.good_keyword_top5,
        'bad_keyword_top5': ner.bad_keyword_top5,
        'good_keyword_size':len(ner.good_keyword_top5),
        'bad_keyword_size': len(ner.bad_keyword_top5),
        'adj_top':adjtop,
        'data':data,
        'id':temp,
        'hotel_name':slug,
    }

    return render(request,'NER.html',context)

def top2(request, slug,id, adj_num): #to do
    temp = id
    if id >len(ner.good_keyword_top5):
        id = id-len(ner.good_keyword_top5)-1

        Filter = ner.bad_sentence ['keyword']==ner.bad_keyword_top5[id]
        clean_data = ner.bad_sentence[Filter]
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
        for ground_truth,keyword,sentence,adj in zip(clean_data['ground_truth'], [0]*len(clean_data['keyword']),clean_data['sentence'],clean_data['adj'] ):
            if adj_top[adj_num] in adj:
                data.append((ground_truth,keyword,sentence))

    else :
        Filter = ner.good_sentence ['keyword']==ner.good_keyword_top5[id]
        clean_data = ner.good_sentence[Filter]
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
        for ground_truth,keyword,sentence,adj in zip(clean_data['ground_truth'], [1]*len(clean_data['keyword']),clean_data['sentence'],clean_data['adj'] ):
            if adj_top[adj_num] in adj:
                data.append((ground_truth,keyword,sentence))
         
    adjtop = adj_top
    context = {
        'good_keyword_top5': ner.good_keyword_top5,
        'bad_keyword_top5': ner.bad_keyword_top5,
        'good_keyword_size':len(ner.good_keyword_top5),
        'bad_keyword_size': len(ner.bad_keyword_top5),
        'adj_top':adjtop,
        'data':data,
        'id':temp,
        'hotel_name':slug,
    }

    return render(request,'NER.html',context)
def sidebar(request, slug,id): #to do
    temp = id
    # for test
    Filter_duplicated = ~ner.all_sentence['sentence'].duplicated() 

    if id == 0: #disadvantage
        temp = len(ner.good_keyword_top5)+len(ner.bad_keyword_top5)+1
        data = ner.bad

        # for test
        Filter = ner.all_sentence['sentiment']== 0
        ground_truth = list(np.array(ner.ground_truth)[Filter_duplicated & Filter])
        data = list(zip(ground_truth,[i[0]for i in data],[i[1] for i in data] ))  

    elif id == 1: #advantage
        temp =  len(ner.good_keyword_top5)
        data = ner.good

        # for test
        Filter = ner.all_sentence['sentiment']== 1
        ground_truth = list(np.array(ner.ground_truth)[Filter_duplicated & Filter])
        data = list(zip(ground_truth,[i[0]for i in data],[i[1] for i in data] )) 

    else: #all
        temp = len(ner.good_keyword_top5)+len(ner.bad_keyword_top5)+2
        data = ner.all
        
        # for test
        ground_truth = list(np.array(ner.ground_truth)[Filter_duplicated])
        data = list(np.array(data)[Filter_duplicated])
        data = list(zip(ground_truth,[i[0]for i in data],[i[1] for i in data] ))
        
        
    context = {
        'good_keyword_top5': ner.good_keyword_top5,
        'bad_keyword_top5': ner.bad_keyword_top5,
        'good_keyword_size':len(ner.good_keyword_top5),
        'bad_keyword_size': len(ner.bad_keyword_top5)+1,
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
    # url = 'https://www.booking.com/hotel/gb/italianflat-brompton.en-gb.html?aid=304142;label=gen173nr-1FCAEoggI46AdIM1gEaOcBiAEBmAEwuAEXyAEM2AEB6AEB-AELiAIBqAID;sid=d1c049c0e9123fddfa3b6174f141e95b;dest_id=-2601889;dest_type=city;dist=0;hapos=3;hpos=3;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1550862196;srpvid=12a885fa98b8042d;type=total;ucfs=1&#hotelTmplhttps://www.booking.com/hotel/gb/italianflat-brompton.en-gb.html?aid=304142;label=gen173nr-1FCAEoggI46AdIM1gEaOcBiAEBmAEwuAEXyAEM2AEB6AEB-AELiAIBqAID;sid=d1c049c0e9123fddfa3b6174f141e95b;dest_id=-2601889;dest_type=city;dist=0;hapos=3;hpos=3;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1550862196;srpvid=12a885fa98b8042d;type=total;ucfs=1&#hotelTmpl'
    # url = request
    # temp = tes
    # crawler = Booking_crawler(url)
    # crawler.Scrapy_Review()
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
    review = pd.read_csv('cache/'+hotel_name+'.csv', dtype={'comm': str})

    pred = sen.prediction(review)
    df = sen.to_csv(pred,hotel_name)
 
    ner.data = pd.read_csv('cache/'+hotel_name+'_label.csv')
    data =0
    if os.path.isfile('cache/'+hotel_name+'_to_CoNLL.csv'):
        data = ner.to_CoNLL(pred,hotel_name)
    else:
        pred = ner.prediction(hotel_name)
        data = ner.to_CoNLL(pred,hotel_name)

    Filter = ~ner.all_sentence['sentence'].duplicated() 
    ground_truth = list(np.array(ner.ground_truth)[Filter]) 
    # ground_truth = ner.ground_truth
    data = list(np.array(data)[Filter])
    data = list(zip(ground_truth,[i[0]for i in data],[i[1] for i in data] ))
    
    # data = list(zip(ner.ground_truth,[i[0]for i in data],[i[1] for i in data] ))        


    # list(sen.dict.values())#label
    # list()
    context = {
        'good_keyword_top5': ner.good_keyword_top5,
        'bad_keyword_top5': ner.bad_keyword_top5,
        'hotel_name':hotel_name,
        # 'adj_top5':ner.adj_top5,
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