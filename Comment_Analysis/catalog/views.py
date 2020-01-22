from .Booking_crawler import Booking_crawler
from .Review_Sentiment import Review_Sentiment
from .Keyword_Extraction import Keyword_Extraction
from .BERT_NER import Bert_NER
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
import pke
from nltk.corpus import stopwords
import numpy as np
from catalog.forms import SearchForm
from catalog.forms import CheckForm
from catalog.forms import UploadFileForm
from pip._vendor.html5lib.filters.sanitizer import Filter

# ke = Keyword_Extraction('wordtovector/GoogleNews-vectors-negative300.bin','data/keyword.csv','1')
good_ner = Bert_NER('model/NER3.pkl','test_data.csv')
bad_ner = Bert_NER('model/NER3.pkl','test_data.csv') 
sen = Review_Sentiment('model/tokenizer.p','model/sentimental.h5')


filename= '' 
def Search(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['booking_url']
            crawler = Booking_crawler(url)
            crawler.Scrapy_Review()
            review =crawler.ToCSV()
            s = review['comm']
            return render(request,'simple_crawl.html',locals())
    else:
        form = SearchForm(initial={'booking_url': 'url'})

    context = {
        'form': form,
    }
    return render(request,'home.html', context)
    
def top1(request, id): #to do
    temp = id
    if id >5:
        id -= 6
        if id == 5:
            keycomm = sen.dict
        else:
            Filter = bad_ner.sentence_label ['label']==bad_ner.keyword_top5[id]
            data = bad_ner.sentence_label[Filter]
            keycomm = dict(zip(data['comm'], [0]*len(data['label']) ))
    else :
        if id == 5:
            keycomm = sen.dict
        else:
            Filter = good_ner.sentence_label ['label']==good_ner.keyword_top5[id]
            data = good_ner.sentence_label[Filter]
            keycomm = dict(zip(data['comm'], [1]*len(data['label']) ))
    context = {
        'good_keyword_top5': good_ner.keyword_top5,
        'bad_keyword_top5': bad_ner.keyword_top5,
        'data':keycomm,
        'id':temp,
    }

    return render(request,'NER.html',context)

def sidebar(request, id): #to do
    temp = id
    data = sen.dict
    if id == 0: #disadvantage
        temp = 11
        data = sen.baddict
    elif id == 1: #advantage
        temp = 5
        data = sen.gooddict
    elif id == 2:
        temp = 12
    context = {
        'good_keyword_top5': good_ner.keyword_top5,
        'bad_keyword_top5': bad_ner.keyword_top5,
        'data':data,
        'id':temp,
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

def simple_crawl(request):  #not use
    # url = 'https://www.booking.com/hotel/gb/italianflat-brompton.en-gb.html?aid=304142;label=gen173nr-1FCAEoggI46AdIM1gEaOcBiAEBmAEwuAEXyAEM2AEB6AEB-AELiAIBqAID;sid=d1c049c0e9123fddfa3b6174f141e95b;dest_id=-2601889;dest_type=city;dist=0;hapos=3;hpos=3;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1550862196;srpvid=12a885fa98b8042d;type=total;ucfs=1&#hotelTmplhttps://www.booking.com/hotel/gb/italianflat-brompton.en-gb.html?aid=304142;label=gen173nr-1FCAEoggI46AdIM1gEaOcBiAEBmAEwuAEXyAEM2AEB6AEB-AELiAIBqAID;sid=d1c049c0e9123fddfa3b6174f141e95b;dest_id=-2601889;dest_type=city;dist=0;hapos=3;hpos=3;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1550862196;srpvid=12a885fa98b8042d;type=total;ucfs=1&#hotelTmpl'
    # url = request
    # temp = tes
    # crawler = Booking_crawler(url)
    # crawler.Scrapy_Review()
    review = pd.read_csv('review.csv')
    s = review['comm']
    value = 0
    return render(request,'simple_crawl.html',locals())

def sentiment(request):
    review = pd.read_csv('temp.csv')
    rs = sen.Sentiment(review['comm'])
    df = sen.to_csv(rs,review)
    
    label = df['label']
    comm = df['comm']
    Dict = {}
    
    for i,j in zip(label,comm):
        Dict[j]=i
 
    return render(request,'sentiment.html',locals())

def Ner(request):
    review = pd.read_csv('temp.csv', dtype={'comm': str})
    rs = sen.Sentiment(review['comm'])
    df = sen.to_csv(rs,review)

    good_ner.data = pd.read_csv('advantage.csv')
    bad_ner.data = pd.read_csv('disadvantage.csv')
    pred = good_ner.prediction()
    good_ner.to_CoNLL(pred)
    pred = bad_ner.prediction()
    bad_ner.to_CoNLL(pred)

    
    context = {
        'good_keyword_top5': good_ner.keyword_top5,
        'bad_keyword_top5': bad_ner.keyword_top5,
        # 'adj_top5':ner.adj_top5,
        'data':sen.dict,
        'good_data': good_ner.data['comm'].tolist(),
        'bad_data' : bad_ner.data['comm'].tolist(),
    }
    return render(request,'NER.html',context)

def keyword_list(request):
    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input='review.csv', language='en',normalization=None)
    stoplist = stopwords.words('english')
    extractor.candidate_selection(n=3, stoplist=stoplist)
    window = 2
    use_stems = False # use stems instead of words for weighting
    extractor.candidate_weighting(window=window,
                              stoplist=stoplist,
                              use_stems=use_stems)
    threshold = 0.8
    keyphrases = extractor.get_n_best(n=10, threshold=threshold)
    # raw_data= {'key':[]}
    # for i in keyphrases:
    #     raw_data['key'].append(i[0])
    
    context = {
        'keyphrases': keyphrases,
    }
    return render(request,'Keyword_List.html',context)
    
def keyword(request):
    
    # ke.final_comm = {}
    # ke.review = pd.read_csv('review_label.csv')
    # ke.cal_sim()
    # key = ke.final_comm

    return render(request,'Keyword_Extraction.html',locals())