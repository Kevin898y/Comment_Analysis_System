import re
import logging
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk.data
from .Split_Class import Bert_Split



class Booking_crawler:
    def __init__(self,url):
        self.url = url
        self.page_number = 0
        self.soup = []
        
    def find_max_page(self):
        link = self.soup.findAll('div',class_='bui-pagination__item')
        page = link[-2].get_text()
        page = re.sub('\r|\n', '', page)
        self.page_number = int(page[-2:])  
    
    def load_soup_online(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        req = requests.get(self.url,headers=headers)
        data = req.text
        req.close()
        self.soup = BeautifulSoup(data, 'html.parser')
    
    def Find_Review_Url(self):
        urlbase1 = 'https://www.booking.com/reviewlist.en-gb.html?'
        urlbase2 = 'cc1=pl;dist=1;'  ##gb
        urlbase3 = 'r_lang=en;'
        urlbase4 = 'type=total&;offset=0;rows=10'
        pattern_start = self.url.find('aid=')
        temp = self.url[pattern_start:].find('sid=')
        pattern_end = self.url[pattern_start+temp:].find('&')
        pattern = self.url[pattern_start:pattern_start+temp+pattern_end+1]
        pattern = pattern.replace('&', ';')

        pagename_start = self.url.find('pl/')
        pagename_end = self.url[pagename_start:].find('.en-gb')
        pagename = self.url[pagename_start+3 : pagename_start+pagename_end]

        srpvid_start = self.url.find('srpvid=')
        srpvid_end = self.url[srpvid_start:].find(';')
        srpvid = self.url[srpvid_start:srpvid_start+srpvid_end+1] 
        self.url = urlbase1+pattern+urlbase2+'pagename='+pagename+';'+urlbase3+srpvid+urlbase4
    def remove_emoji(self,text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F917"
                               "]+"
           , flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text).strip()
    def splitSentence(self,paragraph):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(paragraph)
        Clauses = []
        for i in sentences:
            temp = self.remove_emoji(i)
            if len(temp) > 1:
                Clauses.append(temp)
        return Clauses
    def ToCSV(self):
        review = pd.read_csv('review.csv')
        raw_data= {'label':[],'comm':[]}
        for label,comm in zip(review['label'],review['comm'] ) :
            label = int(label)
            comm = self.splitSentence(str(comm))
            for j in comm:
                if j != 'Nothing' and j!='N/a' and j != 'N/A' and j!= 'n/a' and j != 'n/A' and j!='nan' and len(j)>0:
                    # j = self.remove_emoji(str(j))
                    raw_data['label'].append(label)
                    raw_data['comm'].append(str(j))
        df = pd.DataFrame(raw_data, columns = ['label','comm'])
        df.to_csv('review.csv',index = False)
        # 分割句子
        split = Bert_Split('split2.pkl','review.csv')
        p = split.prediction()
        df = split.split(p)
        df.to_csv('temp.csv',index = False)
        return df

    def Scrapy_Review(self):
        self.Find_Review_Url()
        
        # pattern_start = self.url.find('offset')
        # pattern_end = self.url[pattern_start:].find(';')
        # url_pattern = self.url[pattern_start+7 :pattern_start+pattern_end]

        self.load_soup_online()
        try:
            self.find_max_page()
        except:
            self.page_number = 1

        raw_data= {'label':[],'comm':[]}

        for idx_page in tqdm(range(self.page_number)):
            url_pattern = idx_page*10
            pattern_start = self.url.find('offset')
            pattern_end = self.url[pattern_start:].find(';')
            self.url = self.url[:pattern_start+7]+str(url_pattern)+self.url[pattern_start+pattern_end:]
            self.load_soup_online()

            for i  in self.soup.findAll('p',class_='c-review__inner'):
                comm = i.findAll('span',class_='c-review__body')
                for j in comm:
                    if j != None:
                        j = j.get_text()
                        j = re.sub('\r|\n', '', j)
                        if j != 'There are no comments available for this review' and len(j)>0:
                            raw_data['comm'].append(j)
                            if(i.find('svg',class_='bk-icon -iconset-review_great c-review__icon')  !=None ):     
                                raw_data['label'].append(1)
                            else:
                                raw_data['label'].append(0)
        df = pd.DataFrame(raw_data, columns = ['label','comm'])
        df.to_csv('review.csv',index = False)