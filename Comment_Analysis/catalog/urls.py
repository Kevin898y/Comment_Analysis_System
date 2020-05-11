from django.urls import path
from . import views

urlpatterns = [
    path('', views.Search, name='index'),
    path('crawl/<slug:slug>/', views.simple_crawl, name='crawl'),
    path('upload/', views.upload_file, name='upload'),
    path('split_label/', views.Split_Label, name='split_label'),
    path('sentiment/', views.sentiment, name='sentiment'),
    path('ner/<slug:slug>/', views.Ner, name='ner'),
    # path('keyword_list/', views.keyword_list , name='keyword_list'),
    path('Keyword_Extraction/', views.keyword , name='keyword'),
    path('top/<slug:hotel_name>/<int:id>/<slug:keyowrd>/', views.top_keyowrd, name='top_keyowrd'),
    path('top/<slug:hotel_name>/<int:id>/<slug:keyowrd>/<int:adj_num>', views.top_adj, name='top_adj'),
    path('sidebar/<slug:slug>/<int:id>/', views.sidebar, name='sidebar'),
    # path('top/', include('Comment_Analysis.catalog.urls')),
   

 
]