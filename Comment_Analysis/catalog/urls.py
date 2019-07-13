from django.urls import path
from . import views

urlpatterns = [
    path('', views.Search, name='index'),
    path('crawl/', views.simple_crawl, name='crawl'),
    path('upload/', views.upload_file, name='upload'),
    path('split/', views.Split, name='split'),
    path('sentiment/', views.sentiment, name='sentiment'),
    path('keyword_list/', views.keyword_list , name='keyword_list'),
    path('Keyword_Extraction/', views.keyword , name='keyword'),
 
]