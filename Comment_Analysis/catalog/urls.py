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
    path('top/<slug:slug>/<int:id>/', views.top1, name='top1'),
    path('sidebar/<slug:slug>/<int:id>/', views.sidebar, name='sidebar'),
    # path('top/', include('Comment_Analysis.catalog.urls')),
    path('top/<slug:slug>/<int:id>/<int:adj_num>', views.top2, name='top2'),
    # path('top3/', views.top3, name='top3'),
    # path('top4/', views.top4, name='top4'),
    # path('top5/', views.top5, name='top5'),
 
]