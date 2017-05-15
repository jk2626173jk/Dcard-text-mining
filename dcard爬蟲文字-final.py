 import time
 import urllib.parse
 import os
 import re
 import urllib.request
 import json
 import sys
 import matplotlib
 matplotlib.use('Agg')
 import requests
 from bs4 import BeautifulSoup
 import json
 from collections import defaultdict
 from sklearn.utils import shuffle
 import jieba
 import matplotlib.pyplot as plt
 import numpy as np
 import seaborn as sns

 from sklearn.feature_extraction import DictVectorizer
 from sklearn.feature_extraction.text import TfidfTransformer
 from sklearn.svm import LinearSVC
 sns.set(style='whitegrid')
 url ="https://www.dcard.tw/f/relationship"
 non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

 def get_posts_list(url):
     global i
     response = requests.get(url)
     soup = BeautifulSoup(response.text, 'lxml')
     articles = soup.find_all('div', 'PostEntry_container_245XM')
     lists=[]
     for article in articles:
         push_count = 0
         comment=0
         if article.find('div', 'PostLikeCount_likeCount_2uhBH').string:#找出按喜歡的人數
             try:
                 push_count = int(article.find('div', 'PostLikeCount_likeCount_2uhBH').string)  # 轉換字串為數
             except ValueError:  # 若轉換失敗，不做任何事，push_count 保持為 0
                 pass
         if article.find('div', 'PostEntry_comment_18ty5').string:#找出留言的
             try:
                 comment = int(article.find('div', 'PostEntry_comment_18ty5').string)  # 轉換字串為數字
             except ValueError:  # 若轉換失敗，不做任何事，comment 保持為 0
                 pass
                     # 取得文章連結及標題
         if article.find('a'):  # 有超連結，表示文章存在，未被刪除
             href = article.find('a')['href']
             title = article.find('strong').string
                     #print(title.translate(non_bmp_map))
                     #print(href.translate(non_bmp_map))
                     #print(push_count)
                     #print(comment)
             lists.append({
                 'title': title.translate(non_bmp_map),
                 'href': href.translate(non_bmp_map),
                 'push_count': push_count,
                 'comment' : comment
                 })
                         # 已取得文章列表，開始進入各文章取資料
     for list in lists:
         #print('Processing', list)
         #print('ok')
         page = get_web_page("https://www.dcard.tw" + list['href'].translate(non_bmp_map))
         if page:
         #print("start")
             soup = BeautifulSoup(page, 'lxml')
             if soup.find('div', 'Post_content_1xpMb'):
                 list['paragraph']=soup.find('div', 'Post_content_1xpMb').text.translate(non_bmp_map)
             #print(list['title'].translate(non_bmp_map))
             #print(list['article'])
           #print("----------------------------------------------------------------------------------------------")
     return lists
 def get_web_page(url):#確認每一個文章都可以連上
     time.sleep(0.5)  # 每次爬取前暫停 0.5 秒以免被PTT 網站判定為大量惡意爬取
     resp = requests.get(
     url=url,
     )
     #print(url)
     if resp.status_code!= 200:
         print('Invalid url:',resp.url)
         return None
     else:
         #print("ok")
         #print(resp.text.translate(non_bmp_map))
         return resp.text
 lists=get_posts_list(url)
 #print(lists)
 words=[]
 scores=[]
 for list in lists:
     d=defaultdict(int)
     content=list['paragraph']
     if list['push_count']!=0:
         for l in content.split('\n'):
             if l:
                 for w in jieba.cut(l):
                     d[w]+=1
         if len(d)>0:
             words.append(d)
             scores.append(1 if list['push_count']>1000 else 0)
 dvec=DictVectorizer()
 tfidf=TfidfTransformer()
 X=tfidf.fit_transform(dvec.fit_transform(words))

 #c_dvec=DictVectorizer()
 #c_tfidf=TfidfTransformer()
 #c_X=c_tfidf.fit_transform(c_dvec.fit_transform(c_words))

 svc=LinearSVC()
 X_shuf, Y_shuf = shuffle(X,scores)
 svc.fit(X_shuf,Y_shuf)
#c_svc=LinearSVC()
 #c_svc.fit(c_X,c_scores)
 def display_top_features(weights,names,top_n,select=abs):
     top_features=sorted(zip(weights,names),key=lambda x: select(x[0]), reverse=True)[:top_n]
     top_weights=[x[0] for x in top_features]
     top_names=[x[1] for x in top_features]
     fig,ax=plt.subplots(figsize=(10,8))
     ind=np.arange(top_n)
     bars=ax.bar(ind,top_weights,color='blue',edgecolor='black')
     for bar,w in zip(bars,top_weights):
         if w<0:
             bar.set_facecolor('red')
     width=0.30
     ax.set_xticks(ind+width)
     ax.set_xticklabels(top_names,rotation=45,fontsize=12,fontdict={'fontname':
     'Droid Sans Fallback','fontsize':12})
     plt.show(fig)
     plt.savefig("dcard.png",dpi=300,format="png")
 display_top_features(svc.coef_[0],dvec.get_feature_names(),30)

