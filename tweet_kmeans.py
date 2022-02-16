import tweepy


import requests
import re
import numpy as np
import math
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


# 트위터 API에서 발급 받은 key 입력
consumer_key= ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
bearer_Token = ""


# 한글 토크나이저 사전 위치
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# 핸들러 생성 및 개인정보 인증요청
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# 액세스 요청
auth.set_access_token(access_token, access_token_secret)
#twitter API 생성
api = tweepy.API(auth)

#한국의 트렌드 불러오기
def get_trend():
    trend_header = {'Authorization': 'Bearer {}'.format(bearer_Token)}
    trend_params = {'id': 23424868, } # 한국 Woeid
    trend_url = 'https://api.twitter.com/1.1/trends/place.json'
    trend_resp = requests.get(trend_url, headers=trend_header, params=trend_params)
    tweet_data = trend_resp.json() # json 저장

    # 트렌드 Top 1 가져오기
    trend_top1 = {}
    for i in range(0, 1):
        trend_top1 = tweet_data[0]['trends'][i]
        print(tweet_data[0]['trends'][i])
    print(trend_top1['name'])

    return trend_top1['name']


# 트렌드에 관련된 트윗 검색
def get_tweets(keyword,num):
    location = "%s,%s,%s" % ("35.95", "128.25", "1000km")

    cursor = tweepy.Cursor(api.search,
                           q=keyword+" -filter:retweets",
                           # count='1',
                           # geocode=location,
                           lang='ko',
                           result_type='mixed').items(num)
    tweet_list = []
    n = 0
    for tw in cursor:
        print(n, tw.text)
        tweet_list.append(tw.text)
        n = n + 1
    return tweet_list
'''
    mecab_doc = np.empty([])
    for i in range(len(tweet_list)):
        sentence = ' '.join(preprocessing_mecab(tweet_list[i]))
        # print(sentence)
        mecab_doc = np.append(mecab_doc, sentence)
    print(mecab_doc)
    return mecab_doc
'''

# 텍스트 클리닝
def CleanText(readData, Num=False, Eng=False):

    text = re.sub('RT @[\w_]+: ', '', readData)  # Remove Retweets
    text = re.sub('@[\w_]+', '', text) # Remove Mentions

    # Remove or Replace URL
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ',
                  text)  # http로 시작되는 url
    text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ',
                  text)  # http로 시작되지 않는 url
    text = re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','',text)

    text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', text) # Remove Hashtag
    text = re.sub('[&]+[a-z]+', ' ', text) # Remove Garbage Words (ex. &lt, &gt, etc)

    text = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', text) # Remove Special Characters
    text = re.sub('[ㄱ-ㅎㅏ-ㅣ]+',' ', text) # Remove 자음 모음

    text = text.replace('\n', ' ') # Remove newline

    if Num is True:
        text = re.sub(r'\d+', ' ', text) # Remove Numbers

    if Eng is True:
        text = re.sub('[a-zA-Z]', ' ', text) # Remove English

    text = ' '.join(text.split()) # Remove multi spacing & Reform sentence
    print(text)
    return text

# 형태소 나눈뒤 뜻이 없는 단어 제거
def preprocessing_mecab(readData):
    #### Clean text
    sentence = CleanText(readData)
    #### Tokenize
    morphs = mecab.pos(sentence)

    JOSA = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]  # 조사
    SIGN = ["SF", "SE", "SSO", "SSC", "SC", "SY"]  # 문장 부호

    TERMINATION = ["EP", "EF", "EC", "ETN", "ETM"]  # 어미
    SUPPORT_VERB = ["VX"]  # 보조 용언
    NUMBER = ["SN"]

    # Remove JOSA, EOMI, etc
    morphs[:] = (morph for morph in morphs if morph[1] not in JOSA + SIGN + TERMINATION + SUPPORT_VERB)
    # Remove length-1 words
    morphs[:] = (morph for morph in morphs if not (len(morph[0]) == 1))
    # Remove Numbers
    morphs[:] = (morph for morph in morphs if morph[1] not in NUMBER)
    # Result pop-up
    result = []
    for morph in morphs:
        result.append(morph[0])

    return result


# TF-IDF를 통해 vectorizing
def vectorization(doc):
    vector = TfidfVectorizer(max_df=30)
    X = vector.fit_transform(doc)
    print(X)
    return X


# K값 구하기 위한 공식 (첫 항목과 마지목 항목을 이은 선에서 각 클러스터들과의 거리를 계산)
def calc_distance(x1, y1, a, b ,c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d

# K값 자동계산
def elbow_auto(readVector, Clusters):
    score = []
    K = range(1, Clusters)
    for i in K:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(readVector)
        score.append(kmeans.inertia_)

    # 자동 계산식
    K_end = len(score)-1
    a = score[0] - score[K_end]
    b = K[K_end] - K[0]
    c1 = K[0] * score[K_end]
    c2 = K[K_end] * score[0]
    c = c1 - c2
    distance_from_line = []
    for i in range(K_end+1):
        distance_from_line.append(
            calc_distance(K[i], score[i], a, b, c))
    distance_max = max(distance_from_line)
    selected_K = distance_from_line.index(distance_max) + 1
    print(" ")
    print("자동으로 계산된 클러스터 개수 : "+ str(selected_K)+"\n")

    plt.figure(figsize=(7,6))
    plt.plot(range(1,Clusters), score, marker='o')
    plt.xlabel('number of cluster')
    plt.xticks(np.arange(0,Clusters,1))
    plt.ylabel('SCORE')
    plt.title('Elbow Method - number of cluster : '+str(Clusters))
    plt.show()
    return selected_K


# K-means clustering 진행
def K_means_clustering(vectorDoc, c_num, doc):
    km_cluster = KMeans(n_clusters=c_num, max_iter=10000, random_state=0)
    km_cluster.fit(vectorDoc)
    c_label = km_cluster.labels_
    c_centers = km_cluster.cluster_centers_

    df_dict = {'word': doc,
               'cluster_label': c_label}
    doc_df = pd.DataFrame(df_dict)

    for i in range(c_num):
        print('<<Clustering Label {0}>>'.format(i) + '\n')
        print(doc_df.loc[doc_df['cluster_label'] == i])
        print(' ')


    result_doc = doc_df.sort_values(by=['cluster_label'])


    return result_doc

# Wordcloud 시각화
def visualization(c_doc, c_num, stopWord):
    stopword = preprocessing_mecab(stopWord)
    for k in range(0,c_num):
        s = c_doc[c_doc.cluster_label == k]
        text = s['word'].str.cat(sep=' ')
        text = text.lower()
        text =' '.join([word for word in text.split()])
        wordcloud = WordCloud(font_path='C:\\font\\HYKANB.TTF',
                              stopwords= stopword,
                              max_font_size=100,
                              max_words=100,
                              background_color="white").generate(text)
        print('Cluster: {}'.format(k+1))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
# WordCloud 다중플롯 시각화
def visualization2(c_doc, c_num, stopWord):
    stopword = preprocessing_mecab(stopWord)
    n=1
    for k in range(0,c_num):
        s = c_doc[c_doc.cluster_label == k]
        text = s['word'].str.cat(sep=' ')
        text = text.lower()
        text =' '.join([word for word in text.split()])
        wordcloud = WordCloud(font_path='C:\\font\\HYKANB.TTF',
                              stopwords= stopword,
                              max_font_size=100,
                              max_words=100,
                              background_color="white").generate(text)
        plt.subplot(2,int(c_num/2),n)
        plt.title('Cluster : '+str(n))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        n = n+ 1
    plt.show()


if __name__ =="__main__":
    #한국의 트렌드 불러오기
    search = get_trend()

    #특정 검색어 설정
    #search = '코로나'

    #트렌드에 관한 트윗 검색크롤링, 트윗 갯수 설정
    doc = get_tweets(search, 100)

    #트윗 백터화
    doc_vector = vectorization(doc)

    # 클러스터 수 자동계산
    clusters_num = elbow_auto(doc_vector, 20)

    # K-means clustering 진행
    clustering_doc = K_means_clustering(doc_vector, clusters_num, doc)

    print(clustering_doc)

    #시각화
    visualization(clustering_doc,clusters_num, search)
    visualization2(clustering_doc,clusters_num, search)











