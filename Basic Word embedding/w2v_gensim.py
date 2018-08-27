import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors as KV
os.chdir("G:/NLP/Dataset/techcrunch");
df=pd.read_csv('techcrunch_updated - Test.csv');


xx=[]
x=df['section'].values.tolist()
x=x[0:39410]
for k in range(len(x)):
    x[k]=x[k].lower()
    xx.append(x[k].strip().split())
    for ks in range(len(xx[k])):
        if not xx[k][ks].isalpha():
            xx[k][ks]=''

yy=[]
y=df['description'].values.tolist()
y=y[0:39410]
for k in range(len(y)):
    y[k]=y[k].lower()
    yy.append(y[k].strip().split())
    for ks in range(len(yy[k])):
        if not yy[k][ks].isalpha():
            yy[k][ks]=''
                    
corpus=xx+yy

#print (corpus)
#tok_corp= [nltk.word_tokenize(sent) for sent in corpus]

#stopword removal
filtered=[]
stopset=stopwords.words("english")
for l in range(len(corpus)):
    zoom=[]
    for tok in corpus[l]:
        if tok not in stopset and tok!='':
            zoom.append(tok)
    filtered.append(zoom) 

#modelx = KV.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.Word2Vec(filtered, min_count=1, size = 50)
print (model['santa'])
print (model.most_similar('santa'))