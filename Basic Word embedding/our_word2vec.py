# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:40:16 2018

@author: HP
"""

import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities

os.chdir("G:\\NLP\\Dataset\\techcrunch");
df=pd.read_csv('techcrunch_updated.csv');



x=df['section'].values.tolist()
y=df['content'].values.tolist()

corpus= x+y
  
tok_corp= [nltk.word_tokenize(sent.decode('utf-8')) for sent in corpus]
       
           
model = gensim.models.Word2Vec(tok_corp, min_count=1, size = 50)
