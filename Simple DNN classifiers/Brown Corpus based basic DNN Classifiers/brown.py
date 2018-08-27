# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:31:33 2018

@author: HP
"""

from collections import defaultdict
from nltk.corpus import brown,stopwords
import random
import nltk
#brown.categories()
#brown.words(categories='news')
#brown.words(fileids=['cg22'])
#brown.sents(categories=['news', 'editorial', 'reviews'])

dataset = [] # 500 samples

for category in brown.categories():
    for fileid in brown.fileids(category):
        dataset.append((brown.words(fileids = fileid),category))

dataset = [([w.lower() for w in text],category) for text,category in dataset]

labels=[]
for sample in dataset:
    labels.append(sample[1])

inputset=[]
for sample in dataset:
    inputset.append(sample[0])

categ=brown.categories()

label_class=[]
for x in labels:
    label_class.append(categ.index(x))

len_finder=[]
for dat in inputset:
    len_finder.append(len(dat))
