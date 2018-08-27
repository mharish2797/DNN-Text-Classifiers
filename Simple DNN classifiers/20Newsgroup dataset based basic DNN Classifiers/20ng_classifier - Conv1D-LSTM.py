# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:55:01 2018

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:31:43 2018

@author: HP
"""

import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors as KV
from numpy import asarray
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LSTM ,Dropout,GRU, Bidirectional
from keras.layers import Embedding
from collections import defaultdict
from keras.layers import Conv1D, MaxPooling1D
import random
from sklearn.datasets import fetch_20newsgroups

batch_size=32
embedding_size=128
nclass=20

# Convolution
kernel_size = 5
filters1 = 64
filters2 =128
filters3=256
filters4=512
filters5=1024
pool_size = 4

# GRU
gru_output_size = 70
#LSTM
lstm_output_size = 70

trim_len=200
sample_cnt=500

trainer = fetch_20newsgroups(subset='train')
tester = fetch_20newsgroups(subset='test')

#input - output
train_ip=trainer.data
train_op=list(trainer.target)
test_ip=tester.data
test_op=list(tester.target)

ip=train_ip+test_ip
op=train_op+test_op

ip=ip[0:sample_cnt]

for ty in range(len(ip)):
    ip[ty]=ip[ty][0:trim_len]

op=op[0:sample_cnt]
len_finder=[]
for dat in ip:
 len_finder.append(len(dat))

#Splitting train and test
        
input_train=[]
input_test=[]
input_valid=[]
j=0;
for zz in ip:
    j=j+1
    if (j%5 is 0):
        input_test.append(zz)
    elif(j%5 is 1):
        input_valid.append(zz)
    else:
        input_train.append(zz)

        
label_train=[]
label_test=[]
label_valid=[]
j=0;
for zz in op:
    j=j+1
    if (j%5 is 0):
        label_test.append(zz)
    elif(j%5 is 1):
        label_valid.append(zz)
    else:
        label_train.append(zz)
        
        
#one hot encoding

i=0
y_train=np.zeros((len(label_train),max(label_train)+1))
for x in label_train:
    y_train[i][x]=1
    i=i+1

i=0
y_test=np.zeros((len(label_test),max(label_test)+1))
for x in label_test:
    y_test[i][x]=1
    i=i+1

i=0
y_valid=np.zeros((len(label_valid),max(label_valid)+1))
for x in label_valid:
    y_valid[i][x]=1
    i=i+1

t = Tokenizer()
t.fit_on_texts(input_train)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(input_train)
#print(encoded_docs)
# pad documents to a max length of 4 words
max_length = max(len_finder)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open("G:\\NLP\\Dataset\\GloVe\\glove.6B.100d.txt", encoding="utf8")
for line in f:
 values = line.split()
 word = values[0]
 coefs = asarray(values[1:], dtype='float32')
 embeddings_index[word] = coefs
f.close()
#print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
 embedding_vector = embeddings_index.get(word)
 if embedding_vector is not None:
  embedding_matrix[i] = embedding_vector


#Validating the model
vt = Tokenizer()
vt.fit_on_texts(input_valid)
vvocab_size = len(vt.word_index) + 1
# integer encode the documents
vencoded_docs = vt.texts_to_sequences(input_valid)
#print(encoded_docs)
# pad documents to a max length of 4 words
vpadded_docs = pad_sequences(vencoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)



#Testing the model
tt = Tokenizer()
tt.fit_on_texts(input_test)
tvocab_size = len(tt.word_index) + 1
# integer encode the documents
tencoded_docs = tt.texts_to_sequences(input_test)
#print(encoded_docs)
# pad documents to a max length of 4 words
tpadded_docs = pad_sequences(tencoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)


# define model 
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(nclass, activation='softmax'))


# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs,y_train, epochs=1, verbose=0, validation_data=(vpadded_docs, y_valid))

# evaluate the model
loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


