# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:20:34 2018

@author: HP
"""
from numpy import asarray
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.layers import Flatten, LSTM ,Dropout,GRU, Bidirectional, RepeatVector
from keras.layers import Embedding, SimpleRNN, Conv2D, MaxPooling2D
from collections import defaultdict
from nltk.corpus import brown,stopwords
from keras.layers import Conv1D, MaxPooling1D
import random
import nltk
import matplotlib.pyplot as plt
#brown.categories()
#brown.words(categories='news')
#brown.words(fileids=['cg22'])
#brown.sents(categories=['news', 'editorial', 'reviews'])

batch_size=64
embedding_size=128

nclass=15
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
 inputset.append(' '.join(sample[0]))

categ=brown.categories()

label_class=[]
for x in labels:
 label_class.append(categ.index(x))

len_finder=[]
for dat in inputset:
 len_finder.append(len(dat))

input_train=[]
input_test=[]
input_valid=[]
j=0;
for zz in inputset:
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
for zz in label_class:
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
filter_sizes=[3,4,5]
sequence_length=500


# define model 
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(GRU(70,dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(14653))
model.add(Reshape((1,14653,70)))
model.add(Conv2D(filters1, kernel_size=(filter_sizes[0], 100), padding='valid', kernel_initializer='normal', activation='relu'))
model.add(MaxPool2D(pool_size=(sequence_length -filter_sizes[0] +1, 1), strides=(1,1), padding='valid'))
#model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
#model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
#model.add(Conv1D(512,kernel_size,padding='valid',activation='relu',strides=1))
#model.add(MaxPooling1D(pool_size=pool_size))
#model.add(Bidirectional(GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
#model.add(Bidirectional(LSTM(gru_output_size)))
#model.add(GRU(70,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(nclass, activation='softmax'))



# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
hist=model.fit(padded_docs,y_train, epochs=1, verbose=0, validation_data=(vpadded_docs, y_valid))
#plt(hist)
# evaluate the model
loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
