from numpy import asarray
from numpy import zeros
import pandas as pd
import os
from keras.datasets import reuters

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Flatten,Input
from keras.layers import Dropout
from keras.layers import GRU,CuDNNGRU,Reshape,maximum
from keras.layers import Bidirectional,Concatenate
from keras.layers import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPool2D
from keras.layers import Embedding
from keras.layers.merge import concatenate
from collections import defaultdict
from nltk.corpus import brown,stopwords
import random
import nltk
import numpy as np
from sklearn.datasets import fetch_20newsgroups

#Custom Activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import math as m

batch_size=32
embedding_size=128
nclass=7

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

os.chdir("G:/NLP/Dataset/techcrunch");
df=pd.read_csv('techcrunch_updated - Test.csv');

#input
ip=df['description'].values.tolist()
ip=ip[0:sample_cnt]

for ty in range(len(ip)):
    ip[ty]=ip[ty][0:trim_len]

len_finder=[]
for dat in ip:
 len_finder.append(len(dat))

#output
op=df['section'].values.tolist()
op=op[0:sample_cnt]

labels=[]
for zen in op:
    if zen not in labels:
        labels.append(zen)

label_class=[]
for ix in op:
 label_class.append(labels.index(ix))


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
y=np.zeros((len(label_class),max(label_class)+1))
for x in label_class:
 y[i][x]=1
 i=i+1


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
model.add(Bidirectional(GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Flatten())
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


