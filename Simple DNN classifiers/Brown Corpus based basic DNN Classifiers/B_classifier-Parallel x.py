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
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
     val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
     val_targ = self.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict, average='micro')
     _val_recall = recall_score(val_targ, val_predict, average='micro')
     _val_precision = precision_score(val_targ, val_predict, average='micro')
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return
 
metriczs = Metrics()

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000

batch_size=30
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
sequence_length=500
filter_sizes = [3,4,5]
# GRU
gru_output_size = 70
#LSTM
lstm_output_size = 70

trim_len=200
sample_cnt=500

def newacti( x,alpha=m.exp(-1) ):
  return K.elu(x,alpha)

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
ip=inputset
categ=brown.categories()

label_class=[]
for x in labels:
 label_class.append(categ.index(x))

len_finder=[]
for dat in inputset:
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


sequence_length=max_length

# create the model
embedding_vecor_length = 100
visible = Input(shape=(sequence_length,), dtype='int32')
# first feature extractor
embedding = Embedding(vocab_size,embedding_vecor_length, input_length=sequence_length, trainable=True)(visible)
e=Reshape((sequence_length,embedding_vecor_length,1))(embedding)
print(e.shape)
conv_0 = Conv2D(filters1, kernel_size=(filter_sizes[0], 100), padding='valid', kernel_initializer='normal', activation=newacti)(e)
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] +1, 1), strides=(1,1), padding='valid')(conv_0)
#maxpool_0=Flatten()(maxpool_0)
maxpool_0=Reshape((1,gru_output_size))(maxpool_0)
maxpool_0=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(maxpool_0)
maxpool_0=Reshape((1,gru_output_size))(maxpool_0)
#conv_1 = Conv2D(filters1, kernel_size=(filter_sizes[0], 100), padding='valid', kernel_initializer='normal', activation=newacti)(maxpool_0)

#maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] +1, 1), strides=(1,1), padding='valid')(conv_1)

gru=Reshape((sequence_length,embedding_vecor_length))(e)
gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
#gru=Reshape((1,gru_output_size))(gru)
#gru=RepeatVector(sequence_length)(gru)
gru=Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1)(gru)
gru=MaxPooling1D(pool_size=pool_size)(gru)
#gru=Reshape((1,1,gru_output_size))(gru)

merge = maximum([maxpool_0, gru])
flatten = Flatten()(merge)
dropout = Dropout(0.5)(flatten)
output = Dense(nclass, activation='softmax')(dropout)
model = Model(inputs=visible, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
print(model.summary())
model.fit(padded_docs,y_train, nb_epoch=3, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs])
print('Model built successfully...Please wait.....Evaluating......')
# Final evaluation of the model
scores = model.evaluate(tpadded_docs, y_test)
print("Loss: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))
