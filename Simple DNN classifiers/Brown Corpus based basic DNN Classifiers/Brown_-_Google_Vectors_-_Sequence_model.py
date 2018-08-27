from numpy import asarray
from numpy import zeros
import pandas as pd
import os
#from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Flatten,Input, LSTM
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
from matplotlib import pyplot as plt
from IPython.display import clear_output
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sys
import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import string
# this line doesn't load the trained model 
from gensim.models.keyedvectors import KeyedVectors

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

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('top_k_categorical_accuracy'))
        self.val_acc.append(logs.get('val_top_k_categorical_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
plot = PlotLearning()


# this is how you load the model
model = KeyedVectors.load_word2vec_format("G:\\NLP\\Dataset\\GoogleNews-vectors-negative300.bin", binary=True, limit=100000)

batch_size=32
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
gru_output_size = 64
#LSTM
lstm_output_size = 64

dataset = [] # 500 samples

def newacti( x,alpha=m.exp(-1) ):
  return K.elu(x,alpha)

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
 
sp=stopwords.words("english")

punct=string.punctuation

izp=[]
for sample in inputset:
    dats=[]
    for i in sample:
        if i not in sp and i not in punct:
            dats.append(i.lower())
    izp.append(dats)

inputset=izp

categ=brown.categories()

label_class=[]
for x in labels:
 label_class.append(categ.index(x))

len_finder=[]
for dat in inputset:
 len_finder.append(len(dat))

count=min(len_finder)
ip=[]
for sample in inputset:
    dats=[]
    for i in sample:
        if i in model.vocab.keys():
            dats.append(model[i]) 
        else:
            dats.append(model["0"])
    ip.append(dats)   
    
tmp=[]
for sample in ip:
    tmp.append(sample[:count])

ip=tmp
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

padded_docs=input_train
tpadded_docs=input_test
vpadded_docs=input_valid


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

filter_sizes = [3,4,5]
sequence_length=count
embedding_vecor_length = 100
model = Sequential()
#model.add(Input(shape=(sequence_length,300), dtype='float32'))
model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1, input_shape=(sequence_length,300)))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(512,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Bidirectional(GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(gru_output_size)))
model.add(Dense(nclass, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
print(model.summary())
#plot_model(model, to_file='model_brown_corpus_parallel_x.png', show_shapes=True, show_layer_names=True)
#model.fit(X_train, out_train, nb_epoch=5, batch_size=64, validation_data=(X_val,out_val), callbacks=[metriczs])
model.fit(padded_docs,y_train, nb_epoch=15, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
print('Model built successfully...Please wait.....Evaluating......')
# Final evaluation of the model
scores = model.evaluate(tpadded_docs, y_test)
print("Loss: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))

#model saving
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
## serialize weights to HDF5
model.save("model_bs.h5")
print("Saved model to disk")
