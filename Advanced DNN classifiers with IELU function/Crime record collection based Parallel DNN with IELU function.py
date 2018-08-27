from numpy import asarray
from numpy import zeros
import pandas as pd
import os
#from keras.utils.vis_utils import #plot_model
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
from keras.layers import *
from keras.models import *
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPool2D
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
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
import more_itertools as mit
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000

batch_size=30
embedding_size=128

# Convolution
kernel_size = 5
filters1 = 64
filters2 =128
filters3=256
filters4=512
filters5=1024
pool_size = 4
nclass=46
sequence_length=500
filter_sizes = [3,4,5]

# GRU
gru_output_size = 64
#LSTM
lstm_output_size = 70

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
     log.write(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
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
        ax2.legend();
        
       
        
plot = PlotLearning()
# this is how you load the model
model = KeyedVectors.load_word2vec_format("G:\\NLP\\Dataset\\GoogleNews-vectors-negative300.bin", binary=True, limit=100000)
#model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=100000)

batch_size=64
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
gru_output_size = 64
#LSTM
lstm_output_size = 64

trim_len=15
sample_cnt=600

os.chdir("G:/NLP/Dataset/Crime");
df=pd.read_csv('data.csv');

#input
ip=df['Description'].values.tolist()
ip=ip[0:sample_cnt]

for ty in range(len(ip)):
    ip[ty]=ip[ty][0:trim_len]

len_finder=[]
for dat in ip:
 len_finder.append(len(dat))

#output
op=df['Category'].values.tolist()
op=op[0:sample_cnt]

labels=[]
for zen in op:
    if zen not in labels:
        labels.append(zen)
nclass=len(labels)
label_class=[]
for ix in op:
 label_class.append(labels.index(ix))

def newacti( x,alpha=m.exp(-1) ):
  return K.elu(x,alpha)

sp=stopwords.words("english")

punct=string.punctuation

izp=[]
for sample in ip:
    temp=sample.lower()
    for x in punct:
        temp=temp.replace(x," ")
    izp.append(temp.split())

ip=izp
izp=[]
for sample in ip:
    dats=[]
    for i in sample:
        if i not in sp:
            dats.append(i)
    izp.append(dats)

inputset=izp

len_finder=[]
for dat in inputset:
 len_finder.append(len(dat))

max_pad=trim_len

izp=[]
for sample in inputset:
    izp.append(list(mit.padded(sample,"0", max_pad)))

inputset=izp

count=trim_len
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
y_train=np.zeros((len(label_train),nclass))
for x in label_train:
    y_train[i][x]=1
    i=i+1

i=0
y_test=np.zeros((len(label_test),nclass))
for x in label_test:
    y_test[i][x]=1
    i=i+1

i=0
y_valid=np.zeros((len(label_valid),nclass))
for x in label_valid:
    y_valid[i][x]=1
    i=i+1


filter_sizes = [3,4,5]
sequence_length=count
embedding_vecor_length = 300
visible = Input(shape=(sequence_length,300), dtype='float32')
print(visible.shape)
e=visible

# first feature extractor


def parallel_straight_relu():
    log.write("\nparallel-straight-relu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='relu')(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)

    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=GRU(gru_output_size)(gru)
    gru=Reshape((1,1,gru_output_size))(gru)

    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-straight-relu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel_s.png', show_shapes=True, show_layer_names=True)


def parallel_straight_elu():
    log.write("\nparallel-straight-elu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='elu')(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)

    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=GRU(gru_output_size)(gru)
    gru=Reshape((1,1,gru_output_size))(gru)

    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-straight-elu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel_s.png', show_shapes=True, show_layer_names=True)

def parallel_straight_leakyrelu():
    log.write("\nparallel-straight-leakyrelu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=LeakyReLU(alpha=.001))(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)


    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=GRU(gru_output_size)(gru)
    gru=Reshape((1,1,gru_output_size))(gru)

    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    ##plot_model(model, to_file='model_plot_parallel.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-straight-leakyrelu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    ##plot_model(model, to_file='model_plot_parallel_s.png', show_shapes=True, show_layer_names=True)


def parallel_straight_newactivation():
    log.write("\nparallel-straight-newactivation\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=newacti)(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)

    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=GRU(gru_output_size)(gru)
    gru=Reshape((1,1,gru_output_size))(gru)

    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-straight-newactivation-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel_s.png', show_shapes=True, show_layer_names=True)

def parallel_cross_relu():
    log.write("\nparallel-cross-relu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='relu')(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)
    maxpool_0=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(maxpool_0)
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
    dropout = Dropout(0.25)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel_cross.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-relu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def parallel_cross_elu():
    log.write("\nparallel-cross-elu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='elu')(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)
    maxpool_0=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(maxpool_0)
    
    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=Conv1D(64,kernel_size,padding='valid',activation='elu',strides=1)(gru)
    gru=MaxPooling1D(pool_size=pool_size)(gru)
    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel_cross.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-elu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def parallel_cross_leakyrelu():
    log.write("\nparallel-cross-leakyrelu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=LeakyReLU(alpha=.001))(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)
    maxpool_0=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(maxpool_0)
    
    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=Conv1D(64,kernel_size,padding='valid',activation=LeakyReLU(alpha=.001),strides=1)(gru)
    gru=MaxPooling1D(pool_size=pool_size)(gru)
    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel_cross.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-leakyrelu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def parallel_cross_newactivation():
    log.write("\nparallel-cross-newactivation\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=newacti)(e)
    maxpool_0 = MaxPooling1D(pool_size=pool_size)(conv_0)
    maxpool_0=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(maxpool_0)
    
    gru=Reshape((sequence_length,embedding_vecor_length))(e)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(gru)
    gru=Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1)(gru)
    gru=MaxPooling1D(pool_size=pool_size)(gru)
    merge = maximum([maxpool_0, gru])
    flatten = Flatten()(merge)
    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_plot_parallel_cross.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-newactivation-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def parallel_cross_straight_relu():
    log.write("\nparallel-straight-cross-relu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='relu')(e)
    maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(e)
    maxpool_gru = MaxPooling1D(pool_size=3)(gru)
    merge2 = maximum([maxpool_0, maxpool_gru])
    #merge=Reshape((sequence_length,filters1))(merge2)
    gru1=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(merge2)
    #gru1=MaxPooling1D(pool_size=8)(gru1)
    conv_1=Conv1D(filters1,kernel_size,padding='valid',activation='relu',strides=1)(merge2)
    #maxpool_1=MaxPooling1D(pool_size=8)(conv_1)
    #merge1 = maximum([gru1,maxpool_1])
    merge1 = maximum([gru1,conv_1])
    #merge1=MaxPooling1D(pool_size=8)(merge1)
    flatten = Flatten()(merge1)

    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-straight-relu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel-3.png', show_shapes=True, show_layer_names=True)


def parallel_cross_straight_elu():
    log.write("\nparallel-straight-cross-elu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation='elu')(e)
    maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(e)
    maxpool_gru = MaxPooling1D(pool_size=3)(gru)
    merge2 = maximum([maxpool_0, maxpool_gru])
    #merge=Reshape((sequence_length,filters1))(merge2)
    gru1=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(merge2)
    #gru1=MaxPooling1D(pool_size=8)(gru1)
    conv_1=Conv1D(filters1,kernel_size,padding='valid',activation='elu',strides=1)(merge2)
    #maxpool_1=MaxPooling1D(pool_size=8)(conv_1)
    #merge1 = maximum([gru1,maxpool_1])
    merge1 = maximum([gru1,conv_1])
    #merge1=MaxPooling1D(pool_size=8)(merge1)
    flatten = Flatten()(merge1)

    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-straight-elu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel-3.png', show_shapes=True, show_layer_names=True)

def parallel_cross_straight_leakyrelu():
    log.write("\nparallel-straight-cross-leakyrelu\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=LeakyReLU(alpha=0.01))(e)
    maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(e)
    maxpool_gru = MaxPooling1D(pool_size=3)(gru)
    merge2 = maximum([maxpool_0, maxpool_gru])
    #merge=Reshape((sequence_length,filters1))(merge2)
    gru1=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(merge2)
    #gru1=MaxPooling1D(pool_size=8)(gru1)
    conv_1=Conv1D(filters1,kernel_size,padding='valid',activation=LeakyReLU(alpha=0.01),strides=1)(merge2)
    #maxpool_1=MaxPooling1D(pool_size=8)(conv_1)
    #merge1 = maximum([gru1,maxpool_1])
    merge1 = maximum([gru1,conv_1])
    #merge1=MaxPooling1D(pool_size=8)(merge1)
    flatten = Flatten()(merge1)

    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-straight-leakyrelu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel-3.png', show_shapes=True, show_layer_names=True)

def parallel_cross_straight_newactivation():
    log.write("\nparallel-straight-cross-newactivation\n")
    conv_0 = Conv1D(64,kernel_size,padding='valid',strides=1, activation=newacti)(e)
    maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
    gru=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(e)
    maxpool_gru = MaxPooling1D(pool_size=3)(gru)
    
    merge2 = maximum([maxpool_0, maxpool_gru])
    #merge=Reshape((sequence_length,filters1))(merge2)

    gru1=GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(merge2)
    #gru1=MaxPooling1D(pool_size=8)(gru1)


    conv_1=Conv1D(filters1,kernel_size,padding='valid',activation=newacti,strides=1)(merge2)
    #maxpool_1=MaxPooling1D(pool_size=8)(conv_1)

    #merge1 = maximum([gru1,maxpool_1])
    merge1 = maximum([gru1,conv_1])
    #merge1=MaxPooling1D(pool_size=8)(merge1)
    flatten = Flatten()(merge1)

    dropout = Dropout(0.5)(flatten)
    output = Dense(nclass, activation='softmax')(dropout)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=64, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nparallel-cross-straight-newactivation-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))
    #plot_model(model, to_file='model_plot_parallel-3.png', show_shapes=True, show_layer_names=True)

def sequence_leakyrelu():      
    log.write("\nparallel-sequence-leakyrelu\n")
    model = Sequential()
    model.add(Conv1D(64,kernel_size,padding='valid',activation=LeakyReLU(alpha=0.01),strides=1, input_shape=(count,300)))
    #model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(64,kernel_size,padding='valid',activation=LeakyReLU(alpha=0.01),strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(RepeatVector(1))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_TC_sequence.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=128, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nSequence-leakyrelu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def sequence_elu():
    log.write("\nparallel-sequence-elu\n")
    model = Sequential()
    model.add(Conv1D(64,kernel_size,padding='valid',activation='elu',strides=1, input_shape=(count,300)))
    #model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(64,kernel_size,padding='valid',activation='elu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(RepeatVector(1))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_TC_sequence.png', show_shapes=True, show_layer_names=True)
    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=128, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nSequence-elu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))

def sequence_relu():
    log.write("\nparallel-sequence-relu\n")
    model = Sequential()
    model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1, input_shape=(count,300)))
    #model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(RepeatVector(1))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model_TC_sequence.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=128, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nSequence-relu-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))


def sequence_newactivation():
    log.write("\nparallel-sequence-newactivation\n")
    model = Sequential()
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1, input_shape=(count,300)))
    #model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(RepeatVector(1))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])
    print(model.summary())
   #plot_model(model, to_file='model_TC_sequence.png', show_shapes=True, show_layer_names=True)

    model.fit(padded_docs,y_train, nb_epoch=epochesno, batch_size=128, validation_data=(vpadded_docs, y_valid), callbacks=[metriczs, plot])
    print('Model built successfully...Please wait.....Evaluating......')
    # Final evaluation of the model
    scores = model.evaluate(tpadded_docs, y_test)
    print("Loss: %.2f%%" % (scores[0]*100))
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write('\nSequence-newactivation-accuracy -\n '+str(scores[1]*100)+"\t Loss: "+str(scores[0]*100))


log = open("Crime-Accuracy.txt", "w+")

epochesno=2
architect=['sequence_leakyrelu', 'sequence_elu', 'sequence_relu', 'sequence_newactivation', 'parallel_straight_leakyrelu', 'parallel_straight_elu', 'parallel_straight_relu', 'parallel_straight_newactivation', 'parallel_cross_leakyrelu', 'parallel_cross_elu', 'parallel_cross_relu', 'parallel_cross_newactivation', 'parallel_cross_straight_leakyrelu','parallel_cross_straight_elu', 'parallel_cross_straight_relu', 'parallel_cross_straight_newactivation']
i=0

sequence_leakyrelu()
text='Crime'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

sequence_elu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

sequence_relu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

sequence_newactivation()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_straight_leakyrelu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_straight_elu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_straight_relu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_straight_newactivation()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_leakyrelu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_elu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_relu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_newactivation()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_straight_leakyrelu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_straight_elu()   
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_straight_relu()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)

parallel_cross_straight_newactivation()
text='Crime_'+architect[i]+'_.png'
i=i+1
plt.savefig(text)
