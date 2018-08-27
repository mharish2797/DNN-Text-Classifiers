# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:54:07 2018

@author: HP
"""

# LSTM for sequence classification in the Reuter dataset
from numpy import asarray
from numpy import zeros
import numpy as np
from keras.datasets import reuters
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, LSTM ,Dropout,GRU, Bidirectional
from keras.layers import Embedding
from collections import defaultdict
from nltk.corpus import brown,stopwords
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as k
from keras.utils.generic_utils import get_custom_objects
import random
import nltk
import math as m
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 500

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

# GRU
gru_output_size = 70
#LSTM
lstm_output_size = 70



def zens(x, alpha=m.exp(-1)):
    return k.elu(x,alpha)


#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=top_words,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

X_val=X_test[0:1123]
X_test=X_test[1123:2246]
#one hot encoding
i=0
out_train=np.zeros((len(y_train),max(y_train)+1))
for x in y_train:
 out_train[i][x]=1
 i=i+1

i=0
out_test=np.zeros((len(y_test),max(y_test)+1))
for x in y_test:
 out_test[i][x]=1
 i=i+1

out_val=out_test[0:1123]
out_test=out_test[1123:2246]
# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
#model.add(LSTM(100))
model.add(Conv1D(64,kernel_size,padding='valid',activation=zens,strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(512,kernel_size,padding='valid',activation=zens,strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Bidirectional(GRU(gru_output_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(gru_output_size)))
model.add(Dense(46, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, out_train, nb_epoch=3, batch_size=64, validation_data=(X_val,out_val))
# Final evaluation of the model
scores = model.evaluate(X_test, out_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#model saving
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")