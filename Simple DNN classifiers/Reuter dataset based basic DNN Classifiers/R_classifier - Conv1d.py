from numpy import asarray
from numpy import zeros
import pandas as pd
import os
from keras.datasets import reuters
from keras.preprocessing import sequence
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

batch_size=32
embedding_size=128
nclass=5

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
top_words=500
nclass=46
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

y_train=out_train
y_test=out_test
y_valid=out_val
padded_docs=X_train
vpadded_docs=X_val
tpadded_docs=X_test

# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(nclass, activation='softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
# summarize the model
print(model.summary())
# fit the model

model.fit(X_train, out_train, nb_epoch=5, batch_size=64, validation_data=(X_val,out_val), callbacks=[metriczs])

# evaluate the model
scores = model.evaluate(X_test, out_test, verbose=0)
print("Loss: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))

