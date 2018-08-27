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
model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
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


