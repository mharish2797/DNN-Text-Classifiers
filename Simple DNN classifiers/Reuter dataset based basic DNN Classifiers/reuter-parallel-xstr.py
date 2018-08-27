import numpy as np
from keras.datasets import reuters
from keras.layers import Dense, Dropout ,Reshape
from keras.models import Sequential,Model
from keras.layers import LSTM, Flatten, Bidirectional,GRU,MaxPool2D,Conv1D,MaxPooling1D,Input,maximum,RepeatVector
from keras.layers.convolutional import Conv2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#Custom Activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import math as m
# fix random seed for reproducibility
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

sequence_length=500
filter_sizes = [3,4,5]

# GRU
gru_output_size = 64
#LSTM
lstm_output_size = 70

def newacti( x,alpha=m.exp(-1) ):
  return K.elu(x,alpha)

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


# create the model
embedding_vecor_length = 100
visible = Input(shape=(sequence_length,), dtype='int32')
# first feature extractor
embedding = Embedding(top_words,embedding_vecor_length, input_length=max_review_length, trainable=True)(visible)
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
output = Dense(46, activation='softmax')(dropout)
model = Model(inputs=visible, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, out_train, nb_epoch=3, batch_size=64)
print('Model built successfully...Please wait.....Evaluating......')
# Final evaluation of the model
scores = model.evaluate(X_test, out_test, verbose=0)
print("Loss: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))