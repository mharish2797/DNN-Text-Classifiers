from numpy import asarray
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM, CuDNNLSTM, SimpleRNN
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
from keras.layers import Conv1D, MaxPooling1D
import random
import nltk
#Custom Activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import math as m
#brown.categories()
#brown.words(categories='news')
#brown.words(fileids=['cg22'])
#brown.sents(categories=['news', 'editorial', 'reviews'])

batch_size=64
embedding_size=128
#epoch_size=3
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

categ=brown.categories()

label_class=[]
for x in labels:
 label_class.append(categ.index(x))

len_finder=[]
for dat in inputset:
 len_finder.append(len(dat))

input_train=[]
j=0;
for zz in inputset:
    j=j+1
    if (j%5 is not 0):
        input_train.append(zz)


input_test=[]
j=0;
for zz in inputset:
    j=j+1
    if (j%5 is 0):
        input_test.append(zz)
        
label_train=[]
j=0;
for zz in label_class:
    j=j+1
    if (j%5 is not 0):
        label_train.append(zz)
        
label_test=[]
j=0;
for zz in label_class:
    j=j+1
    if (j%5 is 0):
        label_test.append(zz)
        

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


#Training Data
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

#Testing data
#Testing the model
tt = Tokenizer()
tt.fit_on_texts(input_test)
tvocab_size = len(tt.word_index) + 1
# integer encode the documents
tencoded_docs = tt.texts_to_sequences(input_test)
tpadded_docs = pad_sequences(tencoded_docs, maxlen=max_length, padding='post')

#Data part ends here

accurator=[]
m_col=[]

def simplelstm(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(LSTM(lstm_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_lstm.png', show_shapes=True, show_layer_names=True)
    log.write('\nsimplelstm - \n'+str(accuracy))

def simplegru(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_gru.png', show_shapes=True, show_layer_names=True)
    log.write('\nsimpleGRU -\n '+str(accuracy))

def simplernn(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(SimpleRNN(lstm_output_size,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_rnn.png', show_shapes=True, show_layer_names=True)
    log.write('\nsimpleRNN -\n '+str(accuracy))

def simpleconv1d(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_conv.png', show_shapes=True, show_layer_names=True)
    log.write('\nConvolutional -\n '+str(accuracy))

def activeconv1d(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_conv.png', show_shapes=True, show_layer_names=True)
    log.write('\nConvolutional -\n '+str(accuracy))


def simplebidirectionalLSTM(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(70, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    m_col.append(model)
    plot_model(model, to_file='model_plot_biLSTM.png', show_shapes=True, show_layer_names=True)
    log.write('\nBidirectionalLSTM -\n '+str(accuracy))

def simplebidirectionalGRU(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(GRU(70, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    m_col.append(model)
    plot_model(model, to_file='model_plot_biGRU.png', show_shapes=True, show_layer_names=True)
    log.write('\nBidirectionalGRU -\n '+str(accuracy))

def seq1(epoch_size):
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
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_seq1.png', show_shapes=True, show_layer_names=True)
    log.write('\nSequence 1 -\n '+str(accuracy))

def activeseq1(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_aseq1.png', show_shapes=True, show_layer_names=True)
    log.write('\na,Sequence 1 -\n '+str(accuracy))

def seq2(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_seq2.png', show_shapes=True, show_layer_names=True)
    log.write('\nSequence 2 -\n '+str(accuracy))

def activeseq2(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_aseq2.png', show_shapes=True, show_layer_names=True)
    log.write('\naSequence 2 -\n '+str(accuracy))

def seq3(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(256,kernel_size,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_seq3.png', show_shapes=True, show_layer_names=True)
    log.write('\nSequence 3 -\n '+str(accuracy))

def activeseq3(epoch_size):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(64,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(256,kernel_size,padding='valid',activation=newacti,strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2)))
    #model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dense(nclass, activation='softmax'))
# compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
    print(model.summary())
# fit the model
    model.fit(padded_docs,y_train, epochs=epoch_size, verbose=0)
    loss, accuracy = model.evaluate(tpadded_docs, y_test, verbose=0)
    plot_model(model, to_file='model_plot_aseq3.png', show_shapes=True, show_layer_names=True)
    log.write('\naSequence 3 -\n '+str(accuracy))

#The runnable part

log = open("/home/teja/Desktop/Accuracy.txt", "w+")
epocher=[10,50,100,200,500]

for ech in epocher:
	log.write('\nEpoch Cycle Size-\n '+str(ech))	
	simplernn(ech)
	simplelstm(ech)
	simplegru(ech)
	simpleconv1d(ech)
	activeconv1d(ech)
	simplebidirectionalLSTM(ech)
	simplebidirectionalGRU(ech)
	seq1(ech)
	activeseq1(ech)
	seq2(ech)
	activeseq2(ech)
	seq3(ech)
	activeseq3(ech)