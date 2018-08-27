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
from keras import metrics
from keras.utils.generic_utils import get_custom_objects
import math as m
from keras.models import load_model
from matplotlib import pyplot as plt
from IPython.display import clear_output
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

def newacti( x,alpha=1.618 ):
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

out_val=out_test[0:1123]
out_test=out_test[1123:2246]

X_val=X_test[0:1123]
X_test=X_test[1123:2246]


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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['top_k_categorical_accuracy'])


#model1 = load_model('model.h5')
print(model.summary())
hist=model.fit(X_train, out_train, nb_epoch=5, batch_size=64, validation_data=(X_val,out_val), callbacks=[metriczs, plot])
print('Model built successfully...Please wait.....Evaluating......')
# Final evaluation of the model
scores = model.evaluate(X_test, out_test, verbose=0)
print("Loss: %.2f%%" % (scores[0]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))



#model saving
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
## serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")
