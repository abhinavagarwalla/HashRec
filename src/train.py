import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from preprocessing import load_train_data, load_test_data, load_val_data
from sklearn import preprocessing

maxlen = 30
batch_size = 64
wordvec_size = 300
hidden_states = 100
nb_epoch = 10

X_train, y_train = load_train_data('emb_tweets2009-06.npy', 'tags_tweets2009-06.npy')
#X_val, y_val = load_val_data()
#X_test, y_test = load_test_data()

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

#y_train = [y_train[i%5][0] for i in range(len(y_train))]
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
output_tags = len(lb.classes_)
print X_train.shape, y_train.shape

X_val, y_val = X_train.copy(), y_train.copy()
X_test, y_test = X_train.copy(), y_train.copy()
#exit()

def build_model():
	model = Sequential()
	model.add(LSTM(hidden_states, return_sequences=False, input_shape=(maxlen, wordvec_size)))
	model.add(Dense(output_tags))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
	print model.summary()
	return model

def train():
	model = build_model()
	print "Fitting model.."
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
	print model.predict(X_train)
	print model.evaluate(X_test, y_test, batch_size=batch_size)
	score, acc, pr, re, fm = model.evaluate(X_test, y_test, batch_size=batch_size)
	print "Model Performance Measures: "
	print "Loss: ", score
	print "Accuracy: ", acc
	print "Precision: ", pr
	print "Recall: ", re
	print "F1: ", fm

train()
