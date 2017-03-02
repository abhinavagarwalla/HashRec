import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from preprocessing import load_train_data, load_test_data, load_val_data

maxlen = 30
batch_size = 32
wordvec_size = 300
output_tags = 100
nb_epoch = 10

X_train, y_train = load_train_data()
X_val, y_val = load_val_data()
X_test, y_test = load_test_data()

def build_model():
	model = Sequential()
	model.add(LSTM(wordvec))
	model.add(Dense(output_tags))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['precision', 'recall', 'fmeasure'])
	return model

def train():
	model = build_model()
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val))
	score, pr, re, fm = model.evaluate(X_test, y_test, batch_size=batch_size)
	print "Model Performance Measures: "
	print "Accuracy: ", score
	print "Precision: ", pr
	print "Recall: ", re
	print "F1: ", fm



