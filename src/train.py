import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from preprocessing import load_train_data, load_test_data, load_val_data
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

maxlen = 30
batch_size = 64
wordvec_size = 300
hidden_states = 100
nb_epoch = 50

X_train, y_train = load_train_data('../data/LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
#X_val, y_val = load_val_data()
#X_test, y_test = load_test_data()

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#print X_train.shape
#exit()

#y_train = [y_train[i%5][0] for i in range(len(y_train))]
lb = preprocessing.MultiLabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
output_tags = len(lb.classes_)

X_val, y_val = X_train[:100], y_train[:100]
X_test, y_test = X_train[:100], y_train[:100]
#exit()

def build_model():
	model = Sequential()
	model.add(LSTM(hidden_states, return_sequences=False, input_shape=(maxlen, wordvec_size)))
	model.add(Dense(output_tags))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])#, 'precision', 'recall', 'fmeasure'])
	print model.summary()
	return model

def train():
	model = build_model()
	model.load_weights('simple_model.h5')
	print "Fitting model.."
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
	# print model.predict(X_train)
	print model.evaluate(X_test, y_test, batch_size=batch_size)
	model.save_weights('simple_model.h5')
	'''
	score, acc, pr, re, fm = model.evaluate(X_test, y_test, batch_size=batch_size)
	print "Model Performance Measures: "
	print "Loss: ", score
	print "Accuracy: ", acc
	print "Precision: ", pr
	print "Recall: ", re
	print "F1: ", fm
	'''

train()

def eval():
	model = build_model()
	model.load_weights('simple_model.h5')
	X, y = 
	print model.evaluate(X, y, batch_size=batch_size)