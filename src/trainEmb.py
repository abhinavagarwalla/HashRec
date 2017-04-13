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
nb_epoch = 10
max_features = 108607 # if 450000

def load_transform(file_tw, file_tag):
	td = np.load('../data/tweet_dict.npy')
	tw = np.load(file_tw)
	tag = np.load(file_tag)

	tw = [i.split(' ') for i in tw]
	for i in range(len(tw)):
		for j in range(len(tw[i])):
			tw[i][j] = np.where(td==tw[i][j])[0][0]
	np.save('../data/transformed_tweets', tw)
	exit()
	return tw, tag

X_train, y_train = load_transform('../data/LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
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

X_val, y_val = X_train, y_train
X_test, y_test = X_train, y_train

def build_model():
	model = Sequential()
	model.add(Embedding(max_features, hidden_states))
	model.add(LSTM(hidden_states, return_sequences=False, input_shape=(maxlen, wordvec_size)))
	model.add(Dense(output_tags))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print model.summary()
	return model

def train():
	model = build_model()
	print "Fitting model.."
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
	# print model.predict(X_train)
	print model.evaluate(X_test, y_test, batch_size=batch_size)
	model.save_weights('simple_model_emb.h5')

train()