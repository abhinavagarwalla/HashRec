import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from preprocessing import load_train_data, load_test_data, load_val_data
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

maxlen = 30
batch_size = 64
wordvec_size = 300
hidden_states = 100
nb_epoch = 35
load = False

X_train, y_train = load_train_data('../data/emb_LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

lb = preprocessing.MultiLabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
output_tags = len(lb.classes_)

X_val, y_val = X_train[:100], y_train[:100]
X_test, y_test = X_train[:100], y_train[:100]


def build_model():
	model = Sequential()
	model.add(LSTM(hidden_states, return_sequences=False, input_shape=(maxlen, wordvec_size)))
	model.add(Dense(output_tags))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print model.summary()
	return model

def train():
	model = build_model()
	if load:
    	model.load_weights('simple_model.h5')
	print "Fitting model.."
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
	# print model.predict(X_train)
	# print model.evaluate(X_test, y_test, batch_size=batch_size)
	model.save_weights('simple_model.h5')

def evalu():
    model = build_model()
    model.load_weights('simple_model.h5')
    X, y_true = load_train_data('../data/emb_LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
    X = sequence.pad_sequences(X, maxlen=maxlen)
    y_true = lb.transform(y_true)
    y_pred = model.predict(X, batch_size=batch_size)
    print "Model Performance Measures: "
    print "Loss: ", log_loss(y_true, y_pred)
    ytmp = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    # y_pred = y_pred.argmax(axis=1)
    for i in range(len(ytmp)):
        ytmp[i, y_pred[i].argmax()] = 1
    y_pred = ytmp
    print "Accuracy: ", accuracy_score(y_true, y_pred)
    print "Precision: ", precision_score(y_true, y_pred, average='micro')
    print "Recall: ", recall_score(y_true, y_pred, average='micro')
    print "F1: ", f1_score(y_true, y_pred, average='micro')

train()
# evalu()