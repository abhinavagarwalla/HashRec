import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from preprocessing import load_train_data, load_test_data, load_val_data
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers.core import Layer  
from keras import initializations, regularizers, constraints  
from keras import backend as K
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

maxlen = 30
batch_size = 64
wordvec_size = 300
hidden_states = 100
nb_epoch = 20

X_train, y_train = load_train_data('../data/emb_LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

lb = preprocessing.MultiLabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
output_tags = len(lb.classes_)

X_val, y_val = X_train[:200], y_train[:200]
X_test, y_test = X_train[:200], y_train[:200]

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = K.dot(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model():
    model = Sequential()
    model.add(LSTM(hidden_states, return_sequences=True, input_shape=(maxlen, wordvec_size)))
    model.add(Attention())
    model.add(Dense(output_tags))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])#, 'precision', 'recall', 'fmeasure'])
    print model.summary()
    return model

def train():
    model = build_model()
    model.load_weights('simple_model_att.h5')
    print "Fitting model.."
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
    # print model.evaluate(X_test, y_test, batch_size=batch_size)
    model.save_weights('simple_model_att.h5')


def softmax(x):
    b = np.sum(np.exp(x), axis=1) + 1e-8
    return np.exp(x)/b[:,None]

def evalu():
    model = build_model()
    model.load_weights('simple_model_att.h5')
    X, y_true = load_train_data('../data/emb_LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
    X = sequence.pad_sequences(X, maxlen=maxlen)
    y_true = lb.transform(y_true)
    y_pred = model.predict(X, batch_size=batch_size)
    print "Model Performance Measures: "
    print "Loss: ", log_loss(y_true, y_pred)
    ytmp = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for i in range(len(ytmp)):
        ytmp[i, y_pred[i].argmax()] = 1
    y_pred = ytmp
    print "Accuracy: ", accuracy_score(y_true, y_pred)
    print "Precision: ", precision_score(y_true, y_pred, average='micro')
    print "Recall: ", recall_score(y_true, y_pred, average='micro')
    print "F1: ", f1_score(y_true, y_pred, average='micro')

# train()
# evalu()