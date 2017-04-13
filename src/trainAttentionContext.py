import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, Adadelta
from preprocessing import load_train_data, load_test_data, load_val_data
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers.core import Layer  
from keras import initializations, regularizers, constraints  
from keras import backend as K

maxlen = 30
batch_size = 64
wordvec_size = 300
hidden_states = 100
nb_epoch = 20
# output_tags = 100

# def load_data():
# X_train, y_train = load_train_data('../data/emb_LSTM_train_tweets_cleaned.npy', '../data/LSTM_train_hashtags_cleaned.npy')
X_train, y_train = load_train_data('../data/emb_LSTM_train_tweets_cleaned_10k.npy', '../data/LSTM_train_hashtags_cleaned_10k.npy')
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

X_val, y_val = X_train[:200], y_train[:200]
X_test, y_test = X_train[:200], y_train[:200]
# return X_train, y_train, X_val, y_val, X_test, y_test, output_tags

class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model():
    model = Sequential()
    model.add(LSTM(hidden_states, return_sequences=True, input_shape=(maxlen, wordvec_size)))
    model.add(AttentionWithContext())
    model.add(Dense(output_tags))
    model.add(Activation('softmax'))

    adam = RMSprop(lr = 0.009)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])#, 'precision', 'recall', 'fmeasure'])
    print model.summary()
    return model

def train():
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = build_model()
    model.load_weights('simple_model_att_LDA.h5')
    print "Fitting model.."
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_val, y_val), verbose=2)
    # print model.predict(X_train)
    print model.evaluate(X_test, y_test, batch_size=batch_size)
    model.save_weights('simple_model_att_LDA.h5')
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
