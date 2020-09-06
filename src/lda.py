import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

lines_gen = np.load("LSTM_train_tweets_cleaned.npy")
n_topics=100

#Function to transform the documents to term-document matrix
def tf_transform(doc_complete):
    doc_clean = doc_complete                                                 
    tf_vectorizer = CountVectorizer()                                        #Convert a collection of text documents to a matrix of token counts
    tf = tf_vectorizer.fit_transform(doc_clean)                              #Learn the vocabulary dictionary and return term-document matrix.
    return tf

#Function to train LDA model
def LDA_train(doc_complete):
    tf=tf_transform(doc_complete)                                            
    ldamodel = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,      #Setting up the parameters for our LDA model 
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    ldamodel.fit(tf)                                                         #Fitting the LDA model
    #a = ldamodel.transform(tf)                                              #prints the result of prediction
    #print(ldamodel.transform(tf))                                           #prints the result of prediction
    pickle.dump(ldamodel,open("trainedLDA.pkl","wb"))                        #Using serialisation to save our fitted model ( here Pickle dump)

LDA_train(lines_gen)