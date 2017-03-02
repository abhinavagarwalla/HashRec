import numpy as np
import preprocessor as p
import pickle
import spacy 

def load_train_data(filename):
	return np.load('../data/train/'+filename)

def load_val_data(filename):
	return np.load('../data/val/'+filename)

def load_test_data(filename):
	return np.load('../data/test/'+filename)

def extract_tags_tweets(filename):
	print "Extracting tags, and cleaning tweets"
	f = open(filename).readlines()
	f = [i.split('\t')[1].decode('utf-8').strip() for i in f if i.startswith('W') and "#" in i]
	#print len(f), f[0:10]

	#tags = [[i[1:] for i in j.split() if i.startswith('#')] for j in f]
	#print len(tags), tags[0], tags[1]

	tags = []
	tweets = []
	for i in range(len(f)):
		#print f[i]
		try:
			tags.append([k.match for k in p.parse(f[i]).hashtags])
			tweets.append(p.clean(f[i]))
		except:
			pass
	np.save("../data/tags_"+filename+'npy', tags)
	np.save("../data/tweet_txt_"+filename+'npy', tweets)
	return np.asarray(tags), np.asarray(tweets)

def save_emb(tweets, filename):
	print "Saving embeddings.."
	emb = []
	nlp = spacy.en.English()
	for i in tweets:
		words = i.split()
		emb.append([nlp(w).vector for w in words])
	np.save("../data/"+filename+'npy', emb)

filename = '../data/tweets2009-06.txt'
tags, tweets = extract_tags_tweets(filename)
save_emb(tweets, filename[:-3])