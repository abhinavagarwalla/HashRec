import numpy as np
# import preprocessor as p
import pickle
import spacy 

def load_train_data(file_emb, file_tag):
	fp = open(file_tag).readlines()
	fp = [i.strip('\r\n').split(',') for i in fp]
	return np.load('../data/'+file_emb), fp 

def load_val_data(filename):
	return np.load('../data/val/'+filename)

def load_test_data(filename):
	return np.load('../data/test/'+filename)

# def extract_tags_tweets(filename):
# 	print "Extracting tags, and cleaning tweets"
# 	f = open(filename).readlines()
# 	f = [i.split('\t')[1].decode('utf-8').strip() for i in f if i.startswith('W') and "#" in i]
# 	#print len(f), f[0:10]

# 	#tags = [[i[1:] for i in j.split() if i.startswith('#')] for j in f]
# 	#print len(tags), tags[0], tags[1]

# 	tags = []
# 	tweets = []
# 	for i in range(100000):#len(f)):
# 		#print f[i]
# 		try:
# 			tags.append([k.match for k in p.parse(f[i]).hashtags])
# 			tweets.append(p.clean(f[i]))
# 		except:
# 			pass
# 	filename = filename[:-3].split('/')[-1]
# 	np.save("../data/tags_"+filename+'npy', tags)
# 	np.save("../data/tweet_txt_"+filename+'npy', tweets)
# 	return np.asarray(tags), np.asarray(tweets)

def save_emb(filename):
	tweets = open(filename).readlines()
	tweets = [i.split('\r\n')[0].decode('utf-8') for i in tweets]
	print "Saving embeddings.."
	emb = []
	nlp = spacy.en.English()
	for i in tweets:
		words = i.split()
		emb.append([nlp(w).vector for w in words])
	filename = filename.split('/')[-1]
	np.save("../data/emb_"+filename[:-3]+'npy', emb)

filename = '../data/LSTM_train_tweets.txt'
#dat = open(filename, 'r')
#print dat.ix[:6]
# tags, tweets = extract_tags_tweets(filename)

# save_emb(filename)
