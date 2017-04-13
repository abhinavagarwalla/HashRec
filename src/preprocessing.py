import numpy as np
import preprocessor as p
import pickle
import spacy 

def load_train_data(file_emb, file_tag):
	return np.load(file_emb), np.load(file_tag) 

def load_val_data(filename):
	return np.load('../data/val/'+filename)

def load_test_data(filename):
	return np.load('../data/test/'+filename)

def extract_tags_tweets(filename):
	print "Extracting tags, and cleaning tweets"
	f = open(filename).readlines()
	f = [i.split('\t')[1].decode('utf-8').strip() for i in f if i.startswith('W') and "#" in i]

	tags = []
	tweets = []
	for i in range(100000):#len(f)):
		try:
			tags.append([k.match for k in p.parse(f[i]).hashtags])
			tweets.append(p.clean(f[i]))
		except:
			pass
	filename = filename[:-3].split('/')[-1]
	np.save("../data/tags_"+filename+'npy', tags)
	np.save("../data/tweet_txt_"+filename+'npy', tweets)
	return np.asarray(tags), np.asarray(tweets)

def save_emb(filename):
	tweets = np.load(filename)
	print "Saving embeddings.."
	emb = []
	nlp = spacy.en.English()
	for i in tweets:
		print i
		emb.append([nlp(w.decode('utf-8')).vector for w in i.split()])
	filename = filename.split('/')[-1]
	np.save("../data/emb_"+filename[:-3]+'npy', emb)

filename = '../data/LSTM_test_tweets_cleaned_10k.npy'
#dat = open(filename, 'r')
#print dat.ix[:6]
# tags, tweets = extract_tags_tweets(filename)

# save_emb(filename)

filetw = '../data/LSTM_test_tweets.txt'
fileha = '../data/LSTM_test_hashtags.txt'

def getfreq(tw, ha):
	trh = open(ha).readlines()
	trh = [i.strip().split(',') for i in trh]
	trt = open(tw).readlines()
	trt = [i.strip() for i in trt]
	freq = np.load('../data/Freq_Tweets_10k.npy')

	trtclean, trhclean = [], []
	for i in range(len(trh)):
		print len(trh[i])
		flag = False
		if len(trh[i]) > 1:
			for j in range(len(trh[i])):
				if trh[i][j] not in freq:
					flag = True
					break
			if not flag:
				trtclean.append(trt[i])
				trhclean.append(trh[i])
			flag = False
		else:
			if trh[i] in freq:
				# print "Yes2"
				trtclean.append(trt[i])
				trhclean.append(trh[i])
	trtclean = np.asarray(trtclean)
	trhclean = np.asarray(trhclean)
	print trtclean.shape, trhclean.shape
	np.save('../data/LSTM_test_tweets_cleaned_10k', trtclean)
	np.save('../data/LSTM_test_hashtags_cleaned_10k', trhclean)

# getfreq(filetw, fileha)
