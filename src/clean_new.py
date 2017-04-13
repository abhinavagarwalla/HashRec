from langdetect import detect
import py2casefold as p2c
import preprocessor as p
import enchant
from enchant.checker import SpellChecker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance

class MySpellChecker():

    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        suggestions = self.spell_dict.suggest(word)

        if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]
        return word

wlm = WordNetLemmatizer()
msc = MySpellChecker(max_dist=2)
stop = set(stopwords.words('english'))

print "Reading Tweets File..."
file1=open("../data/tweets2009-08.txt","r")
#file2=open("../data/output.txt","w")
#file3=open("../data/hash.txt","w")

print "Starting Loop..."
for line in file1:
    if line.startswith("W"):
        s=line.split("\t")[1].strip()
        if not("No Post Title" == s):
            if ("#" in s):
                t=s.decode("utf-8")
                print "."
		try:
			if detect(t)=="en":
				#print "English detected..."
                        	ss=p2c.casefold(t)
                        	hashtags=p.parse(ss).hashtags
                        	tweet=p.clean(ss)
                       
				#print "Beginnig spell correct..." 
                        	chkr = SpellChecker("en_US", tweet)
                        	for err in chkr:
                            		err.replace(msc.replace(err.word))
                        	tweet2 = chkr.get_text()
                        	tweet2 = p2c.casefold(tweet2)
                        	
				#print "Tokenizing"
                        	tokens = nltk.word_tokenize(tweet2)
	                        #words=[wlm.lemmatize(w) for w in tokens if w.isalpha() and w not in stop]
				#print "Tokenized"
				words=[w for w in tokens if w.isalpha() and w not in stop]
                	        tweet3=" ".join(words)
                        	
				#print line
				file2=open("../data/output.txt","a")
				file3=open("../data/hash.txt","a")
	                        file3.write(",".join([x.match for x in hashtags]))
        	                file3.write("\n")
                	        file2.write(tweet3)
                        	file2.write("\n")
				file2.close()
				file3.close()
		except:
                        pass
                    

##read a file and break into a smaller chunk###
#==============================================================================
# file1=open("tweets2009-06.txt","r")
# data=''.join([next(file1) for x in xrange(4000)])
# file2=open("sample.txt","w")
# file2.write(data)
# file2.close()
#==============================================================================
