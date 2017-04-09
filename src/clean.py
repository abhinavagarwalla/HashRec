imporimport nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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
t nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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
import nltk
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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

def getTerms(sentences):
    tokens = nltk.word_tokenize(sentences)
    words = [wordnet_lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalpha() and w not in stop]
    return (" ").join(words)

def clean(text):
    tweet_content = getTerms(text)
    my_spell_checker = MySpellChecker(max_dist=2)
    chkr = SpellChecker("en_US", tweet_content)
    for err in chkr:
        err.replace(my_spell_checker.replace(err.word))
    t = chkr.get_text()
    if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]
        return word
import nltk
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
stop = set(stopwords.words('english'))
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

def getTerms(sentences):
    tokens = nltk.word_tokenize(sentences)
    words = [wordnet_lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalpha() and w not in stop]
    return (" ").join(words)

def clean(text):
    tweet_content = getTerms(text)
    my_spell_checker = MySpellChecker(max_dist=2)
    chkr = SpellChecker("en_US", tweet_content)
    for err in chkr:
        err.replace(my_spell_checker.replace(err.word))
    t = chkr.get_text()
    return t
return t
