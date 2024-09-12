import nltk.stem
from pandas import read_csv
from nltk.corpus import stopwords
import numpy

from nltk.tokenize import word_tokenize
import re

numpy.random.seed(1222)
df = read_csv("combined_data.csv", encoding="latin-1")
df['text_final'] = [x.lower() for x in df['text']]
df['text_final'] = [x.strip() for x in df['text_final']]
df['text_final'] = [re.sub("[\d]", "", x) for x in df['text_final']]
df['text_final'] = [re.sub("[^\w\s]", "", x) for x in df['text_final']]
df['text_final'] = [re.sub("[^\x00-\x7F]", "", x) for x in df['text_final']]
df['text_final'] = [re.sub(r"\b\w*escapenumber\w*\b", "", x) for x in df['text_final']]
df['text_final'] = [re.sub(r"\b\w*escapelong\w*\b", "", x) for x in df['text_final']]
df['text_final'] = [word_tokenize(x) for x in df['text_final']]
def get_pos(tag):
    if tag.startswith('J'):
        return nltk.stem.wordnet.wn.ADJ
    elif tag.startswith('V'):
        return nltk.stem.wordnet.wn.VERB
    elif tag.startswith('N'):
        return nltk.stem.wordnet.wn.NOUN
    elif tag.startswith('R'):
        return nltk.stem.wordnet.wn.ADV
    else:
        return nltk.stem.wordnet.wn.NOUN

for idx, token in enumerate(df['text_final']):
    finalized = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for word, tag in nltk.pos_tag(token):
        if word not in stopwords.words('english') and word.isalpha():
            word_finalized = lemmatizer.lemmatize(word, get_pos(tag))
            finalized.append(word_finalized)
    df.loc[idx, 'text_final'] = str(finalized)
df.to_csv("final.csv")