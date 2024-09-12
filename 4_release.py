import pandas as pd
import joblib as jb
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import tkinter as tk
def clean(df):
    df['input'] = [x.lower() for x in df['input']]
    df['input'] = [x.strip() for x in df['input']]
    df['input'] = [re.sub("[\d]", "", x) for x in df['input']]
    df['input'] = [re.sub("[^\w\s]", "", x) for x in df['input']]
    df['input'] = [re.sub("[^\x00-\x7F]", "", x) for x in df['input']]
    df['input'] = [re.sub(r"\b\w*escapenumber\w*\b", "", x) for x in df['input']]
    df['input'] = [re.sub(r"\b\w*escapelong\w*\b", "", x) for x in df['input']]
    df['input'] = [word_tokenize(x) for x in df['input']]
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

    for idx, token in enumerate(df['input']):
        finalized = []
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for word, tag in nltk.pos_tag(token):
            if word not in stopwords.words('english') and word.isalpha():
                word_finalized = lemmatizer.lemmatize(word, get_pos(tag))
                finalized.append(word_finalized)
        df.loc[idx, 'input'] = str(finalized)
    return df

tfidf = jb.load("tfidf.pkl")
model = jb.load("model.pkl")
def checker():
    inp = text_input.get('1.0', tk.END).strip()
    df = {
        "input": [inp]
    }
    df = pd.DataFrame(df)
    df = clean(df)

    prediction = model.predict(tfidf.transform(df['input']))
    output_label.config(text=("True" if prediction[0] == 1 else "False"))


root = tk.Tk()
root.title("Spam Checker")
root.geometry("600x600")
text_input = tk.Text(root, width=50, height=5)
text_input.pack(pady=10)
button = tk.Button(root, text="Check", command=checker)
button.pack(pady=10)
output_label = tk.Label(root, text="")
output_label.pack(pady=10)
root.mainloop()