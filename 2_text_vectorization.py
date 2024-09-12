from pandas import read_csv
import sklearn as sk
import joblib

df = read_csv("final.csv", encoding ="latin-1")

train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(df['text_final'], df['label'], test_size=0.3)
tfidf = sk.feature_extraction.text.TfidfVectorizer(max_features=4000)
tfidf.fit(df['text_final'])
train_x_tfidf = tfidf.transform(train_x)
test_x_tfidf = tfidf.transform(test_x)

joblib.dump(tfidf, "tfidf.pkl")