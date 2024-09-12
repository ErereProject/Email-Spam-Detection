from sklearn.svm import SVC
import joblib

train_x_tfidf = joblib.load("train_x.pkl")
test_x_tfidf = joblib.load("test_x.pkl")
train_y = joblib.load("train_y.pkl")
test_y = joblib.load("test_y.pkl")
tfidf = joblib.load("tfidf.pkl")


model = SVC(kernel="poly", C=1000, coef0=2, degree=3, gamma=0.0001)
model.fit(train_x_tfidf, train_y)
joblib.dump(model, "model.pkl", compress=True)