import numpy as np
import joblib
import pickle
from sklearn.metrics import accuracy_score

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model test accuracy is: {accuracy:.3f}")