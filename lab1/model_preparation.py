import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)