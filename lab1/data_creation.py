import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import pandas as pd

os.makedirs("data", exist_ok=True)

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "rishisankineni/text-similarity",
  "train.csv"
)

df.drop(columns=['ticker_x', 'ticker_y', 'Unnamed: 0'], inplace=True)

X = df.drop('same_security', axis=1)
y = df['same_security'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

train_final = pd.DataFrame(X_train, columns=X_train.columns)
train_final['target'] = y_train.values

test_final = pd.DataFrame(X_test, columns=X_test.columns)
test_final['target'] = y_test.values

train_final.to_csv("data/train.csv", index=False)
test_final.to_csv("data/test.csv", index=False)