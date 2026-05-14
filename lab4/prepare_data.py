from catboost.datasets import titanic
import pandas as pd
import os

train_df, test_df = titanic()
os.makedirs("data", exist_ok=True)
df = train_df[["Pclass", "Sex", "Age"]]
df.to_csv("data/titanic.csv", index=False)
print(df.head())