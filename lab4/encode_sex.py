import pandas as pd

df = pd.read_csv("data/titanic.csv")
df = pd.get_dummies(df, columns=["Sex"])
df.to_csv("data/titanic.csv", index=False)
print(df.head())