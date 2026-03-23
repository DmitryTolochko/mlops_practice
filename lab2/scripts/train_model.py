from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import pathlib

# Split
path = pathlib.Path(__file__).parent.parent.resolve()
train_df = pd.read_csv(f"{path}/data/train.csv")
X, y = train_df.drop('price', axis=1), train_df['price']


# Train
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# Save
with open(f"{path}/model/model.pkl",'wb') as f:
    pickle.dump(model, f)

