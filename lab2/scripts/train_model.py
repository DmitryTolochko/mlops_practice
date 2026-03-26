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
try:
    model_path = pathlib.Path(__file__).parent.parent / "model" / "model.pkl"
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
except Exception as e:
    print(f"An error occurred while saving model: {e}")
