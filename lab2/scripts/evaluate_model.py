import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()

model = joblib.load(f"{path}/model/model.pkl")
X_test = pd.read_csv(f"{path}/data/test.csv").drop('price', axis=1)
y_test = pd.read_csv(f"{path}/data/test.csv")['price'].values
pred = model.predict(X_test)

# Метрики
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"MAE ($): {mae:.2f}")
print(f"MSE ($): {mse:.2f}")
print(f"R²: {r2:.3f}")