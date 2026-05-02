from __future__ import annotations

import pickle
import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/diamonds.csv"
)

'''
Обрабатывает данные:
:param df: pd.DataFrame - исходный датафрейм
:return: pd.DataFrame - обработанный датафрейм
:return: dict - карты категорий
'''
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["volume"] = df["x"] * df["y"] * df["z"]
    df = df[df["volume"] != 0]

    codes_cut = dict(zip(df["cut"].unique(), range(len(df["cut"].unique()))))
    codes_clarity = dict(zip(df["clarity"].unique(), range(len(df["clarity"].unique()))))
    codes_color = dict(zip(df["color"].unique(), range(len(df["color"].unique()))))

    df["cut"] = df["cut"].map(codes_cut)
    df["clarity"] = df["clarity"].map(codes_clarity)
    df["color"] = df["color"].map(codes_color)

    maps = {"cut": codes_cut, "clarity": codes_clarity, "color": codes_color}
    return df, maps


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_URL)
    df, maps = preprocess(df)

    train_df = df.sample(frac=0.8, random_state=42)
    X = train_df.drop("price", axis=1)
    y = train_df["price"]
    feature_columns = list(X.columns)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    artifact = {
        "model": model,
        "maps": maps,
        "feature_columns": feature_columns,
    }
    out_path = model_dir / "artifact.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
