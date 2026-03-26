import pandas as pd
import pathlib

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/diamonds.csv"


csv_name = DATA_URL.split("/")[-1]
data_path = pathlib.Path(__file__).parent.parent / "data" / csv_name
data_path = data_path.resolve() 

try:
    data_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_URL)
    df.to_csv(data_path, index=False)
except Exception as e:
    print(f"An error occurred while getting data: {e}")
