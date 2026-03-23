import pandas as pd
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
df = pd.read_csv(f"{path}/data/diamonds.csv")

# Очищаем от дубликатов и NaN
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Рассчитываем объем камня и очищаем записи, где 0 (невозможно)
df['volume'] = df['x'] * df['y'] * df['z']
df = df[df['volume'] != 0]

# Переводим категориальные данные в числовые (колонки: cut, clarity и color)
codes_for_cut = zip(df['cut'].unique(), range(len(df['cut'].unique())))
codes_for_clarity = zip(df['clarity'].unique(), range(len(df['clarity'].unique())))
codes_for_color = zip(df['color'].unique(), range(len(df['color'].unique())))

df['cut'] = df['cut'].map(dict(codes_for_cut))
df['clarity'] = df['clarity'].map(dict(codes_for_clarity))
df['color'] = df['color'].map(dict(codes_for_color))

# Разбиваем на данные для обучения и валидации
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

train_data.to_csv(f"{path}/data/train.csv", index=False)
test_data.to_csv(f"{path}/data/test.csv", index=False)