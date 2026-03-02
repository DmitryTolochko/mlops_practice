import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import joblib


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def prepare_data(train_file, test_file, max_features=5000):
    # Загрузка
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Очистка текста
    for col in ['description_x', 'description_y']:
        train_df[col] = train_df[col].apply(preprocess_text)
        test_df[col] = test_df[col].apply(preprocess_text)

    # Векторизация
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    all_train_texts = pd.concat([train_df['description_x'], train_df['description_y']])
    vectorizer.fit(all_train_texts)

    train_x_vec = vectorizer.transform(train_df['description_x'])
    train_y_vec = vectorizer.transform(train_df['description_y'])
    X_train = np.hstack([train_x_vec.toarray(), train_y_vec.toarray()])
    y_train = train_df['target'].values

    test_x_vec = vectorizer.transform(test_df['description_x'])
    test_y_vec = vectorizer.transform(test_df['description_y'])
    X_test = np.hstack([test_x_vec.toarray(), test_y_vec.toarray()])
    y_test = test_df['target'].values

    # Сохраняем векторизатор
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # Сохраняем признаки и целевую переменную
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)


prepare_data('data/train.csv', 'data/test.csv')