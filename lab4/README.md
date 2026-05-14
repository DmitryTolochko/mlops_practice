# Лабораторная работа №4  
# Версионирование данных с использованием DVC

## Цель работы

Изучить возможности системы контроля версий данных DVC и научиться:

- подключать DVC к Git-репозиторию;
- хранить большие файлы вне Git;
- создавать версии датасетов;
- переключаться между версиями данных;
- использовать хранилище данных.

---

# Используемые технологии

- PyCharm
- Git
- DVC
- Python 3
- Pandas
- CatBoost Titanic Dataset

---

# Структура проекта

```text
lab4/
│
├── data/
│   ├── .gitignore
│   ├── titanic.csv.dvc
│   └── titanic.csv
│
├── dvc_storage/
│
├── .gitignore
├── encode_sex.py
├── fill_age.py
├── prepare_data.py
│
├── README.md
└── requirements.txt
```

---

# 1. Создание проекта

Проект был создан в среде разработки PyCharm.

Для изоляции зависимостей использовалось виртуальное окружение `venv`.

---

# 2. Инициализация Git и DVC

В терминале PyCharm были выполнены команды:

```bash
git init
dvc init
```

После инициализации выполнен первый коммит:

```bash
git add .
git commit -m "Initialize Git and DVC"
```

---

# 3. Настройка удалённого хранилища

В качестве удалённого хранилища использовался Google Drive.

Подключение remote-хранилища:

```bash
dvc remote add -d storage gdrive://<FOLDER_ID>
```

На данном этапе возникла проблемма с доступом dvc к Google Drive. 
В связи с этим было приняло решение о создании локально хранилища в самом проекте dvc_storage.

```bash
dvc remote add -d my_local_storage lab4/dvc_storage
```
---


---

# 4. Создание исходного датасета

Для работы использовался датасет Titanic из библиотеки CatBoost.

Файл `prepare_data.py`:

```python
from catboost.datasets import titanic
import pandas as pd
import os

train_df, test_df = titanic()
os.makedirs("data", exist_ok=True)
df = train_df[["Pclass", "Sex", "Age"]]
df.to_csv("data/titanic.csv", index=False)
print(df.head())
```

Запуск:

```bash
python prepare_data.py
```

---

# 5. Добавление датасета в DVC

Добавление файла в DVC:

```bash
dvc add data/titanic.csv
```

После выполнения команды были созданы:

```text
data/titanic.csv.dvc
.gitignore
```

Коммит изменений:

```bash
git add data/titanic.csv.dvc .gitignore
git commit -m "Add initial Titanic dataset"
```

Загрузка данных в remote:

```bash
dvc push
```

---

# 6. Создание второй версии датасета

На данном этапе пропущенные значения в столбце `Age` были заполнены средним значением.

Файл `fill_age.py`:

```python
import pandas as pd

df = pd.read_csv("data/titanic.csv")
mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)
df.to_csv("data/titanic.csv", index=False)
print(df.head())
```

Запуск:

```bash
python fill_age.py
```

Обновление DVC:

```bash
dvc add data/titanic.csv
```

Создание коммита:

```bash
git add data/titanic.csv.dvc
git commit -m "Fill missing Age values"
```

Загрузка новой версии:

```bash
dvc push
```
---

# 7. Создание третьей версии датасета

На данном этапе был создан новый признак с использованием One-Hot-Encoding для признака `Sex`.

Файл `encode_sex.py`:

```python
import pandas as pd

df = pd.read_csv("data/titanic.csv")
df = pd.get_dummies(df, columns=["Sex"])
df.to_csv("data/titanic.csv", index=False)
print(df.head())
```

Запуск:

```bash
python encode_sex.py
```

Обновление DVC:

```bash
dvc add data/titanic.csv
```

Создание коммита:

```bash
git add data/titanic.csv.dvc
git commit -m "Add one-hot encoding for Sex"
```

Загрузка новой версии:

```bash
dvc push
```
---

# 8. Просмотр истории версий

Для просмотра истории коммитов использовалась команда:

```bash
git log --oneline
```

Пример:

```text
f614184 Add one-hot encoding for Sex
547c36d Fill missing Age values
ad4fd46 Add initial Titanic dataset
```

---

# 9. Переключение между версиями датасета

## Переключение на первую версию

```bash
git checkout ad4fd46
dvc checkout
```

## Переключение на вторую версию

```bash
git checkout 547c36d
dvc checkout
```

## Возврат к последней версии

```bash
git checkout f614184
dvc checkout
```

После выполнения команд содержимое файла `titanic.csv` изменялось в соответствии с выбранной версией.

---


# 11. Хранение данных в удалённом репозитории

Файлы датасета не хранятся напрямую в Git-репозитории.

Git хранит только:

- `.dvc` файлы;
- метаинформацию;
- хэши версий данных.

Сами данные находятся в dvc_storage.

---

# Вывод

В ходе лабораторной работы были изучены основные возможности системы DVC.

В результате выполнения работы удалось:

- настроить совместную работу Git и DVC;
- подключить хранилище данных;
- создать несколько версий датасета;
- хранить большие файлы вне Git;
- переключаться между версиями данных;
- использовать DVC для отслеживания изменений датасета.

---

# Дополнительные screenshots

Продемонстрировал выводы Terminal при прохождении каждого этапа.
Screenshots находятся в папке lab4/dvc_screenshots.