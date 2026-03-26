# Конвеер автоматизации Jenkins

Для выполнения лабораторной работы были созданы скрипты для загрузки данных, их обработки, обучения модели, её сохранения и оценки метрик. Также создан `Jenkinsfile`, в котором был описан пайплайн для автоматизации всех скриптов.

## Описание скриптов:

- `load_data.py` - загружает данные о стоимости алмазов ([ссылка на датасет](https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/diamonds.csv)) и сохраняет их в папке `data`;
- `preprocess_data.py` - производит препроцессинг данных: очищает дубликаты и пустые значения, переводит категориальные данные в числовые, разделяет на обучающий и тестовый датасеты и сохраняет их в папке `data`;
- `train_model.py` - производит обучение модели и сохраняет её в папку `model`;
- `evaluate_model.py` - испольует сохраненную модель и рассчитывает необходимые метрики.

## Конфигурация VM с Jenkins

- OC: Linux, Ubuntu 22.04
- Установленные пакеты: `python-3.13.7`, `pip`, `setuptools`, `wheels`, `java-21`, `jenkins`

## Конфигурация job-ы Jenkins

**Definition** - Pipline script from SCM

**SCM** - Git

**Repository URL** - https://github.com/DmitryTolochko/mlops_practice

**Branch Specifier** - \*/master

**Script Path** - lab2/Jenkinsfile
