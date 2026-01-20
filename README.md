# regularization-comparative-custom

Запуск:

Скачайте train.csv Kaggle House Prices и положите в data/raw/train.csv

Создайте окружение и установите зависимости:

python -m venv .venv

.\.venv\Scripts\activate

pip install -r requirements.txt

Выполните:

python -m scripts.run_preprocess

python -m scripts.run_experiments

python -m scripts.make_figures

Результаты:

таблица: reports/tables/results.csv

графики: reports/figures/*.png
