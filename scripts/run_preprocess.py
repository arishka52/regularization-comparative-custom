from pathlib import Path
from src.config import Config

def main():
    cfg = Config()

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    if not cfg.data_raw.exists():
        raise FileNotFoundError(
            "Не найден файл data/raw/train.csv.\n"
            "Скачай train.csv с Kaggle (House Prices) и положи в data/raw/train.csv"
        )

    print("OK: структура папок готова, train.csv найден.")

if __name__ == "__main__":
    main()
