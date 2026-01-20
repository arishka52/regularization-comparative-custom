from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2

    data_raw: Path = Path("data/raw/train.csv")
    tables_dir: Path = Path("reports/tables")
    figures_dir: Path = Path("reports/figures")

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    alphas = [0.2, 0.5, 0.8]

    learning_rate: float = 0.01
    max_iter: int = 5000
    tol: float = 1e-7

    log_target: bool = True
