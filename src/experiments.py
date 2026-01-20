from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.preprocessing import preprocess_house_prices
from src.metrics import rmse, r2
from src.models import OLSRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor
from src.optimizers import compute_lipschitz_lr


def run_experiments():
    cfg = Config()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    data = preprocess_house_prices(cfg)
    Xtr, Xte = data.X_train, data.X_test
    ytr, yte = data.y_train, data.y_test

    safe_lr = compute_lipschitz_lr(Xtr)
    lr = min(cfg.learning_rate, safe_lr)

    rows = []

    ols = OLSRegressor()
    fitinfo = ols.fit(Xtr, ytr, lr=lr, max_iter=cfg.max_iter, tol=cfg.tol)
    pred_tr = ols.predict(Xtr)
    pred_te = ols.predict(Xte)

    rows.append({
        "model": "OLS",
        "lambda": 0.0,
        "alpha": np.nan,
        "rmse_train": rmse(ytr, pred_tr),
        "rmse_test": rmse(yte, pred_te),
        "r2_train": r2(ytr, pred_tr),
        "r2_test": r2(yte, pred_te),
        "w_norm2": float(np.linalg.norm(ols.w)),
        "nnz": int(np.sum(np.abs(ols.w) > 1e-10)),
        "n_iter": fitinfo.n_iter,
        "final_loss": fitinfo.final_loss,
        "lr": lr,
    })

    for lam in cfg.lambdas:
        ridge = RidgeRegressor(lam)
        fitinfo = ridge.fit(Xtr, ytr, lr=lr, max_iter=cfg.max_iter, tol=cfg.tol)
        pred_tr = ridge.predict(Xtr)
        pred_te = ridge.predict(Xte)

        rows.append({
            "model": "Ridge",
            "lambda": float(lam),
            "alpha": np.nan,
            "rmse_train": rmse(ytr, pred_tr),
            "rmse_test": rmse(yte, pred_te),
            "r2_train": r2(ytr, pred_tr),
            "r2_test": r2(yte, pred_te),
            "w_norm2": float(np.linalg.norm(ridge.w)),
            "nnz": int(np.sum(np.abs(ridge.w) > 1e-10)),
            "n_iter": fitinfo.n_iter,
            "final_loss": fitinfo.final_loss,
            "lr": lr,
        })

        lasso = LassoRegressor(lam)
        fitinfo = lasso.fit(Xtr, ytr, lr=lr, max_iter=cfg.max_iter, tol=cfg.tol)
        pred_tr = lasso.predict(Xtr)
        pred_te = lasso.predict(Xte)

        rows.append({
            "model": "Lasso",
            "lambda": float(lam),
            "alpha": np.nan,
            "rmse_train": rmse(ytr, pred_tr),
            "rmse_test": rmse(yte, pred_te),
            "r2_train": r2(ytr, pred_tr),
            "r2_test": r2(yte, pred_te),
            "w_norm2": float(np.linalg.norm(lasso.w)),
            "nnz": int(np.sum(np.abs(lasso.w) > 1e-10)),
            "n_iter": fitinfo.n_iter,
            "final_loss": fitinfo.final_loss,
            "lr": lr,
        })

        for a in cfg.alphas:
            en = ElasticNetRegressor(lam, a)
            fitinfo = en.fit(Xtr, ytr, lr=lr, max_iter=cfg.max_iter, tol=cfg.tol)
            pred_tr = en.predict(Xtr)
            pred_te = en.predict(Xte)

            rows.append({
                "model": "ElasticNet",
                "lambda": float(lam),
                "alpha": float(a),
                "rmse_train": rmse(ytr, pred_tr),
                "rmse_test": rmse(yte, pred_te),
                "r2_train": r2(ytr, pred_tr),
                "r2_test": r2(yte, pred_te),
                "w_norm2": float(np.linalg.norm(en.w)),
                "nnz": int(np.sum(np.abs(en.w) > 1e-10)),
                "n_iter": fitinfo.n_iter,
                "final_loss": fitinfo.final_loss,
                "lr": lr,
            })

    res = pd.DataFrame(rows)
    out_path = cfg.tables_dir / "results.csv"
    res.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def make_figures():
    cfg = Config()
    in_path = cfg.tables_dir / "results.csv"
    if not in_path.exists():
        raise FileNotFoundError("Нет reports/tables/results.csv. Сначала запусти python -m scripts.run_experiments")

    df = pd.read_csv(in_path)

    for model_name in ["Ridge", "Lasso"]:
        sub = df[df["model"] == model_name].copy().sort_values("lambda")

        plt.figure()
        plt.xscale("log")
        plt.plot(sub["lambda"], sub["rmse_train"], label="train")
        plt.plot(sub["lambda"], sub["rmse_test"], label="test")
        plt.xlabel("lambda (log scale)")
        plt.ylabel("RMSE")
        plt.title(f"{model_name}: RMSE vs lambda")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.figures_dir / f"rmse_{model_name.lower()}.png")
        plt.close()

    en = df[df["model"] == "ElasticNet"].copy()
    for a in sorted(en["alpha"].dropna().unique()):
        sub = en[en["alpha"] == a].sort_values("lambda")
        plt.figure()
        plt.xscale("log")
        plt.plot(sub["lambda"], sub["rmse_test"], label=f"alpha={a}")
        plt.xlabel("lambda (log scale)")
        plt.ylabel("RMSE test")
        plt.title("ElasticNet: RMSE test vs lambda")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.figures_dir / f"rmse_elasticnet_alpha_{a}.png")
        plt.close()

    lasso = df[df["model"] == "Lasso"].sort_values("lambda")
    plt.figure()
    plt.xscale("log")
    plt.plot(lasso["lambda"], lasso["nnz"], label="Lasso")
    plt.xlabel("lambda (log scale)")
    plt.ylabel("Non-zero coefficients (nnz)")
    plt.title("Sparsity: nnz vs lambda (Lasso)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "nnz_lasso.png")
    plt.close()

    print(f"Saved figures to: {cfg.figures_dir}")
