#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian VAR workflow for goods and services datasets.

Key features:
    • Modular Minnesota-prior BVAR estimation
    • Rolling-origin, multi-horizon forecasts (1–3 months ahead by default)
    • Cross-validation across lag lengths and Minnesota tightness hyperparameters
    • Out-of-sample MAE, RMSE, and R² per variable and horizon
    • Persistence of forecast-vs-actual series and summary metrics to disk

Ensure the updated CSV files are placed in the repository root before running:
    - updated_dataset_goods.csv
    - updated_dataset_services.csv
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import invwishart
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("bvar_pipeline")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    date_column: Optional[str] = "date"


@dataclass(frozen=True)
class ModelConfig:
    lags: int
    lambda1: float = 0.3
    lambda3: float = 1.0
    include_const: bool = True
    draws: int = 300
    horizons: Tuple[int, ...] = (1, 2, 3)
    train_fraction: float = 0.7
    random_seed: Optional[int] = 2025


# ---------------------------------------------------------------------------
# Minnesota prior helpers
# ---------------------------------------------------------------------------
def estimate_residual_variances(series: np.ndarray) -> np.ndarray:
    """
    Estimate residual variances via AR(1) regressions for each variable.
    Falls back to the variable variance when data are insufficient.
    """
    T, K = series.shape
    variances = np.zeros(K)
    eps = 1e-6

    if T < 3:
        variances.fill(1.0)
        return variances

    for idx in range(K):
        y = series[1:, idx]
        x = series[:-1, idx]
        if len(x) < 2:
            variances[idx] = max(np.var(y, ddof=1), eps)
            continue

        beta = np.linalg.lstsq(x[:, None], y, rcond=None)[0]
        resid = y - x * beta
        var = np.var(resid, ddof=1)
        if not np.isfinite(var) or var <= 0:
            var = np.var(series[:, idx], ddof=1)
        variances[idx] = max(var, eps)

    return variances


def minnesota_prior(
    history: np.ndarray,
    p: int,
    lambda1: float = 0.3,
    lambda3: float = 1.0,
    include_const: bool = True,
    random_walk_prior: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Minnesota prior mean (B0) and covariance (V0) for the BVAR.
    """
    _, K = history.shape
    offset = 1 if include_const else 0
    M = K * p + offset

    B0 = np.zeros((M, K))
    if random_walk_prior and p >= 1:
        for k in range(K):
            B0[offset + k, k] = 1.0  # own first lag prior mean

    sigma_sq = estimate_residual_variances(history)
    V0 = np.zeros((M, M))
    if include_const:
        V0[0, 0] = 10.0  # loose prior on intercept

    eps = 1e-6
    for lag in range(1, p + 1):
        for j in range(K):  # regressor variable
            row_idx = offset + (lag - 1) * K + j
            for k in range(K):  # equation index
                if j == k:
                    variance = (lambda1 ** 2) / (lag ** (2 * lambda3))
                else:
                    variance = (
                        (lambda1 ** 2) * (sigma_sq[k] / max(sigma_sq[j], eps))
                    ) / (lag ** (2 * lambda3))
                V0[row_idx, row_idx] = max(variance, eps)

    return B0, V0


# ---------------------------------------------------------------------------
# Core BVAR routines
# ---------------------------------------------------------------------------
def build_var_matrices(
    series: np.ndarray, p: int, include_const: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct (Y, X) matrices for VAR(p) regression.
    """
    T, K = series.shape
    if T <= p:
        raise ValueError("Time series length must exceed the chosen lag order.")

    Y_rows = []
    X_rows = []
    for t in range(p, T):
        lags = [series[t - lag] for lag in range(1, p + 1)]
        reg = np.hstack(lags)
        if include_const:
            reg = np.hstack([1.0, reg])
        X_rows.append(reg)
        Y_rows.append(series[t])

    return np.vstack(Y_rows), np.vstack(X_rows)


def bvar_posterior(
    Y: np.ndarray,
    X: np.ndarray,
    B0: np.ndarray,
    V0: np.ndarray,
    S0: Optional[np.ndarray] = None,
    nu0: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Conjugate posterior update for Matrix-Normal / Inverse-Wishart prior.
    """
    N, K = Y.shape
    M = X.shape[1]

    if S0 is None:
        S0 = np.eye(K)
    if nu0 is None:
        nu0 = K + 2

    V0_inv = inv(V0)
    XtX = X.T @ X
    XtY = X.T @ Y

    Vn = inv(V0_inv + XtX)
    Bn = Vn @ (V0_inv @ B0 + XtY)

    resid = Y - X @ Bn
    term1 = resid.T @ resid
    diff = Bn - B0
    term2 = diff.T @ V0_inv @ diff

    Sn = S0 + term1 + term2
    nun = nu0 + N

    return Bn, Vn, Sn, nun


def sample_posterior(
    Bn: np.ndarray,
    Vn: np.ndarray,
    Sn: np.ndarray,
    nun: int,
    draws: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from the posterior of (B, Sigma).
    """
    rng = np.random.default_rng(random_state)
    M, K = Bn.shape
    B_samples = np.zeros((draws, M, K))
    Sigma_samples = np.zeros((draws, K, K))

    for i in range(draws):
        Sigma_i = invwishart.rvs(df=nun, scale=Sn, random_state=rng)
        Sigma_samples[i] = Sigma_i

        cov = np.kron(Sigma_i, Vn)
        mean = Bn.flatten(order="F")
        vec_sample = rng.multivariate_normal(mean=mean, cov=cov)
        B_samples[i] = vec_sample.reshape(M, K, order="F")

    return B_samples, Sigma_samples


def _stack_lags(history: np.ndarray, p: int) -> np.ndarray:
    if history.shape[0] < p:
        raise ValueError("Insufficient history to stack requested lags.")
    lags = [history[-lag] for lag in range(1, p + 1)]
    return np.hstack(lags)


def forecast_paths(
    B_samples: np.ndarray,
    Sigma_samples: np.ndarray,
    history: np.ndarray,
    p: int,
    horizon: int,
    include_const: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate predictive paths of length `horizon` for each posterior draw.
    """
    rng = np.random.default_rng(random_state)
    draws, _, K = B_samples.shape
    paths = np.zeros((draws, horizon, K))

    for draw_idx in range(draws):
        B = B_samples[draw_idx]
        Sigma = Sigma_samples[draw_idx]
        simulated = history.copy()

        for step in range(horizon):
            reg = _stack_lags(simulated, p)
            if include_const:
                reg = np.hstack([1.0, reg])
            mean = reg @ B
            shock = rng.multivariate_normal(np.zeros(K), Sigma)
            y_next = mean + shock

            paths[draw_idx, step] = y_next
            simulated = np.vstack([simulated, y_next])

    return paths


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------
def rolling_forecast(
    series: np.ndarray,
    var_names: Sequence[str],
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Perform rolling-origin forecasts and collect predictions vs actuals.
    """
    T, K = series.shape
    max_h = max(config.horizons)
    min_train = max(int(T * config.train_fraction), config.lags + 5)

    if T <= min_train + max_h:
        LOGGER.warning(
            "Not enough observations to perform rolling forecast for lags=%s.",
            config.lags,
        )
        return pd.DataFrame()

    records = []
    base_rng = np.random.default_rng(config.random_seed)

    for origin in range(min_train, T - max_h):
        train = series[:origin]
        try:
            Y, X = build_var_matrices(train, config.lags, config.include_const)
        except ValueError:
            continue

        B0, V0 = minnesota_prior(
            train,
            p=config.lags,
            lambda1=config.lambda1,
            lambda3=config.lambda3,
            include_const=config.include_const,
        )

        Bn, Vn, Sn, nun = bvar_posterior(Y, X, B0, V0)

        B_samples, Sigma_samples = sample_posterior(
            Bn,
            Vn,
            Sn,
            nun,
            draws=config.draws,
            random_state=base_rng.integers(0, 2**32 - 1),
        )

        paths = forecast_paths(
            B_samples,
            Sigma_samples,
            history=train,
            p=config.lags,
            horizon=max_h,
            include_const=config.include_const,
            random_state=base_rng.integers(0, 2**32 - 1),
        )

        mean_path = paths.mean(axis=0)

        for horizon in config.horizons:
            target_idx = origin + horizon
            if target_idx >= T:
                continue
            prediction = mean_path[horizon - 1]
            actual = series[target_idx]

            for var_idx in range(K):
                records.append(
                    {
                        "origin_index": origin,
                        "horizon": horizon,
                        "variable_index": var_idx,
                        "variable_name": var_names[var_idx],
                        "prediction": float(prediction[var_idx]),
                        "actual": float(actual[var_idx]),
                    }
                )

    return pd.DataFrame(records)


def compute_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAE, RMSE, and R² grouped by horizon and variable.
    """
    if forecasts.empty:
        return pd.DataFrame()

    metrics = []
    grouped = forecasts.groupby(["horizon", "variable_index", "variable_name"])
    for (horizon, var_idx, var_name), group in grouped:
        y_true = group["actual"].values
        y_pred = group["prediction"].values
        if len(y_true) < 2:
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        metrics.append(
            {
                "horizon": horizon,
                "variable_index": var_idx,
                "variable_name": var_name,
                "n_obs": len(group),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(metrics)


def evaluate_configs(
    series: np.ndarray,
    var_names: Sequence[str],
    dataset_name: str,
    lag_grid: Iterable[int],
    lambda_grid: Iterable[float],
    base_config: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling forecasts for each configuration and collect metrics.

    Returns:
        metrics_df: concatenated metrics for all configurations
        best_forecasts_df: forecast records for the best configuration
    """
    all_metrics = []
    best_score = np.inf
    best_forecasts = pd.DataFrame()

    for lags, lambda1 in itertools.product(lag_grid, lambda_grid):
        config = ModelConfig(
            lags=lags,
            lambda1=lambda1,
            lambda3=base_config.lambda3,
            include_const=base_config.include_const,
            draws=base_config.draws,
            horizons=base_config.horizons,
            train_fraction=base_config.train_fraction,
            random_seed=base_config.random_seed,
        )

        LOGGER.info(
            "Evaluating dataset=%s with lags=%d, lambda1=%.3f",
            dataset_name,
            lags,
            lambda1,
        )

        forecasts = rolling_forecast(series, var_names, config)
        if forecasts.empty:
            LOGGER.warning(
                "Skipping configuration (lags=%d, lambda1=%.3f) due to insufficient forecasts.",
                lags,
                lambda1,
            )
            continue

        metrics = compute_metrics(forecasts)
        if metrics.empty:
            LOGGER.warning(
                "No metrics computed for configuration (lags=%d, lambda1=%.3f).",
                lags,
                lambda1,
            )
            continue

        metrics = metrics.assign(
            dataset=dataset_name,
            lags=lags,
            lambda1=lambda1,
        )
        all_metrics.append(metrics)

        mean_rmse = metrics["RMSE"].mean()
        LOGGER.info(
            "Configuration (lags=%d, lambda1=%.3f) average RMSE=%.4f",
            lags,
            lambda1,
            mean_rmse,
        )

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_forecasts = forecasts.assign(
                dataset=dataset_name,
                lags=lags,
                lambda1=lambda1,
            )

    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = pd.DataFrame()

    return metrics_df, best_forecasts


# ---------------------------------------------------------------------------
# Data loading and persistence
# ---------------------------------------------------------------------------
def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """
    Load dataset from CSV and return numeric columns sorted by date if present.
    """
    if not config.path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.path}")

    df = pd.read_csv(config.path)
    if config.date_column and config.date_column in df.columns:
        df[config.date_column] = pd.to_datetime(df[config.date_column])
        df = df.sort_values(config.date_column)
        df = df.set_index(config.date_column)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in dataset {config.path}")

    return numeric_df


def save_results(
    dataset_name: str,
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Persist metrics and forecast-vs-actual comparisons to CSV files.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{dataset_name}_metrics.csv"
    forecasts_path = results_dir / f"{dataset_name}_forecasts.csv"

    if not metrics.empty:
        metrics.to_csv(metrics_path, index=False)
        LOGGER.info("Saved metrics to %s", metrics_path)
    else:
        LOGGER.info("No metrics to save for dataset=%s", dataset_name)

    if not forecasts.empty:
        forecasts.to_csv(forecasts_path, index=False)
        LOGGER.info("Saved forecasts to %s", forecasts_path)
    else:
        LOGGER.info("No forecasts to save for dataset=%s", dataset_name)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    datasets = [
        DatasetConfig(name="goods", path=Path("updated_dataset_goods.csv")),
        DatasetConfig(name="services", path=Path("updated_dataset_services.csv")),
    ]

    base_model = ModelConfig(
        lags=2,
        lambda1=0.3,
        lambda3=1.0,
        include_const=True,
        draws=250,
        horizons=(1, 2, 3),
        train_fraction=0.7,
        random_seed=2025,
    )

    lag_grid = (2, 3, 4)
    lambda_grid = (0.2, 0.3, 0.4)
    results_dir = Path("results")

    for dataset in datasets:
        LOGGER.info("Starting pipeline for dataset=%s", dataset.name)
        try:
            df = load_dataset(dataset)
        except FileNotFoundError:
            LOGGER.warning(
                "Missing dataset for %s at %s. Please add the file and rerun when available.",
                dataset.name,
                dataset.path,
            )
            continue
        except ValueError as exc:
            LOGGER.error("Dataset %s invalid: %s", dataset.name, exc)
            continue

        series = df.to_numpy()
        var_names = list(df.columns)

        metrics_df, best_forecasts_df = evaluate_configs(
            series=series,
            var_names=var_names,
            dataset_name=dataset.name,
            lag_grid=lag_grid,
            lambda_grid=lambda_grid,
            base_config=base_model,
        )

        save_results(dataset.name, metrics_df, best_forecasts_df, results_dir)

    LOGGER.info("Pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
