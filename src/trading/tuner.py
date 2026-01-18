from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from trading.ma_crossover import (
    MACrossoverBacktestResult,
    MACrossoverMetrics,
    backtest_ma_crossover,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    short_window: int
    long_window: int
    score: float
    metrics: MACrossoverMetrics


@dataclass(frozen=True)
class MATuningResult:
    best_candidate: CandidateEvaluation
    best_result: MACrossoverBacktestResult
    evaluations: list[CandidateEvaluation]


def _metric_selector(metric_name: str) -> Callable[[MACrossoverMetrics], float]:
    normalized = metric_name.strip().lower()
    if normalized == "sharpe":
        return lambda metrics: (
            float(metrics.sharpe) if metrics.sharpe is not None else float("-inf")
        )
    if normalized == "annualized_return":
        return lambda metrics: float(metrics.annualized_return)
    return lambda metrics: float(metrics.total_return)


def _weighted_choice(values: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum())
    if total <= 0:
        probabilities = np.full_like(weights, 1.0 / len(weights), dtype=float)
    else:
        probabilities = weights / total

    idx = int(rng.choice(len(values), p=probabilities))
    return idx


def tune_ma_crossover(
    df: pd.DataFrame,
    *,
    short_window_range: tuple[int, int] = (5, 30),
    long_window_range: tuple[int, int] = (20, 200),
    population_size: int = 16,
    iterations: int = 20,
    evaporation: float = 0.35,
    pheromone_deposit: float = 1.0,
    metric: str = "total_return",
    long_only: bool = True,
    fee_bps: float = 10.0,
    initial_cash: float = 10_000.0,
    random_seed: int | None = None,
) -> MATuningResult:
    """
    Hybrid genetic / ant-colony tuner for MA crossover parameters.
    """
    if not 0 < evaporation < 1:
        raise ValueError("evaporation must be between 0 and 1.")
    if population_size <= 0 or iterations <= 0:
        raise ValueError("population_size and iterations must be positive.")
    if short_window_range[0] >= short_window_range[1]:
        raise ValueError("Invalid short_window_range.")
    if long_window_range[0] >= long_window_range[1]:
        raise ValueError("Invalid long_window_range.")
    if short_window_range[1] >= long_window_range[1]:
        raise ValueError("The long window max must be greater than the short window max.")

    rng = np.random.default_rng(random_seed)
    selector = _metric_selector(metric)

    short_values = np.arange(short_window_range[0], short_window_range[1] + 1, dtype=int)
    long_values = np.arange(long_window_range[0], long_window_range[1] + 1, dtype=int)

    short_index = {value: idx for idx, value in enumerate(short_values)}
    long_index = {value: idx for idx, value in enumerate(long_values)}

    pher_short = np.ones_like(short_values, dtype=float)
    pher_long = np.ones_like(long_values, dtype=float)

    evaluations: list[CandidateEvaluation] = []
    best_candidate: CandidateEvaluation | None = None
    best_result: MACrossoverBacktestResult | None = None

    for _ in range(iterations):
        iteration_candidates: list[tuple[CandidateEvaluation, MACrossoverBacktestResult]] = []

        for _ in range(population_size):
            short_idx = _weighted_choice(short_values, pher_short, rng)
            short_window = int(short_values[short_idx])

            eligible_mask = long_values > short_window
            if not np.any(eligible_mask):
                continue

            eligible_indices = np.where(eligible_mask)[0]
            eligible_weights = pher_long[eligible_indices]
            long_choice_idx_rel = _weighted_choice(
                long_values[eligible_indices], eligible_weights, rng
            )
            long_idx = int(eligible_indices[long_choice_idx_rel])
            long_window = int(long_values[long_idx])

            try:
                result = backtest_ma_crossover(
                    df=df,
                    short_window=short_window,
                    long_window=long_window,
                    long_only=long_only,
                    fee_bps=fee_bps,
                    initial_cash=initial_cash,
                )
            except ValueError:
                continue

            score = selector(result.metrics)
            candidate = CandidateEvaluation(
                short_window=short_window,
                long_window=long_window,
                score=score,
                metrics=result.metrics,
            )
            evaluations.append(candidate)
            iteration_candidates.append((candidate, result))

            if best_candidate is None or score > best_candidate.score:
                best_candidate = candidate
                best_result = result

        if not iteration_candidates:
            continue

        scores = np.array([max(c.score, 0.0) for c, _ in iteration_candidates], dtype=float)
        max_score = float(scores.max()) if len(scores) else 0.0
        min_score = float(scores.min()) if len(scores) else 0.0

        normalized_scores: Iterable[float]
        if max_score - min_score < 1e-9:
            normalized_scores = (1.0 for _ in iteration_candidates)
        else:
            normalized_scores = (
                (max(c.score, 0.0) - min_score) / (max_score - min_score + 1e-9)
                for c, _ in iteration_candidates
            )

        pher_short *= (1.0 - evaporation)
        pher_long *= (1.0 - evaporation)

        for (candidate, _), norm_score in zip(iteration_candidates, normalized_scores):
            deposit = pheromone_deposit * (1.0 + norm_score)
            pher_short[short_index[candidate.short_window]] += deposit
            pher_long[long_index[candidate.long_window]] += deposit

    if not evaluations or best_candidate is None or best_result is None:
        raise ValueError("Unable to evaluate candidates with the provided settings.")

    return MATuningResult(
        best_candidate=best_candidate,
        best_result=best_result,
        evaluations=evaluations,
    )
