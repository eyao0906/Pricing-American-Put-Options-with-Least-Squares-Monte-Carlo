from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
PayoffFn = Callable[[dict[str, Array], int], Array]
BasisFn = Callable[[dict[str, Array], int], Array]
ExerciseMaskFn = Callable[[dict[str, Array], int], Array]


@dataclass
class LSMRegressionStep:
    time_index: int
    coefficients: Array
    n_regression_paths: int
    condition_number: float
    continuation_mean: float
    intrinsic_mean: float


@dataclass
class LSMResult:
    price: float
    stderr: float
    discounted_cashflows: Array
    stopping_times: Array
    stopping_payoffs: Array
    exercise_summary: list[dict[str, float]]
    regression_steps: list[LSMRegressionStep]


class FittedLSMPolicy:
    def __init__(
        self,
        dt: float,
        payoff_fn: PayoffFn,
        basis_fn: BasisFn,
        exercise_mask_fn: ExerciseMaskFn,
        regression_map: dict[int, Array],
    ) -> None:
        self.dt = dt
        self.payoff_fn = payoff_fn
        self.basis_fn = basis_fn
        self.exercise_mask_fn = exercise_mask_fn
        self.regression_map = regression_map

    def evaluate(self, states: dict[str, Array], r: float) -> tuple[float, float, Array, Array]:
        n_paths, n_steps_plus_one = next(iter(states.values())).shape
        steps = n_steps_plus_one - 1
        exercised = np.zeros(n_paths, dtype=bool)
        stopping_times = np.full(n_paths, steps, dtype=int)
        stopping_payoffs = self.payoff_fn(states, steps).copy()

        for t in range(steps - 1, 0, -1):
            eligible = self.exercise_mask_fn(states, t) & (~exercised)
            if not np.any(eligible):
                continue
            intrinsic = self.payoff_fn(states, t)
            itm = eligible & (intrinsic > 0.0)
            if not np.any(itm):
                continue
            beta = self.regression_map.get(t)
            if beta is None:
                continue
            basis = self.basis_fn(states, t)
            continuation = basis[itm] @ beta
            exercise_now = intrinsic[itm] >= continuation
            idx = np.where(itm)[0][exercise_now]
            exercised[idx] = True
            stopping_times[idx] = t
            stopping_payoffs[idx] = intrinsic[idx]

        discounted = np.exp(-r * self.dt * stopping_times) * stopping_payoffs
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
        return price, stderr, stopping_times, discounted


def fit_lsm(
    states: dict[str, Array],
    payoff_fn: PayoffFn,
    basis_fn: BasisFn,
    exercise_mask_fn: ExerciseMaskFn,
    r: float,
    maturity: float,
    regression_path_mask: Array | None = None,
    pricing_path_mask: Array | None = None,
    min_regression_paths: int = 200,
) -> tuple[LSMResult, FittedLSMPolicy]:
    n_paths, n_steps_plus_one = next(iter(states.values())).shape
    steps = n_steps_plus_one - 1
    dt = maturity / steps
    discount = np.exp(-r * dt)

    if regression_path_mask is None:
        regression_path_mask = np.ones(n_paths, dtype=bool)
    if pricing_path_mask is None:
        pricing_path_mask = np.ones(n_paths, dtype=bool)

    regression_path_mask = np.asarray(regression_path_mask, dtype=bool)
    pricing_path_mask = np.asarray(pricing_path_mask, dtype=bool)
    if regression_path_mask.shape != (n_paths,) or pricing_path_mask.shape != (n_paths,):
        raise ValueError("path masks must have shape (n_paths,)")

    payoffs = np.column_stack([payoff_fn(states, t) for t in range(n_steps_plus_one)])
    continuation_cf = payoffs[:, -1].copy()
    alive = np.ones(n_paths, dtype=bool)
    stopping_times = np.full(n_paths, steps, dtype=int)
    stopping_payoffs = payoffs[:, -1].copy()
    regression_steps: list[LSMRegressionStep] = []
    exercise_summary: list[dict[str, float]] = []
    regression_map: dict[int, Array] = {}

    for t in range(steps - 1, 0, -1):
        continuation_cf[alive] *= discount
        eligible = exercise_mask_fn(states, t) & alive
        intrinsic = payoffs[:, t]
        regression_itm = eligible & regression_path_mask & (intrinsic > 0.0)

        beta: Array | None = None
        continuation_all: Array | None = None
        if int(regression_itm.sum()) >= min_regression_paths:
            basis_all = basis_fn(states, t)
            x = basis_all[regression_itm]
            y = continuation_cf[regression_itm]
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            continuation_all = basis_all @ beta
            regression_map[t] = beta
            regression_steps.append(
                LSMRegressionStep(
                    time_index=t,
                    coefficients=beta,
                    n_regression_paths=int(regression_itm.sum()),
                    condition_number=float(np.linalg.cond(x)),
                    continuation_mean=float(np.mean(continuation_all[regression_itm])),
                    intrinsic_mean=float(np.mean(intrinsic[regression_itm])),
                )
            )

        pricing_itm = eligible & pricing_path_mask & (intrinsic > 0.0)
        if beta is not None and continuation_all is not None and np.any(pricing_itm):
            exercise_now_local = intrinsic[pricing_itm] >= continuation_all[pricing_itm]
            exercise_idx = np.where(pricing_itm)[0][exercise_now_local]
            if exercise_idx.size > 0:
                continuation_cf[exercise_idx] = intrinsic[exercise_idx]
                alive[exercise_idx] = False
                stopping_times[exercise_idx] = t
                stopping_payoffs[exercise_idx] = intrinsic[exercise_idx]
                spot = states["spot"][exercise_idx, t]
                exercise_summary.append(
                    {
                        "time_index": float(t),
                        "time": float(t * dt),
                        "exercise_count": float(exercise_idx.size),
                        "boundary_min": float(np.min(spot)),
                        "boundary_max": float(np.max(spot)),
                        "boundary_mean": float(np.mean(spot)),
                    }
                )

    continuation_cf[alive] *= discount
    discounted = continuation_cf
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))

    result = LSMResult(
        price=price,
        stderr=stderr,
        discounted_cashflows=discounted,
        stopping_times=stopping_times,
        stopping_payoffs=stopping_payoffs,
        exercise_summary=sorted(exercise_summary, key=lambda x: x["time_index"]),
        regression_steps=sorted(regression_steps, key=lambda x: x.time_index),
    )
    policy = FittedLSMPolicy(
        dt=dt,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=exercise_mask_fn,
        regression_map=regression_map,
    )
    return result, policy
