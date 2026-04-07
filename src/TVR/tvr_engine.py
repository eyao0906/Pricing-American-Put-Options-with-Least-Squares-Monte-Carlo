from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
PayoffFn = Callable[[dict[str, Array], int], Array]
BasisFn = Callable[[dict[str, Array], int], Array]
ExerciseMaskFn = Callable[[dict[str, Array], int], Array]


@dataclass
class TVRStep:
    time_index: int
    coefficients: Array
    n_regression_paths: int
    condition_number: float


@dataclass
class TVRResult:
    price: float
    stderr: float
    stopping_times: Array
    discounted_cashflows: Array
    value_paths: Array
    exercise_summary: list[dict[str, float]]
    regression_steps: list[TVRStep]


class FittedTVRPolicy:
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
        stopping_times = np.full(n_paths, steps, dtype=int)
        stopping_payoffs = self.payoff_fn(states, steps).copy()
        exercised = np.zeros(n_paths, dtype=bool)

        for t in range(1, steps):
            alive = ~exercised
            eligible = self.exercise_mask_fn(states, t) & alive
            if not np.any(eligible):
                continue
            beta = self.regression_map.get(t)
            if beta is None:
                continue
            intrinsic = self.payoff_fn(states, t)
            basis = self.basis_fn(states, t)
            continuation = basis @ beta
            exercise_now = eligible & (intrinsic >= continuation)
            if np.any(exercise_now):
                stopping_times[exercise_now] = t
                stopping_payoffs[exercise_now] = intrinsic[exercise_now]
                exercised[exercise_now] = True

        discounted = np.exp(-r * self.dt * stopping_times) * stopping_payoffs
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
        return price, stderr, stopping_times, discounted


def fit_tvr(
    states: dict[str, Array],
    payoff_fn: PayoffFn,
    basis_fn: BasisFn,
    exercise_mask_fn: ExerciseMaskFn,
    r: float,
    maturity: float,
    min_regression_paths: int = 200,
) -> tuple[TVRResult, FittedTVRPolicy]:
    n_paths, n_steps_plus_one = next(iter(states.values())).shape
    steps = n_steps_plus_one - 1
    dt = maturity / steps
    discount = np.exp(-r * dt)

    value_paths = np.zeros((n_paths, n_steps_plus_one), dtype=float)
    value_paths[:, -1] = payoff_fn(states, steps)
    regression_map: dict[int, Array] = {}
    regression_steps: list[TVRStep] = []

    for t in range(steps - 1, -1, -1):
        intrinsic = payoff_fn(states, t)
        eligible = exercise_mask_fn(states, t)
        basis = basis_fn(states, t)
        y = discount * value_paths[:, t + 1]
        regression_paths = eligible
        if np.sum(regression_paths) >= min_regression_paths:
            x = basis[regression_paths]
            target = y[regression_paths]
            beta, *_ = np.linalg.lstsq(x, target, rcond=None)
            regression_map[t] = beta
            regression_steps.append(
                TVRStep(
                    time_index=t,
                    coefficients=beta,
                    n_regression_paths=int(np.sum(regression_paths)),
                    condition_number=float(np.linalg.cond(x)),
                )
            )
            continuation = basis @ beta
        else:
            continuation = y

        if t == 0:
            value_paths[:, t] = continuation
        else:
            exercise_value = np.where(eligible, intrinsic, -np.inf)
            value_paths[:, t] = np.maximum(exercise_value, continuation)

    fitted_policy = FittedTVRPolicy(
        dt=dt,
        payoff_fn=payoff_fn,
        basis_fn=basis_fn,
        exercise_mask_fn=exercise_mask_fn,
        regression_map=regression_map,
    )
    in_sample_price, in_sample_stderr, stopping_times, discounted = fitted_policy.evaluate(states, r)

    exercise_summary: list[dict[str, float]] = []
    for t in range(1, steps):
        idx = np.where(stopping_times == t)[0]
        if idx.size > 0:
            spot = states["spot"][idx, t]
            exercise_summary.append(
                {
                    "time_index": float(t),
                    "time": float(t * dt),
                    "exercise_count": float(idx.size),
                    "boundary_min": float(np.min(spot)),
                    "boundary_max": float(np.max(spot)),
                    "boundary_mean": float(np.mean(spot)),
                }
            )

    result = TVRResult(
        price=in_sample_price,
        stderr=in_sample_stderr,
        stopping_times=stopping_times,
        discounted_cashflows=discounted,
        value_paths=value_paths,
        exercise_summary=exercise_summary,
        regression_steps=sorted(regression_steps, key=lambda x: x.time_index),
    )
    return result, fitted_policy
