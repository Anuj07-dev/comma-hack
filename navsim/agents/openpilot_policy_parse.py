"""
Parse comma/openpilot ``driving_policy`` ONNX flat output like ``selfdrive/modeld`` does.

The policy exports one vector (e.g. shape ``(1, 1000)``). ``driving_policy_metadata.pkl``
provides ``output_slices``; the ``plan`` slice is an MDN blob decoded with the same
logic as ``openpilot.selfdrive.modeld.parse_model_outputs.Parser.parse_policy_outputs``.

This module is self-contained (no openpilot import) so NAVSIM runs without the
comma tree on ``PYTHONPATH``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Mirrors openpilot.selfdrive.modeld.constants.ModelConstants (subset).
IDX_N = 33
PLAN_WIDTH = 15
PLAN_MHP_N = 5
PLAN_MHP_SELECTION = 1
DESIRE_PRED_WIDTH = 8


class Plan:
    """Per-timestep layout of the 15-D plan row (same as comma ``constants.Plan``)."""

    POSITION = slice(0, 3)
    VELOCITY = slice(3, 6)
    ACCELERATION = slice(6, 9)
    T_FROM_CURRENT_EULER = slice(9, 12)
    ORIENTATION_RATE = slice(12, 15)


def safe_exp(x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(np.clip(x, -np.inf, 11), out=out)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + safe_exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    if x.dtype == np.float32 or x.dtype == np.float64:
        safe_exp(x, out=x)
    else:
        x = safe_exp(x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


class CommaPolicyOutputParser:
    """Same behavior as ``openpilot.selfdrive.modeld.parse_model_outputs.Parser`` for policy heads."""

    def __init__(self, ignore_missing: bool = False) -> None:
        self.ignore_missing = ignore_missing

    def check_missing(self, outs: Dict[str, np.ndarray], name: str) -> bool:
        missing = name not in outs
        if missing and not self.ignore_missing:
            raise ValueError(f"Missing output {name}")
        return missing

    def parse_categorical_crossentropy(
        self, name: str, outs: Dict[str, np.ndarray], out_shape: Optional[tuple] = None
    ) -> None:
        if self.check_missing(outs, name):
            return
        raw = outs[name]
        if out_shape is not None:
            raw = raw.reshape((raw.shape[0],) + out_shape)
        outs[name] = softmax(raw, axis=-1)

    def parse_mdn(
        self,
        name: str,
        outs: Dict[str, np.ndarray],
        in_N: int = 0,
        out_N: int = 1,
        out_shape: Optional[tuple] = None,
    ) -> None:
        if self.check_missing(outs, name):
            return
        raw = outs[name]
        raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

        n_values = (raw.shape[2] - out_N) // 2
        pred_mu = raw[:, :, :n_values]
        pred_std = safe_exp(raw[:, :, n_values : 2 * n_values])

        if in_N > 1:
            weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
            for i in range(out_N):
                weights[:, :, i - out_N] = softmax(raw[:, :, i - out_N], axis=-1)

            if out_N == 1:
                for fidx in range(weights.shape[0]):
                    idxs = np.argsort(weights[fidx][:, 0])[::-1]
                    weights[fidx] = weights[fidx][idxs]
                    pred_mu[fidx] = pred_mu[fidx][idxs]
                    pred_std[fidx] = pred_std[fidx][idxs]
            full_shape = tuple([raw.shape[0], in_N] + list(out_shape))
            outs[name + "_weights"] = weights
            outs[name + "_hypotheses"] = pred_mu.reshape(full_shape)
            outs[name + "_stds_hypotheses"] = pred_std.reshape(full_shape)

            pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
            for fidx in range(weights.shape[0]):
                for hidx in range(out_N):
                    idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                    pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
                    pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
        else:
            pred_mu_final = pred_mu
            pred_std_final = pred_std

        if out_N > 1:
            final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
        else:
            final_shape = tuple([raw.shape[0]] + list(out_shape))
        outs[name] = pred_mu_final.reshape(final_shape)
        outs[name + "_stds"] = pred_std_final.reshape(final_shape)

    def is_mhp(self, outs: Dict[str, np.ndarray], name: str, shape: int) -> bool:
        if self.check_missing(outs, name):
            return False
        if outs[name].shape[1] == 2 * shape:
            return False
        return True

    def parse_policy_outputs(self, outs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        plan_mhp = self.is_mhp(outs, "plan", IDX_N * PLAN_WIDTH)
        plan_in_N, plan_out_N = (PLAN_MHP_N, PLAN_MHP_SELECTION) if plan_mhp else (0, 0)
        self.parse_mdn(
            "plan",
            outs,
            in_N=plan_in_N,
            out_N=plan_out_N,
            out_shape=(IDX_N, PLAN_WIDTH),
        )
        if "planplus" in outs:
            self.parse_mdn(
                "planplus",
                outs,
                in_N=0,
                out_N=0,
                out_shape=(IDX_N, PLAN_WIDTH),
            )
        self.parse_categorical_crossentropy(
            "desire_state",
            outs,
            out_shape=(DESIRE_PRED_WIDTH,),
        )
        return outs


def slice_flat_policy_output(
    flat: np.ndarray, output_slices: Dict[str, slice]
) -> Dict[str, np.ndarray]:
    """
    Same layout as ``ModelState.slice_outputs`` in comma ``modeld``:
    each head is shape ``(1, L)``.
    """
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    return {k: flat[v][np.newaxis, :].astype(np.float32) for k, v in output_slices.items()}


def plan_to_xy_heading(plan_batch: np.ndarray) -> np.ndarray:
    """
    :param plan_batch: ``(1, IDX_N, PLAN_WIDTH)`` parsed plan
    :return: ``(IDX_N, 3)`` float64 array: x, y, yaw (from euler third component)
    """
    plan = np.asarray(plan_batch[0], dtype=np.float64)
    pos = plan[:, Plan.POSITION]
    euler = plan[:, Plan.T_FROM_CURRENT_EULER]
    yaw = euler[:, 2]
    return np.column_stack([pos[:, 0], pos[:, 1], yaw])
