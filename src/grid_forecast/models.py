"""The two learned models, each with the same ``fit(train, valid)`` and ``predict(frame)`` surface.

LightGBM: one model per series, the horizon hour is a feature (direct multi-horizon), fixed seed, early
stopping on the last fourteen days of the training window with at most 2,000 rounds, then a refit on the
whole window with the round count the early stopping found.

GRU: one small PyTorch network per series on the CPU. Input: the last 168 known hours of the target series
(standardised with training statistics) through a one-layer GRU with hidden size 64, joined with the calendar
of the 24 target hours, two linear layers, 24 outputs. Adam with learning rate 2e-3 and batches of 16 origins
(a comparison on seven folds, recorded in ``results/gru_settings_check.json``, showed that batches of 32 at
1e-3 gave the early folds too few gradient steps), at most 20 epochs, early stopping on the last fourteen
days, fixed seed, deterministic algorithms, then a refit for the epoch count found.

Both can be wrapped as an MLflow pyfunc that takes the feature frame and returns one value per row.
"""
from __future__ import annotations

import math
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import SEED
from .features import FEATURE_COLUMNS, HORIZON, WINDOW_COLUMNS

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
}


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["target_missing"].to_numpy(dtype=bool)]


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, series: str, max_rounds: int = 2000, early_stopping_rounds: int = 100, seed: int = SEED, params: dict | None = None):
        self.series = series
        self.features = list(FEATURE_COLUMNS[series])
        self.max_rounds = int(max_rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.seed = int(seed)
        self.params = {**LGBM_PARAMS, **(params or {}), "seed": self.seed}
        self.booster: lgb.Booster | None = None
        self.best_iteration = 0
        self.fit_seconds = 0.0

    def settings(self) -> dict:
        return {"max_rounds": self.max_rounds, "early_stopping_rounds": self.early_stopping_rounds, "seed": self.seed,
                "learning_rate": self.params["learning_rate"], "num_leaves": self.params["num_leaves"],
                "min_data_in_leaf": self.params["min_data_in_leaf"], "feature_fraction": self.params["feature_fraction"],
                "bagging_fraction": self.params["bagging_fraction"], "lambda_l2": self.params["lambda_l2"], "n_features": len(self.features)}

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        tr = _scored(train)
        va = _scored(valid)
        dtrain = lgb.Dataset(tr[self.features], tr["target"].to_numpy(dtype=float), free_raw_data=False)
        dvalid = lgb.Dataset(va[self.features], va["target"].to_numpy(dtype=float), reference=dtrain, free_raw_data=False)
        probe = lgb.train(self.params, dtrain, num_boost_round=self.max_rounds, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        best = int(probe.best_iteration) if probe.best_iteration else self.max_rounds
        both = pd.concat([tr, va], ignore_index=True)
        dall = lgb.Dataset(both[self.features], both["target"].to_numpy(dtype=float), free_raw_data=False)
        self.booster = lgb.train(self.params, dall, num_boost_round=best)
        self.best_iteration = best
        self.fit_seconds = round(time.perf_counter() - t0, 2)
        return {"best_iteration": best, "fit_seconds": self.fit_seconds, "train_rows": int(len(tr)), "valid_rows": int(len(va))}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("fit first")
        return self.booster.predict(frame[self.features], num_iteration=self.booster.num_trees())


def _calendar_tensor(frame: pd.DataFrame) -> np.ndarray:
    h = frame["hour"].to_numpy(dtype=float)
    d = frame["dow"].to_numpy(dtype=float)
    return np.stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
        frame["holiday"].to_numpy(dtype=float), frame["dst_day"].to_numpy(dtype=float),
        frame["horizon"].to_numpy(dtype=float) / 48.0,
    ], axis=1)


CAL_DIM = 7


class GRUNet(torch.nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + HORIZON * CAL_DIM, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, HORIZON),
        )

    def forward(self, window: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(window)
        z = torch.cat([h[-1], calendar.reshape(calendar.shape[0], -1)], dim=1)
        return self.head(z)


class GRUForecaster:
    name = "gru"

    def __init__(self, series: str, hidden: int = 64, epochs: int = 20, patience: int = 3, lr: float = 2e-3, batch_size: int = 16, seed: int = SEED):
        self.series = series
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.net: GRUNet | None = None
        self.mu = 0.0
        self.sigma = 1.0
        self.best_epoch = 0
        self.fit_seconds = 0.0

    def settings(self) -> dict:
        return {"hidden": self.hidden, "max_epochs": self.epochs, "patience": self.patience, "lr": self.lr,
                "batch_size": self.batch_size, "seed": self.seed, "window_hours": len(WINDOW_COLUMNS)}

    @staticmethod
    def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per origin: window (N,168,1), calendar (N,24,7), target (N,24), scored mask (N,24). Frame order is kept.

        A frame to predict on may carry no target at all (the API hands the registered model the input
        columns only), in which case the target comes back as NaN and the mask as all False.
        """
        n = len(frame) // HORIZON
        if n * HORIZON != len(frame):
            raise ValueError("frame must hold whole 24-row origins")
        win = frame[WINDOW_COLUMNS].to_numpy(dtype=float).reshape(n, HORIZON, -1)[:, 0, :][:, :, None]
        cal = _calendar_tensor(frame).reshape(n, HORIZON, CAL_DIM)
        if "target" in frame.columns:
            y = frame["target"].to_numpy(dtype=float).reshape(n, HORIZON)
            if "target_missing" in frame.columns:
                missing = frame["target_missing"].to_numpy(dtype=bool)
            else:
                missing = np.isnan(frame["target"].to_numpy(dtype=float))
            mask = ~missing.reshape(n, HORIZON)
        else:
            y = np.full((n, HORIZON), np.nan)
            mask = np.zeros((n, HORIZON), dtype=bool)
        return win, cal, y, mask

    def _seed(self) -> torch.Generator:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def _train(self, win, cal, y, mask, epochs: int, valid: tuple | None) -> tuple[GRUNet, int, list[float]]:
        g = self._seed()
        net = GRUNet(self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        W = torch.tensor((win - self.mu) / self.sigma, dtype=torch.float32)
        C = torch.tensor(cal, dtype=torch.float32)
        Y = torch.tensor((np.nan_to_num(y, nan=self.mu) - self.mu) / self.sigma, dtype=torch.float32)
        M = torch.tensor(mask, dtype=torch.float32)
        n = W.shape[0]
        history: list[float] = []
        best_state = None
        best_loss = math.inf
        best_epoch = 0
        bad = 0
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                b = order[i: i + self.batch_size]
                opt.zero_grad()
                out = net(W[b], C[b])
                loss = (((out - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1.0)
                loss.backward()
                opt.step()
            if valid is None:
                history.append(float("nan"))
                best_epoch = epoch
                continue
            vw, vc, vy, vm = valid
            net.eval()
            with torch.no_grad():
                VW = torch.tensor((vw - self.mu) / self.sigma, dtype=torch.float32)
                VC = torch.tensor(vc, dtype=torch.float32)
                VY = torch.tensor((np.nan_to_num(vy, nan=self.mu) - self.mu) / self.sigma, dtype=torch.float32)
                VM = torch.tensor(vm, dtype=torch.float32)
                vout = net(VW, VC)
                vloss = float(((((vout - VY) ** 2) * VM).sum() / VM.sum().clamp(min=1.0)).item())
            history.append(vloss)
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch, history

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        win, cal, y, mask = self._groups(train)
        vwin, vcal, vy, vmask = self._groups(valid)
        scored = y[mask]
        self.mu = float(np.mean(scored)) if len(scored) else 0.0
        self.sigma = float(np.std(scored)) if len(scored) and np.std(scored) > 0 else 1.0
        _, best_epoch, history = self._train(win, cal, y, mask, self.epochs, (vwin, vcal, vy, vmask))
        best_epoch = max(1, best_epoch)
        all_win = np.concatenate([win, vwin])
        all_cal = np.concatenate([cal, vcal])
        all_y = np.concatenate([y, vy])
        all_mask = np.concatenate([mask, vmask])
        self.net, _, _ = self._train(all_win, all_cal, all_y, all_mask, best_epoch, None)
        self.net.eval()
        self.best_epoch = best_epoch
        self.fit_seconds = round(time.perf_counter() - t0, 2)
        return {"best_epoch": best_epoch, "fit_seconds": self.fit_seconds, "train_origins": int(win.shape[0]), "valid_origins": int(vwin.shape[0]),
                "valid_history": [round(h, 5) for h in history]}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit first")
        win, cal, _, _ = self._groups(frame)
        self.net.eval()
        with torch.no_grad():
            W = torch.tensor((win - self.mu) / self.sigma, dtype=torch.float32)
            C = torch.tensor(cal, dtype=torch.float32)
            out = self.net(W, C).numpy()
        return (out * self.sigma + self.mu).reshape(-1)


MODELS = {"lightgbm": LightGBMForecaster, "gru": GRUForecaster}


def make_model(name: str, series: str, **kwargs):
    return MODELS[name](series, **kwargs)
