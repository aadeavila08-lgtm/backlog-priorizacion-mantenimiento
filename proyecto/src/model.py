"""
Componente predictivo del modelo predictivo-prescriptivo.

Implementacion:
    - Regresion logistica con descenso de gradiente (numpy puro).
    - One-hot encoding manual para variables categoricas.
    - Estandarizacion z-score para variables numericas.
    - Calibracion probabilistica con regresion isotonica (Pool Adjacent
      Violators / PAV) para que las probabilidades reflejen frecuencias
      observadas (ver [8], [9], [15] del informe).
    - Reporta AUC, PR-AUC y Brier score sobre un holdout estratificado.

Si scikit-learn esta instalado, el modulo permite usarlo opcionalmente,
pero la implementacion por defecto es numpy-only para portabilidad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_generator import features_modelo


CATEGORICAL: List[str] = ["tipo_wo", "clase_equipo"]
NUMERIC: List[str] = [c for c in features_modelo() if c not in CATEGORICAL]


# ---------------------------------------------------------------------------
# Preprocesamiento manual
# ---------------------------------------------------------------------------
class Preprocesador:
    """One-hot para categoricas + z-score para numericas. Stateful (fit/transform)."""

    def __init__(self) -> None:
        self.categorias_: Dict[str, List[str]] = {}
        self.media_: Dict[str, float] = {}
        self.desv_:  Dict[str, float] = {}
        self.feature_names_: List[str] = []

     ###FIT

    def fit(self, X: pd.DataFrame) -> "Preprocesador":
        for col in CATEGORICAL:
            self.categorias_[col] = sorted(X[col].dropna().unique().tolist())
        for col in NUMERIC:
            self.media_[col] = float(X[col].mean())
            sd = float(X[col].std(ddof=0))
            self.desv_[col] = sd if sd > 1e-8 else 1.0
        self.feature_names_ = []
        for col in CATEGORICAL:
            for cat in self.categorias_[col]:
                self.feature_names_.append(f"{col}={cat}")
        self.feature_names_.extend(NUMERIC)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        bloques: List[np.ndarray] = []
        for col in CATEGORICAL:
            mat = np.zeros((len(X), len(self.categorias_[col])), dtype=float)
            for j, cat in enumerate(self.categorias_[col]):
                mat[:, j] = (X[col].values == cat).astype(float)
            bloques.append(mat)
        for col in NUMERIC:
            z = (X[col].values - self.media_[col]) / self.desv_[col]
            bloques.append(z.reshape(-1, 1))
        return np.hstack(bloques)


# ---------------------------------------------------------------------------
# Regresion logistica (numpy puro)
# ---------------------------------------------------------------------------
class RegresionLogistica:
    def __init__(self, lr: float = 0.05, n_iter: int = 800,
                 l2: float = 0.01, balanceada: bool = True) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.balanceada = balanceada
        self.w_ = None
        self.b_: float = 0.0

    ###SIGMOID
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
        neg = ~pos
        e = np.exp(z[neg])
        out[neg] = e / (1.0 + e)
        return out

    ###FIT 
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresionLogistica":
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        if self.balanceada:
            n_pos = max(int(y.sum()), 1)
            n_neg = max(n - n_pos, 1)
            sample_w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
        else:
            sample_w = np.ones(n)
        for _ in range(self.n_iter):
            z = X @ self.w_ + self.b_
            p = self._sigmoid(z)
            err = (p - y) * sample_w
            grad_w = X.T @ err / n + self.l2 * self.w_
            grad_b = err.sum() / n
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w_ + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.decision(X))


# ---------------------------------------------------------------------------
# Calibracion isotonica (PAV)
# ---------------------------------------------------------------------------
class CalibradorIsotonico:
    def __init__(self) -> None:
        self.x_ = None
        self.y_ = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "CalibradorIsotonico":
        order = np.argsort(scores)
        x = scores[order].astype(float)
        v = y[order].astype(float).copy()
        w = np.ones_like(v)
        i = 0
        while i < len(v) - 1:
            if v[i] > v[i + 1]:
                new_w = w[i] + w[i + 1]
                new_v = (v[i] * w[i] + v[i + 1] * w[i + 1]) / new_w
                v[i] = new_v
                w[i] = new_w
                v = np.delete(v, i + 1)
                w = np.delete(w, i + 1)
                x = np.delete(x, i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.x_ = x
        self.y_ = v
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self.x_, self.y_)


# ---------------------------------------------------------------------------
# Metricas (numpy puro)
# ---------------------------------------------------------------------------
def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s_sorted = scores[order]
    ranks = np.empty(len(y), dtype=float)
    i = 0
    rank = 1
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[i:j] = avg
        rank += (j - i)
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y_true[order].astype(float)
    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-12)
    recall = tp_cum / n_pos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y_true - p) ** 2))


def split_estratificado(df: pd.DataFrame, y: str,
                        test_size: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in df[y].unique():
        idx = df.index[df[y] == cls].to_numpy(copy=True)
        rng.shuffle(idx)
        n_test = max(int(len(idx) * test_size), 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return df.loc[train_idx], df.loc[test_idx]


@dataclass
class MetricasModelo:
    auc: float
    pr_auc: float
    brier: float
    n_train: int
    n_test: int

    def como_dict(self) -> Dict[str, float]:
        return {
            "AUC":     round(self.auc, 4),
            "PR_AUC":  round(self.pr_auc, 4),
            "Brier":   round(self.brier, 4),
            "n_train": self.n_train,
            "n_test":  self.n_test,
        }


class ModeloPriorizacion:
    def __init__(self, lr: float = 0.05, n_iter: int = 800, l2: float = 0.01) -> None:
        self.pre = Preprocesador()
        self.clf = RegresionLogistica(lr=lr, n_iter=n_iter, l2=l2, balanceada=True)
        self.cal = CalibradorIsotonico()

    def fit(self, df_train: pd.DataFrame, target: str = "falla_7d") -> "ModeloPriorizacion":
        df_fit, df_cal = split_estratificado(df_train, y=target, test_size=0.30)
        X_fit = self.pre.fit(df_fit[features_modelo()]).transform(df_fit[features_modelo()])
        y_fit = df_fit[target].astype(int).values
        self.clf.fit(X_fit, y_fit)
        X_cal = self.pre.transform(df_cal[features_modelo()])
        y_cal = df_cal[target].astype(int).values
        scores_cal = self.clf.predict_proba(X_cal)
        self.cal.fit(scores_cal, y_cal)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.pre.transform(df[features_modelo()])
        scores = self.clf.predict_proba(X)
        return self.cal.transform(scores)


def entrenar_modelo(df_train: pd.DataFrame, target: str = "falla_7d",
                    test_size: float = 0.25, random_state: int = 42):
    df_tr, df_te = split_estratificado(df_train, y=target,
                                        test_size=test_size, seed=random_state)
    modelo = ModeloPriorizacion()
    modelo.fit(df_tr, target=target)
    proba = modelo.predict_proba(df_te)
    y_te = df_te[target].astype(int).values
    metricas = MetricasModelo(
        auc=auc_roc(y_te, proba),
        pr_auc=pr_auc(y_te, proba),
        brier=brier(y_te, proba),
        n_train=len(df_tr),
        n_test=len(df_te),
    )
    return modelo, metricas


def importancia_variables(modelo: ModeloPriorizacion) -> pd.DataFrame:
    nombres = modelo.pre.feature_names_
    coefs = modelo.clf.w_
    return pd.DataFrame({
        "variable": nombres,
        "coeficiente": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_generator import ConfigEscenario, generar_backlog_semanal
    cfg = ConfigEscenario()
    semanas = [generar_backlog_semanal(s, cfg, seed=7) for s in range(1, 21)]
    df_train = pd.concat(semanas, ignore_index=True)
    print("Dataset:", df_train.shape)
    modelo, met = entrenar_modelo(df_train)
    print("Metricas:", met.como_dict())
    print(importancia_variables(modelo).head(10))
