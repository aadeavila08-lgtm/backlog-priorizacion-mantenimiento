"""
Componente MCDA (Multi-Criteria Decision Analysis).

Combina la probabilidad calibrada de falla en 7 dias, la severidad
agregada (seguridad/ambiente/produccion) y el costo/penalidad por
repuesto en un puntaje normalizado de riesgo por WO.

Formula base (informe, seccion 2d):
    Riesgo_WO = w_prob * Prob + w_sev * Severidad + w_costo * Costo

Con pesos por defecto (0.5, 0.3, 0.2) y suma = 1.0.
Los pesos son parametrizables por escenario (ej. Escenario 4 enfatiza
seguridad).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PesosMCDA:
    """Pesos del puntaje MCDA. Deben sumar 1.0."""
    prob:  float = 0.50
    sev:   float = 0.30
    costo: float = 0.20

    # Sub-pesos dentro de severidad (alineados con RCM e ISO 31000)
    sev_seguridad:  float = 0.50
    sev_ambiental:  float = 0.20
    sev_produccion: float = 0.30

    def validar(self) -> None:
        s_main = self.prob + self.sev + self.costo
        if abs(s_main - 1.0) > 1e-6:
            raise ValueError(f"Pesos MCDA principales deben sumar 1.0 (actual={s_main})")
        s_sev = self.sev_seguridad + self.sev_ambiental + self.sev_produccion
        if abs(s_sev - 1.0) > 1e-6:
            raise ValueError(f"Sub-pesos severidad deben sumar 1.0 (actual={s_sev})")

    def como_dict(self) -> Dict[str, float]:
        return {
            "w_prob":          self.prob,
            "w_severidad":     self.sev,
            "w_costo":         self.costo,
            "w_sev_seguridad": self.sev_seguridad,
            "w_sev_ambiental": self.sev_ambiental,
            "w_sev_produccion":self.sev_produccion,
        }


def severidad_agregada(df: pd.DataFrame, pesos: PesosMCDA) -> np.ndarray:
    """Severidad combinada [0-1] como promedio ponderado de las 3 dimensiones."""
    return (pesos.sev_seguridad  * df["sev_seguridad"].values
          + pesos.sev_ambiental  * df["sev_ambiental"].values
          + pesos.sev_produccion * df["sev_produccion"].values)


def costo_normalizado(df: pd.DataFrame) -> np.ndarray:
    """Costo / penalidad por repuesto, normalizado a [0-1].

    Heuristica:
        * Si la WO no requiere repuesto (id == -1): costo = 0.10 base.
        * Si lo requiere y esta disponible: costo proporcional al lead-time.
        * Si lo requiere y NO esta disponible: 1.0 (penalidad maxima).
    """
    n = len(df)
    out = np.full(n, 0.10)
    requiere = df["repuesto_id"].values != -1
    disponible = df["repuesto_disponible"].values
    lead = df["lead_time_repuesto"].values

    # WO con repuesto disponible: penalidad por lead time (max 90 dias)
    mask_disp = requiere & disponible
    out[mask_disp] = np.clip(0.20 + 0.50 * lead[mask_disp] / 90.0, 0.20, 0.80)

    # WO sin disponibilidad: penalidad maxima
    mask_nd = requiere & (~disponible)
    out[mask_nd] = 1.0
    return out


def calcular_riesgo(
    df: pd.DataFrame,
    prob_calibrada: np.ndarray,
    pesos: PesosMCDA = None,
) -> pd.DataFrame:
    """Devuelve el DataFrame ampliado con 'severidad', 'costo' y 'riesgo'."""
    if pesos is None:
        pesos = PesosMCDA()
    pesos.validar()

    sev = severidad_agregada(df, pesos)
    cost = costo_normalizado(df)
    riesgo = pesos.prob * prob_calibrada + pesos.sev * sev + pesos.costo * cost

    out = df.copy()
    out["prob_calibrada"] = np.round(prob_calibrada, 4)
    out["severidad"]      = np.round(sev, 4)
    out["costo"]          = np.round(cost, 4)
    out["riesgo"]         = np.round(np.clip(riesgo, 0, 1), 4)
    return out


def es_critica(df: pd.DataFrame, umbral_criticidad: int = 4,
               umbral_seguridad: float = 0.6) -> np.ndarray:
    """Una WO se considera 'critica' si:
        - su criticidad nominal >= umbral_criticidad, O
        - su severidad de seguridad >= umbral_seguridad.
    """
    return ((df["criticidad"].values >= umbral_criticidad)
          | (df["sev_seguridad"].values >= umbral_seguridad))


if __name__ == "__main__":
    from data_generator import ConfigEscenario, generar_backlog_semanal
    cfg = ConfigEscenario()
    df = generar_backlog_semanal(1, cfg, seed=42)
    # En esta prueba usamos prob_real como sustituto de la calibrada
    df_score = calcular_riesgo(df, df["prob_real_falla_7d"].values)
    print(df_score[["wo_id", "criticidad", "prob_calibrada",
                     "severidad", "costo", "riesgo"]].head())
    print("\nWO criticas:", es_critica(df).sum(), "/", len(df))
