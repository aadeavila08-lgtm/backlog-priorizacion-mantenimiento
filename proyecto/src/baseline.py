"""
Baseline: reglas tradicionales de programacion.

Implementa la politica que se usa actualmente en muchos CMMS:
ordenar por prioridad nominal (descendente) y luego por dias en backlog
(antiguedad), llenando el plan respetando las mismas restricciones de
capacidad y disponibilidad de repuestos que el modelo MILP.

Esta es la politica contra la que se compara el modelo predictivo-
prescriptivo (objetivo especifico 4 del informe).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from optimizer import ResultadoPlan, _empacar_resultado


def baseline_prioridad_fecha(
    df: pd.DataFrame,
    horas_total: float,
    horas_habilidad: Dict[str, float],
    on_hand: Optional[Dict[int, int]] = None,
) -> ResultadoPlan:
    """Programa por prioridad_nominal DESC, luego dias_en_backlog DESC."""
    df = df.reset_index(drop=True)
    if on_hand is None:
        ids = sorted(set(df["repuesto_id"].tolist()) - {-1})
        on_hand = {r: 1 for r in ids}

    horas = df["duracion_h"].values
    habs = df["habilidad_requerida"].values
    rep = df["repuesto_id"].values
    disp = df["repuesto_disponible"].values

    # Orden tradicional: prioridad alta primero, luego mas viejo primero.
    orden = np.lexsort((
        -df["dias_en_backlog"].values,
        -df["prioridad_nominal"].values,
    ))

    sel = np.zeros(len(df), dtype=int)
    h_total = 0.0
    h_hab = {h: 0.0 for h in horas_habilidad}
    rep_uso = {r: 0 for r in on_hand}

    for i in orden:
        # Excluir repuestos no disponibles
        if rep[i] != -1 and not disp[i]:
            continue
        if h_total + horas[i] > horas_total + 1e-9:
            continue
        hab = habs[i]
        if h_hab.get(hab, 0.0) + horas[i] > horas_habilidad.get(hab, 0.0) + 1e-9:
            continue
        r = int(rep[i])
        if r != -1 and rep_uso.get(r, 0) + 1 > on_hand.get(r, 0):
            continue
        sel[i] = 1
        h_total += horas[i]
        h_hab[hab] = h_hab.get(hab, 0.0) + horas[i]
        if r != -1:
            rep_uso[r] = rep_uso.get(r, 0) + 1

    return _empacar_resultado(df, sel, "baseline_prioridad_fecha")


if __name__ == "__main__":
    from data_generator import ConfigEscenario, generar_backlog_semanal
    from mcda import calcular_riesgo
    cfg = ConfigEscenario()
    df = generar_backlog_semanal(1, cfg, seed=42)
    df_score = calcular_riesgo(df, df["prob_real_falla_7d"].values)
    res = baseline_prioridad_fecha(df_score, cfg.horas_totales,
                                    cfg.horas_por_habilidad)
    print("Baseline:", res.como_dict())
