"""
KPIs operativos para evaluar y comparar planes semanales.

Indicadores (ver informe, secciones 2d y 3.3):
    1. riesgo_residual_pct        : % del riesgo total del backlog que NO se programo.
    2. riesgo_residual_abs        : suma de riesgo no programado.
    3. wo_criticas_programadas_pct: % de WO criticas que SI entraron al plan.
    4. wo_criticas_no_ejecutadas  : numero absoluto de WO criticas fuera del plan.
    5. utilizacion_total_pct      : % de horas totales utilizadas.
    6. utilizacion_por_habilidad  : dict habilidad -> %.
    7. cumplimiento_ventanas      : fraccion de WO seleccionadas con ventana >= dia 7.
    8. lateness_ponderado         : suma de (dias_en_backlog * riesgo) de WO no programadas.
    9. violaciones_repuestos      : WO seleccionadas con repuesto no disponible (debe ser 0).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd

from mcda import es_critica
from optimizer import ResultadoPlan


@dataclass
class KPIs:
    riesgo_total_backlog:        float
    riesgo_programado:           float
    riesgo_residual_abs:         float
    riesgo_residual_pct:         float
    wo_total:                    int
    wo_seleccionadas:            int
    wo_criticas_total:           int
    wo_criticas_programadas:     int
    wo_criticas_programadas_pct: float
    wo_criticas_no_ejecutadas_pct: float
    utilizacion_total_pct:       float
    utilizacion_por_habilidad:   Dict[str, float]
    cumplimiento_ventanas_pct:   float
    lateness_ponderado:          float
    violaciones_repuestos:       int

    def como_dict(self) -> Dict:
        d = asdict(self)
        # Redondeo razonable
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
            elif isinstance(v, dict):
                d[k] = {kk: round(vv, 4) for kk, vv in v.items()}
        return d


def calcular_kpis(
    df_score: pd.DataFrame,
    plan: ResultadoPlan,
    horas_total: float,
    horas_habilidad: Dict[str, float],
) -> KPIs:
    df = df_score.reset_index(drop=True)
    sel = plan.seleccion.astype(bool)

    # Riesgo
    riesgo = df["riesgo"].values
    riesgo_total = float(riesgo.sum())
    riesgo_prog = float(riesgo[sel].sum())
    riesgo_res = riesgo_total - riesgo_prog
    riesgo_res_pct = 100.0 * riesgo_res / max(riesgo_total, 1e-9)

    # WO criticas
    crit = es_critica(df)
    n_crit = int(crit.sum())
    n_crit_prog = int((crit & sel).sum())
    pct_prog = 100.0 * n_crit_prog / max(n_crit, 1)
    pct_no_ej = 100.0 - pct_prog

    # Utilizacion
    util_total = 100.0 * plan.horas_usadas / max(horas_total, 1e-9)
    util_hab = {}
    for h, cap in horas_habilidad.items():
        usada = plan.horas_por_habilidad.get(h, 0.0)
        util_hab[h] = round(100.0 * usada / max(cap, 1e-9), 2)

    # Ventanas: WO con ventana_dia_limite <= 7 que se programaron
    cumple_vent = ((df["ventana_dia_limite"].values <= 7) & sel)
    cumplimiento = 100.0 * cumple_vent.sum() / max(sel.sum(), 1)

    # Lateness ponderado: para WO no seleccionadas, dias_backlog * riesgo
    no_sel = ~sel
    lateness = float((df["dias_en_backlog"].values[no_sel] *
                      df["riesgo"].values[no_sel]).sum())

    # Violaciones de repuestos: deberia ser 0 si los solvers son correctos
    rep_id = df["repuesto_id"].values
    disp = df["repuesto_disponible"].values
    violaciones = int(((rep_id != -1) & (~disp) & sel).sum())

    return KPIs(
        riesgo_total_backlog=riesgo_total,
        riesgo_programado=riesgo_prog,
        riesgo_residual_abs=riesgo_res,
        riesgo_residual_pct=riesgo_res_pct,
        wo_total=len(df),
        wo_seleccionadas=int(sel.sum()),
        wo_criticas_total=n_crit,
        wo_criticas_programadas=n_crit_prog,
        wo_criticas_programadas_pct=pct_prog,
        wo_criticas_no_ejecutadas_pct=pct_no_ej,
        utilizacion_total_pct=util_total,
        utilizacion_por_habilidad=util_hab,
        cumplimiento_ventanas_pct=cumplimiento,
        lateness_ponderado=lateness,
        violaciones_repuestos=violaciones,
    )


def comparar(kpi_baseline: KPIs, kpi_modelo: KPIs) -> Dict[str, float]:
    """Calcula la mejora del modelo respecto al baseline."""
    def delta_pct(b, m):
        return 100.0 * (b - m) / max(abs(b), 1e-9)
    return {
        "reduccion_riesgo_residual_pct": round(delta_pct(
            kpi_baseline.riesgo_residual_abs,
            kpi_modelo.riesgo_residual_abs), 2),
        "delta_wo_criticas_pct": round(
            kpi_modelo.wo_criticas_programadas_pct -
            kpi_baseline.wo_criticas_programadas_pct, 2),
        "delta_utilizacion_pct": round(
            kpi_modelo.utilizacion_total_pct -
            kpi_baseline.utilizacion_total_pct, 2),
    }


if __name__ == "__main__":
    from data_generator import ConfigEscenario, generar_backlog_semanal
    from mcda import calcular_riesgo
    from optimizer import resolver_plan_semanal
    from baseline import baseline_prioridad_fecha
    cfg = ConfigEscenario()
    df = generar_backlog_semanal(1, cfg, seed=42)
    df_score = calcular_riesgo(df, df["prob_real_falla_7d"].values)
    plan_b = baseline_prioridad_fecha(df_score, cfg.horas_totales, cfg.horas_por_habilidad)
    plan_m = resolver_plan_semanal(df_score, cfg.horas_totales, cfg.horas_por_habilidad)
    kpi_b = calcular_kpis(df_score, plan_b, cfg.horas_totales, cfg.horas_por_habilidad)
    kpi_m = calcular_kpis(df_score, plan_m, cfg.horas_totales, cfg.horas_por_habilidad)
    print("BASELINE:", kpi_b.como_dict())
    print("\nMODELO:  ", kpi_m.como_dict())
    print("\nMEJORA:  ", comparar(kpi_b, kpi_m))
