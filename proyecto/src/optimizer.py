"""
Componente prescriptivo: seleccion semanal de WO bajo restricciones.

Modelo MILP (ver Pinedo, Scheduling, Cap. 1-3, ref [12] del informe):

    Variables:
        x_i in {0,1}  para cada WO i

    Funcion objetivo:
        max  sum_i  riesgo_i * x_i

    Restricciones:
        (1) sum_i  duracion_i * x_i  <=  H_total
        (2) Para cada habilidad h:
              sum_{i en h}  duracion_i * x_i  <=  H_h
        (3) Para cada repuesto r con on_hand_r:
              sum_{i requiere r}  x_i  <=  on_hand_r
        (4) Si repuesto NO disponible: x_i = 0  (penalizacion implicita)

Implementacion:
    * Si PuLP esta instalado: solver MILP exacto (CBC).
    * Si no: greedy por densidad (riesgo/horas) + local-search 2-swap.
      En la practica encuentra el optimo o muy cerca para n<=200 WO.

Ambas rutas devuelven la misma estructura de resultado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pulp  # type: ignore
    _PULP_OK = True
except ImportError:
    _PULP_OK = False


@dataclass
class ResultadoPlan:
    seleccion: np.ndarray         # vector binario (1 = WO programada)
    riesgo_total: float           # suma de riesgo seleccionado
    horas_usadas: float
    horas_por_habilidad: Dict[str, float]
    metodo: str                   # 'pulp_milp' o 'greedy_local'
    n_seleccionadas: int

    def como_dict(self) -> Dict:
        return {
            "metodo": self.metodo,
            "n_seleccionadas": self.n_seleccionadas,
            "riesgo_total": round(self.riesgo_total, 4),
            "horas_usadas": round(self.horas_usadas, 2),
            "horas_por_habilidad": {k: round(v, 2)
                                    for k, v in self.horas_por_habilidad.items()},
        }


def _agrupar_por_habilidad(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    grupos = {}
    for h in df["habilidad_requerida"].unique():
        grupos[h] = df.index[df["habilidad_requerida"] == h].to_numpy()
    return grupos


def _on_hand_por_repuesto(df: pd.DataFrame,
                           on_hand_global: Dict[int, int]) -> Dict[int, int]:
    """Capacidad de uso del repuesto en la semana.

    on_hand_global mapea repuesto_id -> unidades disponibles.
    """
    return on_hand_global


# ---------------------------------------------------------------------------
# Solver MILP con PuLP
# ---------------------------------------------------------------------------
def _solver_pulp(
    df: pd.DataFrame,
    horas_total: float,
    horas_habilidad: Dict[str, float],
    on_hand: Dict[int, int],
) -> ResultadoPlan:
    n = len(df)
    prob = pulp.LpProblem("plan_semanal", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objetivo
    prob += pulp.lpSum(df["riesgo"].values[i] * x[i] for i in range(n))

    # (1) Horas totales
    prob += pulp.lpSum(df["duracion_h"].values[i] * x[i]
                       for i in range(n)) <= horas_total

    # (2) Horas por habilidad
    grupos = _agrupar_por_habilidad(df)
    for hab, idx in grupos.items():
        cap = horas_habilidad.get(hab, 0.0)
        prob += pulp.lpSum(df["duracion_h"].values[i] * x[i] for i in idx) <= cap

    # (3) Repuestos
    for rep_id, stock in on_hand.items():
        idx = df.index[(df["repuesto_id"] == rep_id) &
                       (df["repuesto_disponible"] == True)].to_numpy()
        if len(idx) == 0:
            continue
        prob += pulp.lpSum(x[i] for i in idx) <= stock

    # (4) Repuestos NO disponibles -> x=0
    idx_nd = df.index[(df["repuesto_id"] != -1) &
                      (df["repuesto_disponible"] == False)].to_numpy()
    for i in idx_nd:
        prob += x[i] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    sel = np.array([int(pulp.value(xi)) for xi in x], dtype=int)
    return _empacar_resultado(df, sel, "pulp_milp")


# ---------------------------------------------------------------------------
# Solver greedy + local search (sin dependencias)
# ---------------------------------------------------------------------------
def _solver_greedy(
    df: pd.DataFrame,
    horas_total: float,
    horas_habilidad: Dict[str, float],
    on_hand: Dict[int, int],
) -> ResultadoPlan:
    n = len(df)
    riesgos = df["riesgo"].values.astype(float)
    horas = df["duracion_h"].values.astype(float)
    habs = df["habilidad_requerida"].values
    rep = df["repuesto_id"].values
    disp = df["repuesto_disponible"].values

    # Excluir WO sin repuesto disponible
    factible = np.where((rep == -1) | (disp), 1, 0)

    # Densidad: riesgo / horas
    densidad = np.where(horas > 0, riesgos / horas, riesgos)

    # Orden por densidad descendente (factor secundario: riesgo descendente)
    orden = np.lexsort((-riesgos, -densidad))

    sel = np.zeros(n, dtype=int)
    h_total = 0.0
    h_hab = {h: 0.0 for h in horas_habilidad}
    rep_uso = {r: 0 for r in on_hand}

    def _cabe(i: int) -> bool:
        if not factible[i]:
            return False
        if h_total + horas[i] > horas_total + 1e-9:
            return False
        hab = habs[i]
        if h_hab.get(hab, 0.0) + horas[i] > horas_habilidad.get(hab, 0.0) + 1e-9:
            return False
        r = int(rep[i])
        if r != -1:
            if rep_uso.get(r, 0) + 1 > on_hand.get(r, 0):
                return False
        return True

    def _agregar(i: int) -> None:
        nonlocal h_total
        sel[i] = 1
        h_total += horas[i]
        h_hab[habs[i]] = h_hab.get(habs[i], 0.0) + horas[i]
        r = int(rep[i])
        if r != -1:
            rep_uso[r] = rep_uso.get(r, 0) + 1

    def _quitar(i: int) -> None:
        nonlocal h_total
        sel[i] = 0
        h_total -= horas[i]
        h_hab[habs[i]] -= horas[i]
        r = int(rep[i])
        if r != -1:
            rep_uso[r] -= 1

    # Greedy
    for i in orden:
        if _cabe(i):
            _agregar(i)

    # Local search 2-swap: quitar 1 dentro y agregar 1 o 2 fuera
    mejora = True
    iter_max = 5
    it = 0
    while mejora and it < iter_max:
        mejora = False
        it += 1
        adentro = np.where(sel == 1)[0].tolist()
        afuera = np.where(sel == 0)[0].tolist()
        for i in adentro:
            base_riesgo = riesgos[i]
            _quitar(i)
            mejor_ganancia = 0.0
            mejor_par = None
            for j in afuera:
                if not factible[j] or sel[j] == 1:
                    continue
                if _cabe(j):
                    g = riesgos[j] - base_riesgo
                    if g > mejor_ganancia + 1e-9:
                        # Intentar agregar tambien un k extra
                        _agregar(j)
                        for k in afuera:
                            if k == j or not factible[k]:
                                continue
                            if _cabe(k):
                                g2 = (riesgos[j] + riesgos[k]) - base_riesgo
                                if g2 > mejor_ganancia + 1e-9:
                                    mejor_ganancia = g2
                                    mejor_par = (j, k)
                                break
                        if mejor_par is None or mejor_par[0] != j:
                            mejor_ganancia = g
                            mejor_par = (j, None)
                        _quitar(j)
            if mejor_par is not None:
                _agregar(mejor_par[0])
                if mejor_par[1] is not None:
                    _agregar(mejor_par[1])
                mejora = True
            else:
                _agregar(i)

    return _empacar_resultado(df, sel, "greedy_local")


def _empacar_resultado(df: pd.DataFrame, sel: np.ndarray, metodo: str) -> ResultadoPlan:
    sel = sel.astype(int)
    horas_usadas = float((df["duracion_h"].values * sel).sum())
    horas_hab: Dict[str, float] = {}
    for h in df["habilidad_requerida"].unique():
        m = (df["habilidad_requerida"].values == h) & (sel == 1)
        horas_hab[h] = float(df["duracion_h"].values[m].sum())
    return ResultadoPlan(
        seleccion=sel,
        riesgo_total=float((df["riesgo"].values * sel).sum()),
        horas_usadas=horas_usadas,
        horas_por_habilidad=horas_hab,
        metodo=metodo,
        n_seleccionadas=int(sel.sum()),
    )


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def resolver_plan_semanal(
    df: pd.DataFrame,
    horas_total: float,
    horas_habilidad: Dict[str, float],
    on_hand: Optional[Dict[int, int]] = None,
    forzar_solver: Optional[str] = None,
) -> ResultadoPlan:
    """Resuelve la seleccion del plan semanal.

    Args:
        df: DataFrame con columnas riesgo, duracion_h, habilidad_requerida,
            repuesto_id, repuesto_disponible.
        horas_total: capacidad total de horas-hombre.
        horas_habilidad: dict habilidad -> horas disponibles.
        on_hand: dict repuesto_id -> unidades disponibles. Si es None se
            asume 1 unidad por cada repuesto referenciado en df.
        forzar_solver: 'pulp' o 'greedy' para forzar el camino.

    Returns:
        ResultadoPlan
    """
    df = df.reset_index(drop=True)
    if on_hand is None:
        # Cada repuesto que aparece tiene 1 unidad disponible.
        ids = sorted(set(df["repuesto_id"].tolist()) - {-1})
        on_hand = {r: 1 for r in ids}

    if forzar_solver == "greedy":
        return _solver_greedy(df, horas_total, horas_habilidad, on_hand)
    if forzar_solver == "pulp":
        if not _PULP_OK:
            raise RuntimeError("PuLP no esta instalado")
        return _solver_pulp(df, horas_total, horas_habilidad, on_hand)

    if _PULP_OK:
        return _solver_pulp(df, horas_total, horas_habilidad, on_hand)
    return _solver_greedy(df, horas_total, horas_habilidad, on_hand)


if __name__ == "__main__":
    from data_generator import ConfigEscenario, generar_backlog_semanal
    from mcda import calcular_riesgo

    cfg = ConfigEscenario()
    df = generar_backlog_semanal(1, cfg, seed=42)
    df_score = calcular_riesgo(df, df["prob_real_falla_7d"].values)
    res = resolver_plan_semanal(df_score, cfg.horas_totales,
                                 cfg.horas_por_habilidad)
    print("Resultado:", res.como_dict())
    print(f"Riesgo total backlog: {df_score['riesgo'].sum():.2f}")
    print(f"Riesgo programado:   {res.riesgo_total:.2f}")
    print(f"Riesgo residual:     {df_score['riesgo'].sum() - res.riesgo_total:.2f}")
