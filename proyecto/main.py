"""
Orquestador principal: ejecuta los 5 escenarios con 30 replicas cada uno
y produce los CSV/Excel de resultados.

Uso:
    python main.py [--replicas N] [--semanas-train K]

Salidas en proyecto/results/:
    - resultados_detalle.csv : KPIs por escenario x replica x metodo
    - resultados_resumen.csv : promedios y desviaciones por escenario
    - metricas_modelo.csv    : AUC/PR-AUC/Brier por replica
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import ConfigEscenario, generar_backlog_semanal
from model import entrenar_modelo
from mcda import calcular_riesgo
from optimizer import resolver_plan_semanal
from baseline import baseline_prioridad_fecha
from evaluation import calcular_kpis, comparar
from scenarios import escenarios


def _on_hand_dict(cfg: ConfigEscenario) -> Dict[int, int]:
    """Replica la asignacion on-hand del data_generator (consistente con el dataset)."""
    rng = np.random.default_rng(cfg.repuesto_forzado_faltante or 0)
    # NOTA: Esta funcion no se usa para construir on-hand del optimizador,
    # ya que cada WO trae su flag repuesto_disponible. Aqui devolvemos el
    # numero de "unidades" disponibles por id (1 si hay, 0 si no).
    on_hand = {}
    for r in range(cfg.n_repuestos_criticos):
        on_hand[r] = 1
    if cfg.repuesto_forzado_faltante is not None:
        on_hand[cfg.repuesto_forzado_faltante % cfg.n_repuestos_criticos] = 0
    return on_hand


def correr(replicas: int = 30, semanas_train: int = 25) -> None:
    print("=" * 78)
    print("PROYECTO: Modelo predictivo-prescriptivo de priorizacion de WO")
    print(f"Replicas por escenario: {replicas}  |  Semanas para train: {semanas_train}")
    print("=" * 78)

    detalle: List[Dict] = []
    metricas_lista: List[Dict] = []

    escs = escenarios()
    t0 = time.time()

    for esc_id, esc in escs.items():
        print(f"\n--- Escenario {esc_id}: {esc.nombre} ---")
        cfg = esc.config

        for replica in range(replicas):
            seed = 1000 * (1 + ord(esc_id[1]) - ord("1")) + replica

            # 1. Dataset de entrenamiento (semanas anteriores, mismo escenario)
            semanas_tr = [generar_backlog_semanal(s, cfg, seed=seed)
                          for s in range(1, semanas_train + 1)]
            df_train = pd.concat(semanas_tr, ignore_index=True)

            # 2. Entrena y calibra el modelo
            modelo, met = entrenar_modelo(df_train, random_state=seed)
            metricas_lista.append({
                "escenario": esc_id, "replica": replica,
                **met.como_dict(),
            })

            # 3. Semana de prueba (semana K+1)
            df_test = generar_backlog_semanal(semanas_train + 1, cfg,
                                               seed=seed + 99)
            proba = modelo.predict_proba(df_test)
            df_score = calcular_riesgo(df_test, proba, pesos=esc.pesos)

            # 4. Capacidades efectivas
            h_total = cfg.horas_efectivas_total()
            h_hab   = cfg.horas_efectivas_por_habilidad()
            on_hand = _on_hand_dict(cfg)

            # 5. Plan baseline vs modelo
            plan_b = baseline_prioridad_fecha(df_score, h_total, h_hab, on_hand)
            plan_m = resolver_plan_semanal(df_score, h_total, h_hab, on_hand)

            # 6. KPIs
            kpi_b = calcular_kpis(df_score, plan_b, h_total, h_hab)
            kpi_m = calcular_kpis(df_score, plan_m, h_total, h_hab)

            base = {"escenario": esc_id, "replica": replica}
            for k, v in kpi_b.como_dict().items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        detalle.append({**base, "metodo": "baseline",
                                        "kpi": f"{k}.{kk}", "valor": vv})
                else:
                    detalle.append({**base, "metodo": "baseline",
                                    "kpi": k, "valor": v})
            for k, v in kpi_m.como_dict().items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        detalle.append({**base, "metodo": "modelo",
                                        "kpi": f"{k}.{kk}", "valor": vv})
                else:
                    detalle.append({**base, "metodo": "modelo",
                                    "kpi": k, "valor": v})

            # Guarda comparacion
            for k, v in comparar(kpi_b, kpi_m).items():
                detalle.append({**base, "metodo": "comparacion",
                                "kpi": k, "valor": v})

        print(f"  {replicas} replicas completadas")

    # --- Persistir resultados ---
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    df_det = pd.DataFrame(detalle)
    df_met = pd.DataFrame(metricas_lista)

    df_det.to_csv(os.path.join(out_dir, "resultados_detalle.csv"), index=False)
    df_met.to_csv(os.path.join(out_dir, "metricas_modelo.csv"), index=False)

    # Resumen pivotado
    pivot = (df_det[df_det["metodo"].isin(["baseline", "modelo"])]
             .pivot_table(index=["escenario", "kpi"],
                          columns="metodo",
                          values="valor",
                          aggfunc=["mean", "std"])
             .reset_index())
    pivot.columns = ['_'.join([str(x) for x in c if x != ""]).strip('_')
                     for c in pivot.columns]
    pivot.to_csv(os.path.join(out_dir, "resultados_resumen.csv"), index=False)

    print(f"\nTiempo total: {time.time() - t0:.1f} s")
    print(f"Resultados en {out_dir}")
    print("\n--- Resumen rapido (riesgo_residual_abs) ---")
    rr = (df_det[(df_det["kpi"] == "riesgo_residual_abs") &
                 (df_det["metodo"].isin(["baseline", "modelo"]))]
          .groupby(["escenario", "metodo"])["valor"].mean().unstack())
    rr["reduccion_%"] = 100 * (rr["baseline"] - rr["modelo"]) / rr["baseline"]
    print(rr.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=30)
    parser.add_argument("--semanas-train", type=int, default=25)
    args = parser.parse_args()
    correr(replicas=args.replicas, semanas_train=args.semanas_train)
