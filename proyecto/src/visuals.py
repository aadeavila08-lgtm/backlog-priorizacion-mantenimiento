"""
Generacion de graficas comparativas para el informe.

Lee resultados/resultados_detalle.csv y produce:
    - boxplot_riesgo_residual.png
    - boxplot_pct_criticas.png
    - boxplot_utilizacion.png
    - barras_reduccion_riesgo.png
    - calibracion_modelo.png
    - importancia_variables.png
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETA = {"baseline": "#C0504D", "modelo": "#4F81BD"}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _boxplot_kpi(df: pd.DataFrame, kpi: str, titulo: str,
                  ylabel: str, salida: str) -> None:
    sub = df[(df["kpi"] == kpi) & (df["metodo"].isin(["baseline", "modelo"]))]
    if sub.empty:
        print(f"  (sin datos para {kpi})")
        return
    escs = sorted(sub["escenario"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    pos = np.arange(len(escs))
    width = 0.35
    for off, met in zip([-width / 2, width / 2], ["baseline", "modelo"]):
        datos = [sub[(sub["escenario"] == e) &
                     (sub["metodo"] == met)]["valor"].values
                 for e in escs]
        bp = ax.boxplot(datos, positions=pos + off, widths=width * 0.9,
                        patch_artist=True, manage_ticks=False,
                        medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set_facecolor(PALETA[met])
            box.set_alpha(0.85)
    ax.set_xticks(pos)
    ax.set_xticklabels(escs)
    ax.set_xlabel("Escenario")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETA[m]) for m in ["baseline", "modelo"]]
    ax.legend(handles, ["Baseline", "Modelo"], loc="best")
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def _barras_reduccion(df: pd.DataFrame, salida: str) -> None:
    rr = (df[(df["kpi"] == "riesgo_residual_abs") &
              (df["metodo"].isin(["baseline", "modelo"]))]
          .groupby(["escenario", "metodo"])["valor"].mean().unstack())
    rr["reduccion_%"] = 100 * (rr["baseline"] - rr["modelo"]) / rr["baseline"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colores = ["#4F81BD" if v >= 0 else "#C0504D" for v in rr["reduccion_%"]]
    ax.bar(rr.index, rr["reduccion_%"], color=colores, alpha=0.85)
    for i, v in enumerate(rr["reduccion_%"]):
        ax.text(i, v + (0.5 if v >= 0 else -1.5),
                f"{v:.1f}%", ha="center",
                fontweight="bold", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhspan(25, 40, color="#9BBB59", alpha=0.18, label="Objetivo 25-40%")
    ax.set_ylabel("Reduccion de riesgo residual (%)")
    ax.set_xlabel("Escenario")
    ax.set_title("Reduccion del riesgo residual: Modelo vs Baseline")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def _curva_calibracion(df_metricas: pd.DataFrame, salida: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pos = np.arange(len(df_metricas["escenario"].unique()))
    escs = sorted(df_metricas["escenario"].unique())
    auc_med = [df_metricas[df_metricas["escenario"] == e]["AUC"].mean() for e in escs]
    pra_med = [df_metricas[df_metricas["escenario"] == e]["PR_AUC"].mean() for e in escs]
    bri_med = [df_metricas[df_metricas["escenario"] == e]["Brier"].mean() for e in escs]
    width = 0.27
    ax.bar(pos - width, auc_med, width, label="AUC",      color="#4F81BD", alpha=0.9)
    ax.bar(pos,         pra_med, width, label="PR-AUC",   color="#9BBB59", alpha=0.9)
    ax.bar(pos + width, bri_med, width, label="Brier",    color="#C0504D", alpha=0.9)
    ax.set_xticks(pos)
    ax.set_xticklabels(escs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor (0-1)")
    ax.set_title("Desempeno del modelo predictivo por escenario")
    ax.axhline(0.75, ls="--", color="grey", alpha=0.7,
               label="Umbral AUC objetivo (0.75)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def _importancia(modelo, salida: str) -> None:
    from model import importancia_variables
    imp = importancia_variables(modelo).head(12)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colores = ["#4F81BD" if c >= 0 else "#C0504D" for c in imp["coeficiente"]]
    ax.barh(imp["variable"][::-1], imp["coeficiente"][::-1],
            color=colores[::-1], alpha=0.9)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Coeficiente (variables estandarizadas)")
    ax.set_title("Top variables explicativas del modelo (Escenario E1)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def generar_todas(carpeta_resultados: str) -> Dict[str, str]:
    _setup_style()
    df = pd.read_csv(os.path.join(carpeta_resultados, "resultados_detalle.csv"))
    df_met = pd.read_csv(os.path.join(carpeta_resultados, "metricas_modelo.csv"))
    g = os.path.join(carpeta_resultados, "graficas")
    os.makedirs(g, exist_ok=True)

    archivos = {}
    archivos["riesgo_residual"] = os.path.join(g, "boxplot_riesgo_residual.png")
    _boxplot_kpi(df, "riesgo_residual_abs",
                 "Distribucion del riesgo residual por escenario",
                 "Riesgo residual (suma)", archivos["riesgo_residual"])

    archivos["pct_criticas"] = os.path.join(g, "boxplot_pct_criticas.png")
    _boxplot_kpi(df, "wo_criticas_programadas_pct",
                 "% de WO criticas programadas por escenario",
                 "% WO criticas programadas", archivos["pct_criticas"])

    archivos["utilizacion"] = os.path.join(g, "boxplot_utilizacion.png")
    _boxplot_kpi(df, "utilizacion_total_pct",
                 "Utilizacion total de horas por escenario",
                 "% utilizacion", archivos["utilizacion"])

    archivos["reduccion"] = os.path.join(g, "barras_reduccion_riesgo.png")
    _barras_reduccion(df, archivos["reduccion"])

    archivos["calibracion"] = os.path.join(g, "calibracion_modelo.png")
    _curva_calibracion(df_met, archivos["calibracion"])

    # Importancia: re-entrenar un modelo en E1 para visualizar
    from data_generator import ConfigEscenario, generar_backlog_semanal
    from model import entrenar_modelo
    cfg = ConfigEscenario()
    semanas = [generar_backlog_semanal(s, cfg, seed=7) for s in range(1, 21)]
    df_train = pd.concat(semanas, ignore_index=True)
    modelo, _ = entrenar_modelo(df_train)
    archivos["importancia"] = os.path.join(g, "importancia_variables.png")
    _importancia(modelo, archivos["importancia"])
    return archivos


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    out = os.path.abspath(out)
    archivos = generar_todas(out)
    for k, v in archivos.items():
        print(f"  {k:18s} -> {v}")
