"""
Exporta los resultados experimentales a un archivo Excel formateado.

Hojas:
    1. Resumen           : promedios y desviaciones por escenario y KPI.
    2. Comparacion       : metrica clave (riesgo residual) baseline vs modelo.
    3. Detalle           : datos crudos por escenario x replica x metodo.
    4. Metricas_Modelo   : AUC/PR-AUC/Brier por replica.
    5. Diccionario_Datos : metadata de los campos generados.
"""
from __future__ import annotations

import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils.dataframe import dataframe_to_rows

from data_generator import diccionario_datos


AZUL  = "4F81BD"
ROJO  = "C0504D"
VERDE = "9BBB59"
GRIS  = "D9D9D9"


def _formato_header(cell) -> None:
    cell.font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    cell.fill = PatternFill("solid", start_color=AZUL)
    cell.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
    th = Side(style="thin", color="000000")
    cell.border = Border(left=th, right=th, top=th, bottom=th)


def _formato_body(cell, alinear="left") -> None:
    cell.font = Font(name="Arial", size=10)
    cell.alignment = Alignment(horizontal=alinear, vertical="center")
    th = Side(style="thin", color="BFBFBF")
    cell.border = Border(left=th, right=th, top=th, bottom=th)


def _ajustar_anchos(ws) -> None:
    for col_idx, col_cells in enumerate(ws.columns, start=1):
        valores = [str(c.value) if c.value is not None else "" for c in col_cells]
        ancho = max(len(v) for v in valores) + 2
        ws.column_dimensions[
            ws.cell(row=1, column=col_idx).column_letter
        ].width = min(ancho, 35)


def _escribir_df(ws, df: pd.DataFrame) -> None:
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True),
                                 start=1):
        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                _formato_header(cell)
            else:
                aln = "right" if isinstance(val, (int, float)) else "left"
                _formato_body(cell, aln)


def construir(carpeta_resultados: str, salida: str) -> str:
    df_det = pd.read_csv(os.path.join(carpeta_resultados, "resultados_detalle.csv"))
    df_met = pd.read_csv(os.path.join(carpeta_resultados, "metricas_modelo.csv"))

    wb = Workbook()

    # ---- Hoja 1: Resumen
    ws = wb.active
    ws.title = "Resumen"

    sub = df_det[df_det["metodo"].isin(["baseline", "modelo"])]
    pivot = (sub.groupby(["escenario", "kpi", "metodo"])["valor"]
                .agg(["mean", "std"]).round(4).reset_index())
    pivot = pivot.pivot(index=["escenario", "kpi"],
                        columns="metodo", values=["mean", "std"]).reset_index()
    pivot.columns = ["_".join([str(x) for x in c if x != ""]).strip("_")
                     for c in pivot.columns]
    _escribir_df(ws, pivot)
    _ajustar_anchos(ws)

    # ---- Hoja 2: Comparacion (riesgo residual)
    ws2 = wb.create_sheet("Comparacion")
    rr = (df_det[(df_det["kpi"] == "riesgo_residual_abs") &
                  (df_det["metodo"].isin(["baseline", "modelo"]))]
          .groupby(["escenario", "metodo"])["valor"]
          .agg(["mean", "std"]).round(3))
    rr_b = rr.xs("baseline", level="metodo").rename(
        columns={"mean": "baseline_mean", "std": "baseline_std"})
    rr_m = rr.xs("modelo", level="metodo").rename(
        columns={"mean": "modelo_mean", "std": "modelo_std"})
    cmp = rr_b.join(rr_m).reset_index()
    cmp["reduccion_pct"] = (
        100 * (cmp["baseline_mean"] - cmp["modelo_mean"]) / cmp["baseline_mean"]
    ).round(2)
    cmp["objetivo"] = "25-40%"
    _escribir_df(ws2, cmp)
    # Resaltar columna reduccion
    n = len(cmp) + 1
    col_red = cmp.columns.get_loc("reduccion_pct") + 1
    for r in range(2, n + 1):
        v = ws2.cell(row=r, column=col_red).value
        try:
            color = VERDE if (v is not None and v >= 25) else \
                    ("FFD966" if (v is not None and v >= 10) else ROJO)
            ws2.cell(row=r, column=col_red).fill = PatternFill(
                "solid", start_color=color)
            ws2.cell(row=r, column=col_red).font = Font(
                name="Arial", size=10, bold=True)
        except Exception:
            pass
    _ajustar_anchos(ws2)

    # ---- Hoja 3: Detalle
    ws3 = wb.create_sheet("Detalle")
    _escribir_df(ws3, df_det)
    _ajustar_anchos(ws3)

    # ---- Hoja 4: Metricas modelo
    ws4 = wb.create_sheet("Metricas_Modelo")
    resumen_met = (df_met.groupby("escenario")[["AUC", "PR_AUC", "Brier"]]
                   .agg(["mean", "std"]).round(4))
    resumen_met.columns = ["_".join(c) for c in resumen_met.columns]
    resumen_met = resumen_met.reset_index()
    _escribir_df(ws4, resumen_met)
    _ajustar_anchos(ws4)

    # ---- Hoja 5: Diccionario de datos
    ws5 = wb.create_sheet("Diccionario_Datos")
    _escribir_df(ws5, diccionario_datos())
    _ajustar_anchos(ws5)

    wb.save(salida)
    return salida


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    out_dir = os.path.abspath(os.path.join(base, "..", "results"))
    salida = os.path.join(out_dir, "tabla_resumen.xlsx")
    print(construir(out_dir, salida))
