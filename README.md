# Modelo predictivo–prescriptivo para priorizar el backlog del mantenimiento

**MCDA y optimización ligera con criterios de criticidad**
Trabajo de Grado · Especialización Tecnológica en Automatización Industrial
Universidad Manuela Beltrán · 2026

**Autor:** Aníbal Antonio De Avila Rueda
**Director:** Mg. Fabián Castro

---

## Resumen

Este proyecto desarrolla y evalúa un modelo predictivo–prescriptivo para la priorización del backlog semanal de mantenimiento bajo recursos limitados. Integra tres componentes en un pipeline reproducible y auditable:

1. **Predicción calibrada** — Regresión logística con calibración isotónica (Pool Adjacent Violators) que estima la probabilidad de falla en 7 días por orden de trabajo.
2. **Análisis multicriterio (MCDA)** — Combina probabilidad, severidad y costo en un puntaje de riesgo explicable (Riesgo = 0,5·P + 0,3·S + 0,2·C).
3. **Optimización ligera** — Selección semanal tipo knapsack multi-restricción (PuLP/CBC + fallback greedy) que respeta capacidad total, capacidad por habilidad y disponibilidad de repuestos.

El estudio se ejecutó sobre **5 escenarios × 30 réplicas = 150 corridas** con datos sintéticos reproducibles.

## Resultados destacados

| KPI | Valor |
|---|---|
| Reducción media del riesgo residual vs. baseline | **12 %** (rango 9–13 %) |
| AUC del componente predictivo | **0,762** (DE < 0,03) |
| Brier Score | **0,201** |
| Cumplimiento de restricciones operativas | **100 %** |
| Violaciones de repuestos | **0** en 150 corridas |
| Significancia estadística (t pareada) | **p < 0,001** en los 5 escenarios |

## Estructura del proyecto

```
proyecto/
├── src/
│   ├── data_generator.py      # Generador de backlogs sintéticos reproducibles (OE1)
│   ├── model.py               # Regresión logística + calibración isotónica (OE2)
│   ├── mcda.py                # Puntaje multicriterio de riesgo
│   ├── optimizer.py           # MILP PuLP/CBC + fallback greedy (OE3)
│   ├── baseline.py            # Política tradicional para comparación
│   ├── evaluation.py          # Cálculo de KPIs operativos
│   ├── scenarios.py           # Definición de los 5 escenarios
│   ├── visuals.py             # Generación de gráficas matplotlib
│   ├── exportar_excel.py      # Tabla resumen Excel
│   ├── exportar_dashboard.py  # Dashboard HTML interactivo
│   ├── exportar_diccionario.py # Diccionario de datos en Word
│   └── exportar_informe.py    # Informe Word con figuras
├── notebooks/
│   └── 01_pipeline_completo.ipynb
├── results/                   # CSV, Excel, dashboard, gráficas
├── docs/                      # Diccionario de datos
├── main.py                    # Orquestador (5 esc × 30 réplicas)
├── requirements.txt
└── README.md
```

## Requisitos

- Python ≥ 3.10
- numpy, pandas, matplotlib, scikit-learn (opcional), PuLP (opcional), openpyxl, python-docx

```bash
pip install -r requirements.txt
```

> **Nota:** El proyecto fue diseñado para funcionar con **numpy/pandas puros**. Si scikit-learn o PuLP no están instalados, los módulos hacen *fallback* automático a implementaciones internas equivalentes.

## Cómo reproducir los resultados

### Reproducción completa (≈ 10 segundos)

```bash
# 1. Ejecutar las 150 corridas (5 escenarios × 30 réplicas)
python main.py --replicas 30 --semanas-train 25

# 2. Generar gráficas y tablas
python src/visuals.py
python src/exportar_excel.py

# 3. Generar dashboard interactivo
python src/exportar_dashboard.py

# 4. Generar diccionario de datos y exportar informe
python src/exportar_diccionario.py
python src/exportar_informe.py
```

### Verificación rápida de cada componente

```bash
# OE1 — Generador de datos
python -c "
from src.data_generator import generar_backlog_semanal, ConfigEscenario
df = generar_backlog_semanal(semana=1, config=ConfigEscenario(), seed=42)
print(df.head())
"

# OE2 — Modelo predictivo calibrado
python -c "
import pandas as pd
from src.data_generator import ConfigEscenario, generar_backlog_semanal
from src.model import entrenar_modelo
cfg = ConfigEscenario()
df = pd.concat([generar_backlog_semanal(s, cfg, seed=7) for s in range(1,21)])
modelo, met = entrenar_modelo(df)
print(met.como_dict())
"

# OE3+OE4 — Optimización vs Baseline
python -c "
from src.data_generator import ConfigEscenario, generar_backlog_semanal
from src.model import entrenar_modelo
from src.mcda import calcular_riesgo
from src.optimizer import resolver_plan_semanal
from src.baseline import baseline_prioridad_fecha
from src.evaluation import calcular_kpis, comparar
import pandas as pd
cfg = ConfigEscenario()
df_train = pd.concat([generar_backlog_semanal(s, cfg, seed=7) for s in range(1,21)])
modelo, _ = entrenar_modelo(df_train)
df_test = generar_backlog_semanal(21, cfg, seed=99)
df_score = calcular_riesgo(df_test, modelo.predict_proba(df_test))
plan_m = resolver_plan_semanal(df_score, cfg.horas_totales, cfg.horas_por_habilidad)
plan_b = baseline_prioridad_fecha(df_score, cfg.horas_totales, cfg.horas_por_habilidad)
kpi_m = calcular_kpis(df_score, plan_m, cfg.horas_totales, cfg.horas_por_habilidad)
kpi_b = calcular_kpis(df_score, plan_b, cfg.horas_totales, cfg.horas_por_habilidad)
print('Mejora:', comparar(kpi_b, kpi_m))
"
```

## Notebook orquestador

Para una vista paso a paso del pipeline completo, abre:

```bash
jupyter notebook notebooks/01_pipeline_completo.ipynb
```

## Salidas generadas

Tras ejecutar `main.py` + scripts complementarios:

| Archivo | Descripción |
|---|---|
| `results/resultados_detalle.csv` | KPIs por escenario × réplica × método |
| `results/resultados_resumen.csv` | Promedios y desviaciones pivotados |
| `results/metricas_modelo.csv` | AUC/PR-AUC/Brier por réplica |
| `results/tabla_resumen.xlsx` | Excel formateado con 5 hojas |
| `results/dashboard_resultados.html` | Dashboard interactivo (Chart.js) |
| `results/graficas/*.png` | 6 gráficas comparativas |
| `docs/diccionario_datos.docx` | Diccionario de datos formal |
| `results/informe_final_resultados.docx` | Informe Word integrado |

## Citas y trabajos relacionados

Las referencias formales del trabajo (27 entradas en formato IEEE) están en el informe técnico final del proyecto.

Referencias seleccionadas:

- Kull, M., Silva Filho, T., & Flach, P. (2017). Beta calibration. *AISTATS*.
- Pinedo, M. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.
- Greco, S., Ehrgott, M., & Figueira, J. R. (Eds.). (2016). *Multiple Criteria Decision Analysis: State of the Art Surveys* (2nd ed.). Springer.
- Lei, Y. et al. (2020). Machine learning for fault diagnosis and prognosis. *Mechanical Systems and Signal Processing*, 138.
- ISO 14224 (2016) · ISO 31000 (2018) · ISO 55002 (2018) · IEC 60300-3-11 (2017).

## Licencia

Este proyecto se distribuye con fines académicos y de investigación. El código fuente es de acceso abierto bajo licencia MIT. Los datos son completamente sintéticos y no contienen información confidencial.

## Contacto

**Aníbal Antonio De Avila Rueda**
Especialización en Automatización Industrial — UMB
*Correo institucional disponible en el informe técnico.*
