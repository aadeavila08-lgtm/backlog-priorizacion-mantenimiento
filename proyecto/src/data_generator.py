"""
Generador de datos sinteticos reproducibles para el modelo de priorizacion
del backlog de mantenimiento.

Cada llamada a generar_backlog_semanal() produce un DataFrame con N WO
(20 <= N <= 100) representando una semana de trabajo en un area de
utilidades / planta de proceso (foco I&C, mecanico, electrico).

La probabilidad real de "falla en 7 dias" se construye como una funcion
no lineal de las features (criticidad, PM vencidos, uso, edad, etc.),
permitiendo entrenar y calibrar un clasificador con relacion senal-ruido
controlada.

Autor: Anibal Antonio De Avila Rueda
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Catalogos de referencia (alineados con ISO 14224 y RCM)
# ---------------------------------------------------------------------------
CLASES_EQUIPO: List[str] = [
    "instrumentacion",   # transmisores, valvulas de control, PLC
    "mecanico",          # bombas, compresores, motores
    "electrico",         # tableros, variadores, transformadores
]

HABILIDADES: List[str] = [
    "instrumentista",
    "mecanico",
    "electricista",
]

# Mapeo clase de equipo -> habilidad principal requerida
HABILIDAD_POR_CLASE: Dict[str, str] = {
    "instrumentacion": "instrumentista",
    "mecanico":        "mecanico",
    "electrico":       "electricista",
}

TIPOS_WO: List[str] = ["PM", "CM"]  # Preventivo / Correctivo


# ---------------------------------------------------------------------------
# Configuracion del escenario
# ---------------------------------------------------------------------------
@dataclass
class ConfigEscenario:
    """Parametros que controlan la generacion de un backlog semanal."""

    # Tamaño del backlog
    n_wo_min: int = 20
    n_wo_max: int = 100

    # Capacidades (horas-hombre disponibles en la semana)
    horas_totales: float = 200.0
    horas_por_habilidad: Dict[str, float] = field(default_factory=lambda: {
        "instrumentista": 80.0,
        "mecanico":       80.0,
        "electricista":   60.0,
    })

    # Mezcla PM/CM (probabilidad de PM)
    p_pm: float = 0.6

    # Inventario de repuestos criticos
    n_repuestos_criticos: int = 10
    lead_time_promedio: float = 30.0
    lead_time_std: float = 3.0

    # Probabilidad de que una WO requiera repuesto critico
    p_requiere_repuesto: float = 0.35

    # Forzar faltante de un repuesto especifico (escenario 3)
    repuesto_forzado_faltante: Optional[int] = None

    # Multiplicador global de capacidad (escenario 2: 0.8 -> -20%)
    factor_capacidad: float = 1.0

    # Ruido en la generacion del target
    nivel_ruido: float = 0.10

    def horas_efectivas_total(self) -> float:
        return self.horas_totales * self.factor_capacidad

    def horas_efectivas_por_habilidad(self) -> Dict[str, float]:
        return {h: v * self.factor_capacidad
                for h, v in self.horas_por_habilidad.items()}


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------
def generar_backlog_semanal(
    semana: int,
    config: ConfigEscenario,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera un DataFrame con un backlog semanal de WO."""
    rng = np.random.default_rng(seed + semana * 1000)

    #### CRITICIDAD, SEVERIDADES Y ATRIBUTOS NUMERICOS ###

    # ---- 1. Numero de WO de la semana
    n_wo = rng.integers(config.n_wo_min, config.n_wo_max + 1)

    # ---- 2. Atributos basicos de cada WO
    wo_id = [f"WO-S{semana:02d}-{i:04d}" for i in range(n_wo)]
    clase_eq = rng.choice(CLASES_EQUIPO, size=n_wo,
                          p=[0.45, 0.35, 0.20])  # I&C predomina
    habilidad = np.array([HABILIDAD_POR_CLASE[c] for c in clase_eq])

    tipo_wo = rng.choice(TIPOS_WO, size=n_wo,
                         p=[config.p_pm, 1 - config.p_pm])

    # ---- 3. Atributos numericos
    # Criticidad nominal (1=baja, 5=critica) — sesgada hacia valores medios
    criticidad = rng.choice([1, 2, 3, 4, 5], size=n_wo,
                            p=[0.10, 0.25, 0.30, 0.25, 0.10])

    # Severidades por dimension (escala 0-1)
    sev_seguridad  = np.clip(rng.beta(2, 5, n_wo) + 0.1 * (criticidad - 3) / 2, 0, 1)
    sev_ambiental  = np.clip(rng.beta(2, 6, n_wo) + 0.05 * (criticidad - 3) / 2, 0, 1)
    sev_produccion = np.clip(rng.beta(3, 4, n_wo) + 0.10 * (criticidad - 3) / 2, 0, 1)

    # Historial: PM vencidos (0-5), edad del equipo, uso acumulado
    pm_vencidos = rng.poisson(0.6, n_wo).clip(0, 5)
    edad_equipo_anios = np.round(rng.uniform(0.5, 25, n_wo), 1)
    uso_acumulado_pct = np.round(rng.uniform(20, 110, n_wo), 1)  # % vida util

    # Duracion estimada en horas (lognormal, media ~4h)
    duracion_h = np.round(rng.lognormal(mean=1.2, sigma=0.5, size=n_wo), 2)
    duracion_h = np.clip(duracion_h, 0.5, 24.0)

    # Ventana de ejecucion (dia limite dentro de la semana 1-7)
    ventana_dia_limite = rng.integers(1, 8, n_wo)

    # Fecha de creacion (dias antes del lunes de la semana actual)
    dias_en_backlog = rng.integers(0, 21, n_wo)  # hasta 3 semanas

    ### PRIORIDAD NOMINAL ###
    # Prioridad nominal del CMMS (heredada del proceso actual: 1-5).
    # Realidad observada: en muchos CMMS legacy la prioridad se asigna
    # con criterio rapido (urgencia percibida, presion del solicitante)
    # y queda desacoplada de la criticidad RCM real. Modelamos correlacion
    # DEBIL: ~30% de las WO tienen prioridad alineada con criticidad,
    # el resto se asigna por una distribucion sesgada hacia 3 (default).
    alineado = rng.random(n_wo) < 0.30
    prio_aleatoria = rng.choice([1, 2, 3, 4, 5], size=n_wo,
                                 p=[0.10, 0.25, 0.40, 0.20, 0.05])
    prioridad_nominal = np.where(alineado, criticidad, prio_aleatoria).astype(int)

    # ---- 4. REPUESTOS
    repuesto_id = np.where(
        rng.random(n_wo) < config.p_requiere_repuesto,
        rng.integers(0, config.n_repuestos_criticos, n_wo),
        -1,  # -1 = no requiere repuesto
    )

    # On-hand del repuesto: 70% de los criticos tienen stock, 30% no
    on_hand_arr = (rng.random(config.n_repuestos_criticos) > 0.30).astype(int)
    if config.repuesto_forzado_faltante is not None:
        idx = config.repuesto_forzado_faltante % config.n_repuestos_criticos
        on_hand_arr[idx] = 0

    repuesto_disponible = np.array([
        True if r == -1 else bool(on_hand_arr[r])
        for r in repuesto_id
    ])

    lead_time_repuesto = np.where(
        repuesto_id == -1,
        0.0,
        np.clip(
            rng.normal(config.lead_time_promedio, config.lead_time_std, n_wo),
            5.0, 90.0,
        ),
    )

    #### LOGIT ###
    # ---- 5. Construccion de la probabilidad real de falla en 7 dias
    # Combina las features de forma no lineal con ruido controlado.
    # Esto es el "ground truth" que el modelo intentara aprender.
    logit = (
        -3.0
        + 0.55 * criticidad
        + 0.85 * pm_vencidos
        + 0.020 * (uso_acumulado_pct - 60)
        + 0.040 * (edad_equipo_anios - 10)
        + 1.30 * sev_seguridad
        + 0.70 * sev_produccion
        + 0.40 * (tipo_wo == "CM").astype(float)
        + rng.normal(0, config.nivel_ruido * 3, n_wo)
    )
    prob_real = 1.0 / (1.0 + np.exp(-logit))
    falla_7d = (rng.random(n_wo) < prob_real).astype(int)

    # ---- 6. Ensamble del DataFrame
    df = pd.DataFrame({
        "wo_id": wo_id,
        "semana": semana,
        "tipo_wo": tipo_wo,
        "clase_equipo": clase_eq,
        "habilidad_requerida": habilidad,
        "criticidad": criticidad.astype(int),
        "prioridad_nominal": prioridad_nominal.astype(int),
        "sev_seguridad": np.round(sev_seguridad, 3),
        "sev_ambiental": np.round(sev_ambiental, 3),
        "sev_produccion": np.round(sev_produccion, 3),
        "pm_vencidos": pm_vencidos.astype(int),
        "edad_equipo_anios": edad_equipo_anios,
        "uso_acumulado_pct": uso_acumulado_pct,
        "duracion_h": duracion_h,
        "ventana_dia_limite": ventana_dia_limite.astype(int),
        "dias_en_backlog": dias_en_backlog.astype(int),
        "repuesto_id": repuesto_id.astype(int),
        "repuesto_disponible": repuesto_disponible,
        "lead_time_repuesto": np.round(lead_time_repuesto, 1),
        "prob_real_falla_7d": np.round(prob_real, 4),  # ground truth (no usar como feature)
        "falla_7d": falla_7d.astype(int),               # target del entrenamiento
    })
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def features_modelo() -> List[str]:
    """Lista de features que se usan para entrenar el clasificador.

    NOTA: prob_real_falla_7d esta excluida porque es el ground truth.
    """
    return [
        "tipo_wo",
        "clase_equipo",
        "criticidad",
        "prioridad_nominal",
        "sev_seguridad",
        "sev_ambiental",
        "sev_produccion",
        "pm_vencidos",
        "edad_equipo_anios",
        "uso_acumulado_pct",
        "duracion_h",
        "ventana_dia_limite",
        "dias_en_backlog",
        "lead_time_repuesto",
    ]


def diccionario_datos() -> pd.DataFrame:
    """Devuelve la metadata completa de cada campo del backlog."""
    rows = [
        ("wo_id",                "string", "-",            "WO-S##-####", "Identificador unico de la orden"),
        ("semana",               "int",    "semana",       "1-N",         "Indice de la semana simulada"),
        ("tipo_wo",              "string", "-",            "{PM, CM}",    "Tipo de orden: preventiva o correctiva"),
        ("clase_equipo",         "string", "-",            "{I&C, mec, elec}", "Clase del equipo asociado"),
        ("habilidad_requerida",  "string", "-",            "{inst, mec, elec}", "Habilidad principal para ejecutar"),
        ("criticidad",           "int",    "escala",       "1-5",         "Criticidad nominal RCM (5=critica)"),
        ("prioridad_nominal",    "int",    "escala",       "1-5",         "Prioridad asignada por el CMMS"),
        ("sev_seguridad",        "float",  "[0-1]",        "0.0-1.0",     "Severidad consecuencia seguridad"),
        ("sev_ambiental",        "float",  "[0-1]",        "0.0-1.0",     "Severidad consecuencia ambiental"),
        ("sev_produccion",       "float",  "[0-1]",        "0.0-1.0",     "Severidad consecuencia produccion"),
        ("pm_vencidos",          "int",    "PM",           "0-5",         "Cantidad de PM vencidos del equipo"),
        ("edad_equipo_anios",    "float",  "anios",        "0.5-25",      "Edad del activo"),
        ("uso_acumulado_pct",    "float",  "%",            "20-110",      "Uso acumulado (% vida util)"),
        ("duracion_h",           "float",  "horas",        "0.5-24",      "Duracion estimada de la WO"),
        ("ventana_dia_limite",   "int",    "dia",          "1-7",         "Dia limite de ejecucion en la semana"),
        ("dias_en_backlog",      "int",    "dias",         "0-21",        "Antiguedad en el backlog"),
        ("repuesto_id",          "int",    "-",            "-1 o 0..N",   "ID del repuesto critico (-1 = no requiere)"),
        ("repuesto_disponible",  "bool",   "-",            "{True,False}", "Disponibilidad on-hand del repuesto"),
        ("lead_time_repuesto",   "float",  "dias",         "0-90",        "Lead time de reposicion (0 si no aplica)"),
        ("prob_real_falla_7d",   "float",  "[0-1]",        "0.0-1.0",     "Ground truth (solo simulacion)"),
               ("falla_7d",             "int",    "-",            "{0,1}",       "Variable objetivo binaria del modelo"),
    ]
    return pd.DataFrame(rows, columns=[
        "Campo", "Tipo", "Unidad", "Rango", "Descripcion"
    ])


if __name__ == "__main__":
    cfg = ConfigEscenario()
    df = generar_backlog_semanal(semana=1, config=cfg, seed=42)
    print(f"Backlog generado: {len(df)} WO")
    print(df.head())
    print("\nTasa de falla real:", df["falla_7d"].mean())
