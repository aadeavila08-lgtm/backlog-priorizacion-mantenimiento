"""
Definicion de los 5 escenarios experimentales (informe, seccion 2b/4).

E1. Recursos nominales (linea base de comparacion).
E2. -20% de capacidad de mano de obra.
E3. Faltante de un repuesto critico (repuesto id 0 con on-hand = 0).
E4. MCDA con mayor ponderacion de seguridad.
E5. Variacion en mezcla PM/CM (mas correctivo, mas urgencia).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from data_generator import ConfigEscenario
from mcda import PesosMCDA


@dataclass
class Escenario:
    id: str
    nombre: str
    descripcion: str
    config: ConfigEscenario
    pesos:  PesosMCDA


def escenarios() -> Dict[str, Escenario]:
    return {
        "E1": Escenario(
            id="E1",
            nombre="Recursos nominales",
            descripcion="Capacidad y stock estandar; mezcla 60% PM / 40% CM.",
            config=ConfigEscenario(),
            pesos=PesosMCDA(),
        ),
        "E2": Escenario(
            id="E2",
            nombre="-20% capacidad mano de obra",
            descripcion="Reduccion del 20% en horas totales y por habilidad.",
            config=ConfigEscenario(factor_capacidad=0.80),
            pesos=PesosMCDA(),
        ),
        "E3": Escenario(
            id="E3",
            nombre="Faltante repuesto critico",
            descripcion="Repuesto id=0 forzado a stock 0 (impacta WO que lo requieren).",
            config=ConfigEscenario(repuesto_forzado_faltante=0),
            pesos=PesosMCDA(),
        ),
        "E4": Escenario(
            id="E4",
            nombre="MCDA con peso seguridad alto",
            descripcion="Severidad pesa 0.50; sub-peso seguridad 0.70.",
            config=ConfigEscenario(),
            pesos=PesosMCDA(prob=0.40, sev=0.50, costo=0.10,
                            sev_seguridad=0.70, sev_ambiental=0.15,
                            sev_produccion=0.15),
        ),
        "E5": Escenario(
            id="E5",
            nombre="Mezcla PM/CM 30/70",
            descripcion="Mas correctivo (CM): 30% PM, 70% CM.",
            config=ConfigEscenario(p_pm=0.30),
            pesos=PesosMCDA(),
        ),
    }
