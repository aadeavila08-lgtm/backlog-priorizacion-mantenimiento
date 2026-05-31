"""
Modelo predictivo-prescriptivo para priorizacion del backlog de mantenimiento.

Modulos:
    data_generator : generacion de datos sinteticos reproducibles
    model          : entrenamiento y calibracion del clasificador
    mcda           : scoring multicriterio
    optimizer      : seleccion semanal con PuLP (MILP)
    baseline       : reglas tradicionales por prioridad/fecha
    evaluation     : KPIs operativos
    scenarios      : definicion de los 5 escenarios experimentales
"""

__version__ = "1.0.0"
__author__ = "Anibal Antonio De Avila Rueda"
