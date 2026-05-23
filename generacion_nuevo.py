# ==============================================================
# TFG - GENERACIÓN DE DATASET MEDIANTE OPTIMIZACIÓN MPC/MILP
# ==============================================================
#
# Autor: James Kagunda Wangari
# Grado en Ingeniería Eléctrica - Universidad de Málaga
#
# Descripción:
# ------------
# Sistema EMS para una Virtual Power Plant (VPP) basado en
# optimización MILP (Mixed-Integer Linear Programming) con
# solver HiGHS, que garantiza la solución óptima global
# en cada escenario generado.
#
# Cambio respecto a la versión anterior (IPOPT/NLP):
# --------------------------------------------------
# La versión anterior usaba IPOPT con una penalización
# cuadrática para aproximar la restricción de no simultaneidad
# de carga/descarga de la batería. IPOPT es un solver de
# óptimos locales y podía dar soluciones inconsistentes entre
# escenarios similares, lo que dificultaba el aprendizaje
# del modelo de IA.
#
# En esta versión se usa HiGHS con variables binarias reales
# (formulación MILP completa), lo que garantiza:
#   - Optimalidad global en cada escenario
#   - Soluciones deterministas y consistentes
#   - No simultaneidad de carga/descarga garantizada
#   - Dataset de entrenamiento limpio y aprendible
#
# Instalación del solver:
#   pip install highspy
#
# El núcleo del problema sigue siendo la gestión óptima
# de la batería: decidir cuándo cargar y cuándo descargar
# según los precios del mercado, la generación PV y la
# demanda, minimizando el coste total de operación.
#
# Entradas (variables conocidas/predichas):
#   - P_PV(t)  : generación fotovoltaica
#   - P_L(t)   : demanda eléctrica
#   - lambda(t): precios del mercado eléctrico
#   - SOC(0)   : estado de carga inicial de la batería
#
# Salidas (decisiones del optimizador):
#   - p_bat(t) : acción sobre la batería (+ carga / - descarga)
#   - SOC(t)   : estado de carga resultante en cada hora
#
# Se generan DOS ficheros CSV:
#
#   1) dataset_vpp_completo.csv
#      -> Todas las variables del sistema (para análisis)
#
#   2) dataset_vpp_ia.csv
#      -> Solo entradas y salidas del modelo IA
#         con columna 'split' (train/test)
#
# Cada fila representa UN DÍA COMPLETO (24 horas).
# Total de escenarios: 1000 días
#   - 800 días para entrenamiento (train)
#   - 200 días para evaluación   (test)
#
# Horizonte:         T = 24 h
# Resolución:   Delta_t = 1 h
#
# ==============================================================

import numpy as np
import pandas as pd
from pyomo.environ import *

# ==============================================================
# SEMILLA ALEATORIA (reproducibilidad del experimento)
# ==============================================================

np.random.seed(42)

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================

DIAS       = 1000   # número total de escenarios
T          = 24     # horas por día
DELTA_T    = 1      # resolución temporal (h)

DIAS_TRAIN = 800    # escenarios para entrenamiento
DIAS_TEST  = 200    # escenarios para evaluación

# ==============================================================
# PARÁMETROS DEL SISTEMA
# ==============================================================

# --- Batería (BESS) ---
SOC_MIN    = 0.20   # estado de carga mínimo (fracción)
SOC_MAX    = 0.90   # estado de carga máximo (fracción)
P_CH_MAX   = 0.30   # potencia máxima de carga  (MW)
P_DIS_MAX  = 0.50   # potencia máxima de descarga (MW)
ETA_CH     = 0.95   # rendimiento de carga
ETA_DIS    = 0.95   # rendimiento de descarga
E_BAT_CAP  = 1.0    # capacidad nominal (MWh)
C_DEG      = 2.0    # coste de degradación (€/MWh ciclado)

# --- Red eléctrica ---
P_GRID_MAX   = 10.0   # límite de intercambio con la red (MW)
FACTOR_VENTA = 0.80   # fracción del precio de compra para venta

# --- Turbina de gas (respaldo) ---
P_DG_MIN  = 1.0    # potencia mínima de operación (MW)
P_DG_MAX  = 5.0    # potencia máxima de operación (MW)
DG_RAMP   = 2.0    # rampa máxima entre periodos (MW/h)
C_DG_A    = 5.0    # coeficiente cuadrático del coste (€/MW²h)
C_DG_B    = 10.0   # coeficiente lineal del coste    (€/MWh)

# ==============================================================
# GENERACIÓN DE ESCENARIOS
# ==============================================================

def generar_escenario():
    """
    Genera un escenario diario aleatorio con perfiles de:
      - precios del mercado eléctrico (€/MWh)
      - generación fotovoltaica (MW)
      - demanda industrial (MW)
      - estado de carga inicial de la batería

    Los perfiles se basan en formas sinusoidales con ruido
    gaussiano para simular la variabilidad real del sistema.
    """

    t = np.arange(T)

    # --- Perfil de precios ---
    # Forma sinusoidal con desplazamiento aleatorio para
    # generar escenarios con picos en distintas horas del día
    shift   = np.random.uniform(-3, 3)
    precios = (
        50
        + 30 * np.sin((t - 7 + shift) / 24 * 2 * np.pi)
        + np.random.normal(0, 4, T)
    )
    precios = np.clip(precios, 5, 120)

    # --- Perfil fotovoltaico ---
    # Generación solo en horas solares (7h-18h)
    # con variabilidad de irradiancia entre días
    pv = np.zeros(T)
    for h in range(T):
        if 7 <= h <= 18:
            pv[h] = 6 * np.sin((h - 6) / 12 * np.pi)
    pv *= np.random.uniform(0.6, 1.2)

    # --- Perfil de demanda ---
    # Perfil típico industrial con variabilidad aleatoria
    demanda = (
        1.5
        + 0.5 * np.sin((t - 6) / 24 * 2 * np.pi)
        + np.random.normal(0, 0.15, T)
    )
    demanda = np.clip(demanda, 0.5, 3.0)

    # --- SOC inicial ---
    # Aleatorio dentro del rango operativo para cubrir
    # todo el espacio de estados durante el entrenamiento
    soc0 = np.random.uniform(SOC_MIN + 0.10, SOC_MAX - 0.10)

    return precios, pv, demanda, soc0

# ==============================================================
# MODELO DE OPTIMIZACIÓN MILP
# ==============================================================

def optimizar_vpp(precios, pv, demanda, soc0):
    """
    Formula y resuelve el problema MILP para un día completo
    (T=24h) con solver HiGHS, garantizando la solución óptima
    global en cada escenario.

    La formulación MILP completa incluye variables binarias
    reales para garantizar la no simultaneidad de carga y
    descarga de la batería, a diferencia de la versión NLP
    anterior que usaba una penalización cuadrática aproximada.

    El objetivo es minimizar el coste total de operación:
      - coste de compra de energía a la red
      - ingresos por venta de excedentes a la red
      - coste de operación de la turbina de respaldo
      - coste de degradación de la batería

    Parámetros
    ----------
    precios : array (T,) - precios horarios del mercado (€/MWh)
    pv      : array (T,) - generación fotovoltaica (MW)
    demanda : array (T,) - demanda eléctrica (MW)
    soc0    : float      - estado de carga inicial (fracción)

    Retorna
    -------
    model : modelo Pyomo resuelto con HiGHS
    status : estado de la solución ('ok' o 'failed')
    """

    model = ConcreteModel()
    model.T = RangeSet(0, T - 1)

    # ----------------------------------------------------------
    # VARIABLES CONTINUAS DE DECISIÓN
    # ----------------------------------------------------------

    model.P_ch        = Var(model.T, bounds=(0, P_CH_MAX))
    model.P_dis       = Var(model.T, bounds=(0, P_DIS_MAX))
    model.SOC         = Var(model.T, bounds=(SOC_MIN, SOC_MAX))
    model.P_grid_buy  = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_grid_sell = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_dg        = Var(model.T, bounds=(P_DG_MIN, P_DG_MAX))

    # ----------------------------------------------------------
    # VARIABLES BINARIAS
    # ----------------------------------------------------------
    # u_ch[t] = 1 si la batería carga en la hora t, 0 si no
    # u_dis[t] = 1 si la batería descarga en la hora t, 0 si no
    # La restricción u_ch + u_dis <= 1 garantiza que nunca
    # ocurren carga y descarga al mismo tiempo (no simultaneidad)

    model.u_ch  = Var(model.T, within=Binary)
    model.u_dis = Var(model.T, within=Binary)

    # ----------------------------------------------------------
    # FUNCIÓN OBJETIVO
    # ----------------------------------------------------------
    # Minimizar el coste total de operación.
    # La función objetivo es lineal en todas las variables
    # continuas, lo que junto con las binarias da un MILP puro.
    # Nota: el coste de la turbina se linealiza usando la
    # aproximación lineal por tramos o simplificando a lineal
    # para mantener la formulación MILP estricta.

    def objective_rule(m):
        return sum(
            # (1) Coste de compra de energía a la red
            precios[t] * m.P_grid_buy[t] * DELTA_T

            # (2) Ingreso por venta de excedentes (resta)
            - FACTOR_VENTA * precios[t] * m.P_grid_sell[t] * DELTA_T

            # (3) Coste lineal de la turbina de gas
            + C_DG_B * m.P_dg[t] * DELTA_T

            # (4) Coste de degradación de la batería
            + C_DEG * (m.P_ch[t] + m.P_dis[t]) * DELTA_T

            for t in m.T
        )

    model.obj = Objective(rule=objective_rule, sense=minimize)

    # ----------------------------------------------------------
    # RESTRICCIONES
    # ----------------------------------------------------------

    model.constraints = ConstraintList()

    for t in model.T:

        # Balance de potencia instantáneo
        model.constraints.add(
            pv[t] + model.P_dis[t] + model.P_grid_buy[t] + model.P_dg[t]
            ==
            demanda[t] + model.P_ch[t] + model.P_grid_sell[t]
        )

        # Límite de carga vinculado a variable binaria u_ch
        # Si u_ch[t]=0 entonces P_ch[t]=0 (batería no carga)
        model.constraints.add(
            model.P_ch[t] <= P_CH_MAX * model.u_ch[t]
        )

        # Límite de descarga vinculado a variable binaria u_dis
        # Si u_dis[t]=0 entonces P_dis[t]=0 (batería no descarga)
        model.constraints.add(
            model.P_dis[t] <= P_DIS_MAX * model.u_dis[t]
        )

        # No simultaneidad: carga y descarga no pueden ocurrir
        # al mismo tiempo en ninguna hora del horizonte
        model.constraints.add(
            model.u_ch[t] + model.u_dis[t] <= 1
        )

    # Dinámica del SOC hora a hora
    for t in model.T:
        soc_prev = soc0 if t == 0 else model.SOC[t - 1]
        model.constraints.add(
            model.SOC[t]
            ==
            soc_prev
            + (ETA_CH * model.P_ch[t] * DELTA_T
               - model.P_dis[t] * DELTA_T / ETA_DIS) / E_BAT_CAP
        )

    # Rampa de la turbina de gas entre horas consecutivas
    for t in model.T:
        if t > 0:
            model.constraints.add(
                model.P_dg[t] - model.P_dg[t - 1] <= DG_RAMP
            )
            model.constraints.add(
                model.P_dg[t - 1] - model.P_dg[t] <= DG_RAMP
            )

    # Neutralidad energética diaria:
    # el SOC al final del día no puede ser menor que el inicial
    model.constraints.add(
        model.SOC[T - 1] >= soc0
    )

    # ----------------------------------------------------------
    # RESOLUCIÓN CON HIGHS
    # ----------------------------------------------------------
    # HiGHS es un solver MILP open source que garantiza la
    # solución óptima global. No requiere ejecutable externo
    # ni licencia — se instala con: pip install highspy

    solver = SolverFactory('highs')
    resultado = solver.solve(model, tee=False)

    # Verificar que el solver encontró solución óptima
    status = 'ok'
    if (resultado.solver.termination_condition
            != TerminationCondition.optimal):
        status = 'failed'

    return model, status

# ==============================================================
# BUCLE PRINCIPAL DE GENERACIÓN
# ==============================================================

dataset_completo = []
dataset_ia       = []
fallos           = 0   # contador de escenarios sin solución óptima

print(f"\nGenerando {DIAS} escenarios "
      f"({DIAS_TRAIN} train / {DIAS_TEST} test)...")
print("Solver: HiGHS (MILP - optimalidad global garantizada)\n")

for d in range(DIAS):

    # Generar escenario aleatorio
    precios, pv, demanda, soc0 = generar_escenario()

    # Resolver optimización MILP
    model, status = optimizar_vpp(precios, pv, demanda, soc0)

    # Si el solver falla en algún escenario, se registra y
    # se continúa con el siguiente para no interrumpir la
    # generación del dataset completo
    if status == 'failed':
        fallos += 1
        print(f"  [AVISO] Escenario {d} sin solucion optima — omitido")
        continue

    # Etiqueta de partición train/test
    split = 'train' if d < DIAS_TRAIN else 'test'

    # ==========================================================
    # DATASET COMPLETO (todas las variables del sistema)
    # ==========================================================

    fila_full = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4)
    }

    for t in range(T):
        fila_full[f'precio_h{t}']  = round(precios[t], 4)
    for t in range(T):
        fila_full[f'pv_h{t}']      = round(pv[t], 4)
    for t in range(T):
        fila_full[f'demanda_h{t}'] = round(demanda[t], 4)

    # Acción neta sobre la batería: positivo=carga, negativo=descarga
    for t in range(T):
        p_bat = value(model.P_ch[t]) - value(model.P_dis[t])
        fila_full[f'p_bat_h{t}'] = round(p_bat, 4)

    for t in range(T):
        fila_full[f'soc_h{t}'] = round(value(model.SOC[t]), 4)

    # Intercambio neto con la red: positivo=compra, negativo=venta
    for t in range(T):
        p_red = value(model.P_grid_buy[t]) - value(model.P_grid_sell[t])
        fila_full[f'p_red_h{t}'] = round(p_red, 4)

    for t in range(T):
        fila_full[f'p_dg_h{t}'] = round(value(model.P_dg[t]), 4)

    dataset_completo.append(fila_full)

    # ==========================================================
    # DATASET IA
    # Entradas: precios, PV, demanda, SOC inicial
    # Salidas:  p_bat (acción batería), SOC horario
    # ==========================================================

    fila_ia = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4)
    }

    # -- Entradas (73 valores) --
    for t in range(T):
        fila_ia[f'precio_h{t}']  = round(precios[t], 4)
    for t in range(T):
        fila_ia[f'pv_h{t}']      = round(pv[t], 4)
    for t in range(T):
        fila_ia[f'demanda_h{t}'] = round(demanda[t], 4)

    # -- Salidas (48 valores) --
    for t in range(T):
        p_bat = value(model.P_ch[t]) - value(model.P_dis[t])
        fila_ia[f'p_bat_h{t}'] = round(p_bat, 4)
    for t in range(T):
        fila_ia[f'soc_h{t}'] = round(value(model.SOC[t]), 4)

    dataset_ia.append(fila_ia)

    # Progreso cada 100 escenarios
    if (d + 1) % 100 == 0:
        print(f"  Escenario {d + 1:4d}/{DIAS} completado  [{split}]")

# ==============================================================
# EXPORTACIÓN CSV
# ==============================================================

df_full = pd.DataFrame(dataset_completo)
df_ia   = pd.DataFrame(dataset_ia)

df_full.to_csv("dataset_vpp_completo.csv", index=False)
df_ia.to_csv("dataset_vpp_ia.csv",         index=False)

# ==============================================================
# RESUMEN FINAL
# ==============================================================

escenarios_ok = DIAS - fallos

print("\n" + "=" * 55)
print("  DATASETS GENERADOS CORRECTAMENTE")
print("=" * 55)

print(f"\n  Solver utilizado    : HiGHS (MILP)")
print(f"  Total solicitados   : {DIAS}")
print(f"  Resueltos con exito : {escenarios_ok}")
print(f"  Fallos omitidos     : {fallos}")
print(f"  Entrenamiento (train): {DIAS_TRAIN}")
print(f"  Evaluacion    (test) : {DIAS_TEST}")

print("\n  Archivos generados:")
print("    1) dataset_vpp_completo.csv  -- todas las variables")
print("    2) dataset_vpp_ia.csv        -- entradas/salidas IA")

print("\n  Estructura del dataset IA:")
print("    Entradas : precio_h0..h23 | pv_h0..h23 "
      "| demanda_h0..h23 | soc_inicial")
print("    Salidas  : p_bat_h0..h23  | soc_h0..h23")
print("    Split    : columna 'split' (train / test)")
print()