# ==============================================================
# TFG - GENERACIÓN DE DATASET MEDIANTE OPTIMIZACIÓN MILP
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
# Archivos generados (en la misma carpeta dataset/):
#   1) dataset_vpp_completo.csv  -> todas las variables del sistema
#   2) dataset_vpp_ia.csv        -> entradas/salidas para la IA
#
# Cada fila = UN DÍA COMPLETO (24 horas).
# Total: 1000 escenarios  (800 train / 200 test)
#
# IMPORTANTE: este script debe estar en la carpeta dataset/
# Los CSV se guardan automáticamente en esa misma carpeta.
#
# Horizonte:       T = 24 h
# Resolución: Delta_t = 1 h
# ==============================================================

import os
import numpy as np
import pandas as pd
from pyomo.environ import *

# ==============================================================
# RUTA DE SALIDA — carpeta donde está este script (dataset/)
# ==============================================================

DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================
# SEMILLA ALEATORIA (reproducibilidad)
# ==============================================================

np.random.seed(42)

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================

DIAS       = 1000
T          = 24
DELTA_T    = 1

DIAS_TRAIN = 800
DIAS_TEST  = 200

# ==============================================================
# PARÁMETROS DEL SISTEMA
# ==============================================================

# --- Batería (BESS) ---
SOC_MIN    = 0.20
SOC_MAX    = 0.90
P_CH_MAX   = 0.30
P_DIS_MAX  = 0.50
ETA_CH     = 0.95
ETA_DIS    = 0.95
E_BAT_CAP  = 1.0
C_DEG      = 2.0

# --- Red eléctrica ---
P_GRID_MAX   = 10.0
FACTOR_VENTA = 0.80

# --- Turbina de gas (respaldo) ---
P_DG_MIN  = 1.0
P_DG_MAX  = 5.0
DG_RAMP   = 2.0
C_DG_A    = 5.0
C_DG_B    = 10.0

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
    """
    t = np.arange(T)

    # Perfil de precios
    shift   = np.random.uniform(-3, 3)
    precios = (
        50
        + 30 * np.sin((t - 7 + shift) / 24 * 2 * np.pi)
        + np.random.normal(0, 4, T)
    )
    precios = np.clip(precios, 5, 120)

    # Perfil fotovoltaico (solo horas solares 7h-18h)
    pv = np.zeros(T)
    for h in range(T):
        if 7 <= h <= 18:
            pv[h] = 6 * np.sin((h - 6) / 12 * np.pi)
    pv *= np.random.uniform(0.6, 1.2)

    # Perfil de demanda industrial
    demanda = (
        1.5
        + 0.5 * np.sin((t - 6) / 24 * 2 * np.pi)
        + np.random.normal(0, 0.15, T)
    )
    demanda = np.clip(demanda, 0.5, 3.0)

    # SOC inicial aleatorio dentro del rango operativo
    soc0 = np.random.uniform(SOC_MIN + 0.10, SOC_MAX - 0.10)

    return precios, pv, demanda, soc0

# ==============================================================
# MODELO DE OPTIMIZACIÓN MILP
# ==============================================================

def optimizar_vpp(precios, pv, demanda, soc0):
    """
    Formula y resuelve el problema MILP para un día completo
    (T=24h) con solver HiGHS.
    """
    model   = ConcreteModel()
    model.T = RangeSet(0, T - 1)

    # Variables continuas
    model.P_ch        = Var(model.T, bounds=(0, P_CH_MAX))
    model.P_dis       = Var(model.T, bounds=(0, P_DIS_MAX))
    model.SOC         = Var(model.T, bounds=(SOC_MIN, SOC_MAX))
    model.P_grid_buy  = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_grid_sell = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_dg        = Var(model.T, bounds=(P_DG_MIN, P_DG_MAX))

    # Variables binarias (no simultaneidad carga/descarga)
    model.u_ch  = Var(model.T, within=Binary)
    model.u_dis = Var(model.T, within=Binary)

    # Función objetivo
    def objective_rule(m):
        return sum(
            precios[t] * m.P_grid_buy[t]  * DELTA_T
            - FACTOR_VENTA * precios[t] * m.P_grid_sell[t] * DELTA_T
            + C_DG_B * m.P_dg[t] * DELTA_T
            + C_DEG  * (m.P_ch[t] + m.P_dis[t]) * DELTA_T
            for t in m.T
        )
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # Restricciones
    model.constraints = ConstraintList()
    for t in model.T:
        # Balance de potencia
        model.constraints.add(
            pv[t] + model.P_dis[t] + model.P_grid_buy[t] + model.P_dg[t]
            == demanda[t] + model.P_ch[t] + model.P_grid_sell[t]
        )
        # Límites vinculados a binarias
        model.constraints.add(model.P_ch[t]  <= P_CH_MAX  * model.u_ch[t])
        model.constraints.add(model.P_dis[t] <= P_DIS_MAX * model.u_dis[t])
        # No simultaneidad
        model.constraints.add(model.u_ch[t] + model.u_dis[t] <= 1)

    # Dinámica del SOC
    for t in model.T:
        soc_prev = soc0 if t == 0 else model.SOC[t - 1]
        model.constraints.add(
            model.SOC[t] == soc_prev
            + (ETA_CH * model.P_ch[t] * DELTA_T
               - model.P_dis[t] * DELTA_T / ETA_DIS) / E_BAT_CAP
        )

    # Rampas de la turbina
    for t in model.T:
        if t > 0:
            model.constraints.add(model.P_dg[t] - model.P_dg[t-1] <=  DG_RAMP)
            model.constraints.add(model.P_dg[t-1] - model.P_dg[t] <=  DG_RAMP)

    # Neutralidad energética diaria
    model.constraints.add(model.SOC[T - 1] >= soc0)

    # Resolver con HiGHS
    solver    = SolverFactory('highs')
    resultado = solver.solve(model, tee=False)

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
fallos           = 0

print(f"\nGenerando {DIAS} escenarios "
      f"({DIAS_TRAIN} train / {DIAS_TEST} test)...")
print("Solver: HiGHS (MILP - optimalidad global garantizada)\n")

for d in range(DIAS):

    precios, pv, demanda, soc0 = generar_escenario()
    model, status = optimizar_vpp(precios, pv, demanda, soc0)

    if status == 'failed':
        fallos += 1
        print(f"  [AVISO] Escenario {d} sin solucion optima — omitido")
        continue

    split = 'train' if d < DIAS_TRAIN else 'test'

    # ----------------------------------------------------------
    # Dataset completo (todas las variables)
    # ----------------------------------------------------------
    fila_full = {'dia_id': d, 'split': split, 'soc_inicial': round(soc0, 4)}
    for t in range(T):
        fila_full[f'precio_h{t}']  = round(precios[t], 4)
    for t in range(T):
        fila_full[f'pv_h{t}']      = round(pv[t], 4)
    for t in range(T):
        fila_full[f'demanda_h{t}'] = round(demanda[t], 4)
    for t in range(T):
        fila_full[f'p_bat_h{t}']   = round(value(model.P_ch[t]) - value(model.P_dis[t]), 4)
    for t in range(T):
        fila_full[f'soc_h{t}']     = round(value(model.SOC[t]), 4)
    for t in range(T):
        fila_full[f'p_red_h{t}']   = round(value(model.P_grid_buy[t]) - value(model.P_grid_sell[t]), 4)
    for t in range(T):
        fila_full[f'p_dg_h{t}']    = round(value(model.P_dg[t]), 4)
    dataset_completo.append(fila_full)

    # ----------------------------------------------------------
    # Dataset IA (entradas + salidas para el modelo)
    # ----------------------------------------------------------
    fila_ia = {'dia_id': d, 'split': split, 'soc_inicial': round(soc0, 4)}
    for t in range(T):
        fila_ia[f'precio_h{t}']  = round(precios[t], 4)
    for t in range(T):
        fila_ia[f'pv_h{t}']      = round(pv[t], 4)
    for t in range(T):
        fila_ia[f'demanda_h{t}'] = round(demanda[t], 4)
    for t in range(T):
        fila_ia[f'p_bat_h{t}']   = round(value(model.P_ch[t]) - value(model.P_dis[t]), 4)
    for t in range(T):
        fila_ia[f'soc_h{t}']     = round(value(model.SOC[t]), 4)
    dataset_ia.append(fila_ia)

    if (d + 1) % 100 == 0:
        print(f"  Escenario {d + 1:4d}/{DIAS} completado  [{split}]")

# ==============================================================
# EXPORTACIÓN CSV — se guardan en la carpeta dataset/
# ==============================================================

ruta_completo = os.path.join(DIR, "dataset_vpp_completo.csv")
ruta_ia       = os.path.join(DIR, "dataset_vpp_ia.csv")

df_full = pd.DataFrame(dataset_completo)
df_ia   = pd.DataFrame(dataset_ia)

df_full.to_csv(ruta_completo, index=False)
df_ia.to_csv(ruta_ia,         index=False)

# ==============================================================
# RESUMEN FINAL
# ==============================================================

escenarios_ok = DIAS - fallos

print("\n" + "=" * 55)
print("  DATASETS GENERADOS CORRECTAMENTE")
print("=" * 55)
print(f"\n  Solver              : HiGHS (MILP)")
print(f"  Total solicitados   : {DIAS}")
print(f"  Resueltos con exito : {escenarios_ok}")
print(f"  Fallos omitidos     : {fallos}")
print(f"  Train               : {DIAS_TRAIN} escenarios")
print(f"  Test                : {DIAS_TEST}  escenarios")
print(f"\n  Archivos guardados en: {DIR}")
print(f"    >> dataset_vpp_completo.csv  ({escenarios_ok} filas, todas las variables)")
print(f"    >> dataset_vpp_ia.csv        ({escenarios_ok} filas, entradas/salidas IA)")
print(f"\n  Estructura dataset_vpp_ia.csv:")
print(f"    Entradas : soc_inicial | precio_h0..h23 | pv_h0..h23 | demanda_h0..h23")
print(f"    Salidas  : p_bat_h0..h23 | soc_h0..h23")
print(f"    Split    : columna 'split' (train / test)")
print()