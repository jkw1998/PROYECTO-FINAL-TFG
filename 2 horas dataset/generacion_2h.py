# ==============================================================
# TFG - GENERACIÓN DE DATASET MEDIANTE OPTIMIZACIÓN MPC/MILP
# HORIZONTE TEMPORAL: T = 2 horas (ventana deslizante)
# ==============================================================
#
# Autor: James Kagunda Wangari
# Grado en Ingeniería Eléctrica - Universidad de Málaga
#
# Descripción:
# ------------
# Genera el dataset de 1000 escenarios diarios resolviendo
# 23 problemas MILP de 2 horas con ventana deslizante de 1h.
#
# Estrategia de horizonte T=2h (ventana deslizante):
# ---------------------------------------------------
# Cada ventana optimiza 2 horas consecutivas pero solo se
# EJECUTA (guarda como solución definitiva) la PRIMERA hora.
# La segunda hora se re-optimiza en la siguiente ventana con
# información actualizada. Esto sigue el principio del MPC:
# optimizar N pasos pero ejecutar solo el primero.
#
# El SOC final de la hora ejecutada se pasa como SOC inicial
# de la siguiente ventana, encadenando físicamente la batería.
#
# Esquema por día (23 ventanas de 2h, paso de 1h):
#
#   Ventana 1:  horas [0, 1]   -> ejecuta hora 0
#   Ventana 2:  horas [1, 2]   -> ejecuta hora 1
#   Ventana 3:  horas [2, 3]   -> ejecuta hora 2
#   ...
#   Ventana 23: horas [22, 23] -> ejecuta hora 22
#   Hora 23: se toma directamente de la ventana 23 (segunda hora)
#
# Por qué T=2h puede ser mejor que T=1h y T=24h:
# ------------------------------------------------
# - Frente a T=1h: tiene 1h de anticipación, puede planificar
#   micro-arbitraje (si la hora siguiente es más cara, carga ahora)
# - Frente a T=24h: el problema es más pequeño y resuelve
#   con mayor certeza en horizontes cortos donde la incertidumbre
#   de precios y PV es menor → soluciones más consistentes
#
# Estructura del dataset generado:
# ---------------------------------
#   dataset_vpp_completo_T2h.csv  -> todas las variables
#   dataset_vpp_ia_T2h.csv        -> entradas/salidas IA
#
# Columnas idénticas al dataset T=24h para comparación directa.
#
# ==============================================================

import os
import numpy as np
import pandas as pd
from pyomo.environ import *

# Directorio donde está este script — los CSV se guardarán aquí
DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================
# SEMILLA ALEATORIA (misma que T=24h para escenarios idénticos)
# ==============================================================

np.random.seed(42)

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================

DIAS       = 1000   # número total de escenarios diarios
T_DIA      = 24     # horas totales por día
T_VEN      = 2      # horas por ventana de optimización
PASO       = 1      # desplazamiento de la ventana (horas)
DELTA_T    = 1      # resolución temporal (h)

# Número de ventanas por día
# Ventana i cubre horas [i, i+1], i = 0..22
N_VENTANAS = T_DIA - T_VEN + 1   # = 23

DIAS_TRAIN = 800
DIAS_TEST  = 200

# ==============================================================
# PARÁMETROS DEL SISTEMA
# (idénticos a T=24h para comparación directa)
# ==============================================================

SOC_MIN    = 0.20
SOC_MAX    = 0.90
P_CH_MAX   = 0.30
P_DIS_MAX  = 0.50
ETA_CH     = 0.95
ETA_DIS    = 0.95
E_BAT_CAP  = 1.0
C_DEG      = 2.0

P_GRID_MAX   = 10.0
FACTOR_VENTA = 0.80

P_DG_MIN  = 1.0
P_DG_MAX  = 5.0
DG_RAMP   = 2.0
C_DG_B    = 10.0

# ==============================================================
# GENERACIÓN DE ESCENARIOS
# (función idéntica a T=24h — misma semilla → mismos escenarios)
# ==============================================================

def generar_escenario():
    t = np.arange(T_DIA)

    shift   = np.random.uniform(-3, 3)
    precios = (
        50
        + 30 * np.sin((t - 7 + shift) / 24 * 2 * np.pi)
        + np.random.normal(0, 4, T_DIA)
    )
    precios = np.clip(precios, 5, 120)

    pv = np.zeros(T_DIA)
    for h in range(T_DIA):
        if 7 <= h <= 18:
            pv[h] = 6 * np.sin((h - 6) / 12 * np.pi)
    pv *= np.random.uniform(0.6, 1.2)

    demanda = (
        1.5
        + 0.5 * np.sin((t - 6) / 24 * 2 * np.pi)
        + np.random.normal(0, 0.15, T_DIA)
    )
    demanda = np.clip(demanda, 0.5, 3.0)

    soc0 = np.random.uniform(SOC_MIN + 0.10, SOC_MAX - 0.10)

    return precios, pv, demanda, soc0

# ==============================================================
# MODELO DE OPTIMIZACIÓN MILP — VENTANA DE 2 HORAS
# ==============================================================

def optimizar_ventana_2h(precios_v, pv_v, demanda_v, soc_ini,
                          p_dg_prev=None):
    """
    Resuelve el problema MILP para una ventana de 2 horas.

    Parámetros
    ----------
    precios_v  : array (2,) - precios de las 2 horas de la ventana
    pv_v       : array (2,) - generación PV de las 2 horas
    demanda_v  : array (2,) - demanda de las 2 horas
    soc_ini    : float      - SOC al inicio de la ventana (p.u.)
    p_dg_prev  : float      - potencia DG hora anterior (para rampa)
                              None si es la primera ventana del día

    Retorna
    -------
    hora0 : dict con la solución de la primera hora de la ventana
            (la que se ejecuta como decisión definitiva)
            Incluye: p_bat, soc, p_red, p_dg, coste
    None si el solver no encuentra solución óptima
    """

    model = ConcreteModel()
    model.T = RangeSet(0, T_VEN - 1)   # índices 0, 1

    model.P_ch        = Var(model.T, bounds=(0, P_CH_MAX))
    model.P_dis       = Var(model.T, bounds=(0, P_DIS_MAX))
    model.SOC         = Var(model.T, bounds=(SOC_MIN, SOC_MAX))
    model.P_grid_buy  = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_grid_sell = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_dg        = Var(model.T, bounds=(P_DG_MIN, P_DG_MAX))
    model.u_ch        = Var(model.T, within=Binary)
    model.u_dis       = Var(model.T, within=Binary)

    # Función objetivo: coste total de las 2 horas de la ventana
    def objective_rule(m):
        return sum(
              precios_v[t] * m.P_grid_buy[t]  * DELTA_T
            - FACTOR_VENTA * precios_v[t] * m.P_grid_sell[t] * DELTA_T
            + C_DG_B * m.P_dg[t] * DELTA_T
            + C_DEG  * (m.P_ch[t] + m.P_dis[t]) * DELTA_T
            for t in m.T
        )

    model.obj = Objective(rule=objective_rule, sense=minimize)

    model.constraints = ConstraintList()

    for t in model.T:
        # Balance de potencia
        model.constraints.add(
            pv_v[t] + model.P_dis[t] + model.P_grid_buy[t] + model.P_dg[t]
            ==
            demanda_v[t] + model.P_ch[t] + model.P_grid_sell[t]
        )
        # Límites vinculados a binarias
        model.constraints.add(model.P_ch[t]  <= P_CH_MAX  * model.u_ch[t])
        model.constraints.add(model.P_dis[t] <= P_DIS_MAX * model.u_dis[t])
        # No simultaneidad
        model.constraints.add(model.u_ch[t] + model.u_dis[t] <= 1)

    # Dinámica del SOC hora a hora dentro de la ventana
    for t in model.T:
        soc_prev = soc_ini if t == 0 else model.SOC[t - 1]
        model.constraints.add(
            model.SOC[t]
            == soc_prev
               + (ETA_CH * model.P_ch[t] * DELTA_T
                  - model.P_dis[t] * DELTA_T / ETA_DIS) / E_BAT_CAP
        )

    # Rampa de la turbina entre la hora anterior y la primera hora
    # de esta ventana (continuidad entre ventanas)
    if p_dg_prev is not None:
        model.constraints.add(
            model.P_dg[0] - p_dg_prev <= DG_RAMP
        )
        model.constraints.add(
            p_dg_prev - model.P_dg[0] <= DG_RAMP
        )

    # Rampa entre las 2 horas internas de la ventana
    model.constraints.add(
        model.P_dg[1] - model.P_dg[0] <= DG_RAMP
    )
    model.constraints.add(
        model.P_dg[0] - model.P_dg[1] <= DG_RAMP
    )

    solver    = SolverFactory('highs')
    resultado = solver.solve(model, tee=False)

    if (resultado.solver.termination_condition
            != TerminationCondition.optimal):
        return None

    # Solo devolvemos la PRIMERA hora (hora 0 de la ventana)
    # que es la decisión que se ejecuta definitivamente
    return {
        'p_ch'   : value(model.P_ch[0]),
        'p_dis'  : value(model.P_dis[0]),
        'p_bat'  : value(model.P_ch[0]) - value(model.P_dis[0]),
        'soc'    : value(model.SOC[0]),
        'p_buy'  : value(model.P_grid_buy[0]),
        'p_sell' : value(model.P_grid_sell[0]),
        'p_red'  : value(model.P_grid_buy[0]) - value(model.P_grid_sell[0]),
        'p_dg'   : value(model.P_dg[0]),
        'coste'  : (  value(model.P_grid_buy[0])  * precios_v[0] * DELTA_T
                    - value(model.P_grid_sell[0]) * FACTOR_VENTA * precios_v[0] * DELTA_T
                    + C_DG_B * value(model.P_dg[0]) * DELTA_T
                    + C_DEG  * (value(model.P_ch[0]) + value(model.P_dis[0])) * DELTA_T),
        # Guardamos también la segunda hora para la última ventana
        'p_bat_h1' : value(model.P_ch[1]) - value(model.P_dis[1]),
        'soc_h1'   : value(model.SOC[1]),
        'p_red_h1' : value(model.P_grid_buy[1]) - value(model.P_grid_sell[1]),
        'p_dg_h1'  : value(model.P_dg[1]),
        'coste_h1' : (  value(model.P_grid_buy[1])  * precios_v[1] * DELTA_T
                      - value(model.P_grid_sell[1]) * FACTOR_VENTA * precios_v[1] * DELTA_T
                      + C_DG_B * value(model.P_dg[1]) * DELTA_T
                      + C_DEG  * (value(model.P_ch[1]) + value(model.P_dis[1])) * DELTA_T),
    }

# ==============================================================
# BUCLE PRINCIPAL DE GENERACIÓN
# ==============================================================

dataset_completo = []
dataset_ia       = []
fallos           = 0

print(f"\nGenerando {DIAS} escenarios con horizonte T=2h")
print(f"  Estrategia: ventana deslizante de 2h, paso 1h")
print(f"  Ventanas por día: {N_VENTANAS}")
print(f"  Se ejecuta: solo la primera hora de cada ventana")
print(f"  SOC encadenado: SOC_fin(ventana i) -> SOC_ini(ventana i+1)")
print(f"  Solver: HiGHS (MILP — optimalidad global)\n")

for d in range(DIAS):

    precios, pv, demanda, soc0 = generar_escenario()

    p_bat_dia  = np.zeros(T_DIA)
    soc_dia    = np.zeros(T_DIA)
    p_red_dia  = np.zeros(T_DIA)
    p_dg_dia   = np.zeros(T_DIA)
    coste_dia  = 0.0

    soc_actual = soc0
    p_dg_prev  = None
    fallo_dia  = False

    # --- 23 ventanas deslizantes de 2h ---
    # La ventana i cubre las horas [i, i+1]
    # Se ejecuta la hora i (primera de la ventana)
    # La hora 23 se toma de la última ventana (hora 1 de ventana 22)

    for i in range(N_VENTANAS):   # i = 0..22

        h_ini = i
        h_fin = i + T_VEN         # h_fin = i+2 (exclusivo)

        res = optimizar_ventana_2h(
            precios_v = precios[h_ini:h_fin],
            pv_v      = pv[h_ini:h_fin],
            demanda_v = demanda[h_ini:h_fin],
            soc_ini   = soc_actual,
            p_dg_prev = p_dg_prev
        )

        if res is None:
            fallo_dia = True
            break

        # Guardar la hora ejecutada (hora i = primera de la ventana)
        p_bat_dia[i] = round(res['p_bat'], 4)
        soc_dia[i]   = round(res['soc'],   4)
        p_red_dia[i] = round(res['p_red'], 4)
        p_dg_dia[i]  = round(res['p_dg'],  4)
        coste_dia   += res['coste']

        # Encadenar SOC: el SOC al final de la hora ejecutada
        # pasa como SOC inicial de la siguiente ventana
        soc_actual = res['soc']
        p_dg_prev  = res['p_dg']

        # La última ventana (i=22) cubre horas [22,23]
        # Guardamos también la hora 23 (segunda hora de la ventana 22)
        if i == N_VENTANAS - 1:
            p_bat_dia[23] = round(res['p_bat_h1'], 4)
            soc_dia[23]   = round(res['soc_h1'],   4)
            p_red_dia[23] = round(res['p_red_h1'], 4)
            p_dg_dia[23]  = round(res['p_dg_h1'],  4)
            coste_dia    += res['coste_h1']

    if fallo_dia:
        fallos += 1
        print(f"  [AVISO] Escenario {d} con fallo en alguna ventana — omitido")
        continue

    split = 'train' if d < DIAS_TRAIN else 'test'

    # ==========================================================
    # DATASET COMPLETO
    # ==========================================================

    fila_full = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4),
        'coste_total' : round(coste_dia, 4),
        'horizonte'   : 2,
    }
    for h in range(T_DIA):
        fila_full[f'precio_h{h}']  = round(precios[h], 4)
    for h in range(T_DIA):
        fila_full[f'pv_h{h}']      = round(pv[h], 4)
    for h in range(T_DIA):
        fila_full[f'demanda_h{h}'] = round(demanda[h], 4)
    for h in range(T_DIA):
        fila_full[f'p_bat_h{h}']   = p_bat_dia[h]
    for h in range(T_DIA):
        fila_full[f'soc_h{h}']     = soc_dia[h]
    for h in range(T_DIA):
        fila_full[f'p_red_h{h}']   = p_red_dia[h]
    for h in range(T_DIA):
        fila_full[f'p_dg_h{h}']    = p_dg_dia[h]

    dataset_completo.append(fila_full)

    # ==========================================================
    # DATASET IA (misma estructura que T=24h)
    # ==========================================================

    fila_ia = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4),
    }
    for h in range(T_DIA):
        fila_ia[f'precio_h{h}']  = round(precios[h], 4)
    for h in range(T_DIA):
        fila_ia[f'pv_h{h}']      = round(pv[h], 4)
    for h in range(T_DIA):
        fila_ia[f'demanda_h{h}'] = round(demanda[h], 4)
    for h in range(T_DIA):
        fila_ia[f'p_bat_h{h}']   = p_bat_dia[h]
    for h in range(T_DIA):
        fila_ia[f'soc_h{h}']     = soc_dia[h]

    dataset_ia.append(fila_ia)

    if (d + 1) % 100 == 0:
        print(f"  Escenario {d+1:4d}/{DIAS} completado  [{split}]")

# ==============================================================
# EXPORTACIÓN CSV
# ==============================================================

df_full = pd.DataFrame(dataset_completo)
df_ia   = pd.DataFrame(dataset_ia)

df_full.to_csv(os.path.join(DIR, "dataset_vpp_completo_T2h.csv"), index=False)
df_ia.to_csv(  os.path.join(DIR, "dataset_vpp_ia_T2h.csv"),       index=False)

# ==============================================================
# RESUMEN FINAL
# ==============================================================

escenarios_ok = DIAS - fallos

print("\n" + "=" * 58)
print("  DATASET T=2h GENERADO")
print("=" * 58)
print(f"  Horizonte por ventana : T = 2 h")
print(f"  Ventanas por día      : {N_VENTANAS} (deslizante, paso 1h)")
print(f"  Hora ejecutada        : primera hora de cada ventana")
print(f"  Anticipación          : 1 hora vista")
print(f"  Estrategia SOC        : encadenado ventana a ventana")
print(f"  Escenarios solicitados: {DIAS}")
print(f"  Resueltos con éxito   : {escenarios_ok}")
print(f"  Fallos omitidos       : {fallos}")
print(f"  Train / Test          : {DIAS_TRAIN} / {DIAS_TEST}")
print()
print("  Archivos generados:")
print("    1) dataset_vpp_completo_T2h.csv")
print("    2) dataset_vpp_ia_T2h.csv")
print()
print("  Estructura dataset IA (igual que T=24h):")
print("    Entradas (73): soc_inicial + precio×24 + pv×24 + demanda×24")
print("    Salidas  (48): p_bat×24 + soc×24")
print("=" * 58)