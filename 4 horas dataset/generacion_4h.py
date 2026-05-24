# ==============================================================
# TFG - GENERACIÓN DE DATASET MEDIANTE OPTIMIZACIÓN MPC/MILP
# HORIZONTE TEMPORAL: T = 4 horas (ventana deslizante)
# ==============================================================
#
# Autor: James Kagunda Wangari
# Grado en Ingeniería Eléctrica - Universidad de Málaga
#
# Descripción:
# ------------
# Genera el dataset de 1000 escenarios diarios resolviendo
# problemas MILP de 4 horas con ventana deslizante de paso 3h.
#
# Estrategia de horizonte T=4h (ventana deslizante):
# ---------------------------------------------------
# Cada ventana optimiza 4 horas consecutivas pero solo se
# EJECUTAN (guardan como solución definitiva) las primeras
# 3 horas. La 4ª hora actúa como "buffer de anticipación"
# y se re-optimiza en la siguiente ventana.
# Principio MPC: optimizar N pasos, ejecutar N-1.
#
# El SOC al final de la hora 3 ejecutada (es decir, la 3ª
# hora de la ventana actual) se pasa como SOC inicial de la
# siguiente ventana, encadenando físicamente la batería.
#
# Esquema por día:
#
#   Ventana 1: horas [0,1,2,3]   -> ejecuta horas 0,1,2
#   Ventana 2: horas [3,4,5,6]   -> ejecuta horas 3,4,5
#   Ventana 3: horas [6,7,8,9]   -> ejecuta horas 6,7,8
#   Ventana 4: horas [9,10,11,12]-> ejecuta horas 9,10,11
#   Ventana 5: horas [12,13,14,15]->ejecuta horas 12,13,14
#   Ventana 6: horas [15,16,17,18]->ejecuta horas 15,16,17
#   Ventana 7: horas [18,19,20,21]->ejecuta horas 18,19,20
#   Ventana 8: horas [21,22,23]   -> ejecuta horas 21,22,23
#              (última ventana: solo 3h disponibles, T=3h)
#
# Total: 7 ventanas completas de 4h + 1 ventana final de 3h
#        = 24 horas cubiertas en total
#
# Por qué T=4h:
# --------------
# Más anticipación que T=2h (ve 4h en vez de 2h), pero menor
# que T=24h. Permite planificar ciclos de carga/descarga dentro
# de un bloque de medio turno laboral. Buen compromiso entre
# calidad de solución y certeza de la información.
#
# Estructura del dataset generado:
# ---------------------------------
#   dataset_vpp_completo_T4h.csv  -> todas las variables
#   dataset_vpp_ia_T4h.csv        -> entradas/salidas IA
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

DIAS       = 1000
T_DIA      = 24
T_VEN      = 4      # horas por ventana
PASO       = 3      # horas ejecutadas por ventana (= paso deslizante)
DELTA_T    = 1

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
# MODELO DE OPTIMIZACIÓN MILP — VENTANA DE T_V HORAS
# ==============================================================

def optimizar_ventana(precios_v, pv_v, demanda_v, soc_ini,
                       p_dg_prev=None,
                       es_ultima_ventana=False, soc0_dia=None):
    """
    Resuelve el problema MILP para una ventana de T_v horas
    (T_v puede ser 4 o 3 para la última ventana del día).

    Parámetros
    ----------
    precios_v         : array (T_v,) - precios de las T_v horas
    pv_v              : array (T_v,) - generación PV de las T_v horas
    demanda_v         : array (T_v,) - demanda de las T_v horas
    soc_ini           : float        - SOC al inicio de la ventana
    p_dg_prev         : float        - potencia DG hora anterior
    es_ultima_ventana : bool          - True si es la última ventana
    soc0_dia          : float         - SOC inicial del día

    Retorna
    -------
    dict con la solución de TODAS las horas de la ventana,
    indexadas como 'p_bat_h0', 'soc_h0', ..., 'p_bat_h{T_v-1}'
    None si el solver falla
    """

    T_v   = len(precios_v)
    model = ConcreteModel()
    model.T = RangeSet(0, T_v - 1)

    model.P_ch        = Var(model.T, bounds=(0, P_CH_MAX))
    model.P_dis       = Var(model.T, bounds=(0, P_DIS_MAX))
    model.SOC         = Var(model.T, bounds=(SOC_MIN, SOC_MAX))
    model.P_grid_buy  = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_grid_sell = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_dg        = Var(model.T, bounds=(P_DG_MIN, P_DG_MAX))
    model.u_ch        = Var(model.T, within=Binary)
    model.u_dis       = Var(model.T, within=Binary)

    P_NEU = 500.0

    if es_ultima_ventana and soc0_dia is not None:
        model.s_neg = Var(bounds=(0, None))

    def objective_rule(m):
        coste = sum(
              precios_v[t] * m.P_grid_buy[t]  * DELTA_T
            - FACTOR_VENTA * precios_v[t] * m.P_grid_sell[t] * DELTA_T
            + C_DG_B * m.P_dg[t] * DELTA_T
            + C_DEG  * (m.P_ch[t] + m.P_dis[t]) * DELTA_T
            for t in m.T
        )
        if es_ultima_ventana and soc0_dia is not None:
            coste += P_NEU * m.s_neg
        return coste

    model.obj = Objective(rule=objective_rule, sense=minimize)
    model.constraints = ConstraintList()

    for t in model.T:
        model.constraints.add(
            pv_v[t] + model.P_dis[t] + model.P_grid_buy[t] + model.P_dg[t]
            ==
            demanda_v[t] + model.P_ch[t] + model.P_grid_sell[t]
        )
        model.constraints.add(model.P_ch[t]  <= P_CH_MAX  * model.u_ch[t])
        model.constraints.add(model.P_dis[t] <= P_DIS_MAX * model.u_dis[t])
        model.constraints.add(model.u_ch[t]  + model.u_dis[t] <= 1)

    for t in model.T:
        soc_prev = soc_ini if t == 0 else model.SOC[t - 1]
        model.constraints.add(
            model.SOC[t]
            == soc_prev
               + (ETA_CH * model.P_ch[t] * DELTA_T
                  - model.P_dis[t] * DELTA_T / ETA_DIS) / E_BAT_CAP
        )

    # Rampa entre hora anterior al inicio de la ventana
    if p_dg_prev is not None:
        model.constraints.add(model.P_dg[0] - p_dg_prev <= DG_RAMP)
        model.constraints.add(p_dg_prev - model.P_dg[0] <= DG_RAMP)

    # Rampas internas de la ventana
    for t in model.T:
        if t > 0:
            model.constraints.add(model.P_dg[t] - model.P_dg[t-1] <= DG_RAMP)
            model.constraints.add(model.P_dg[t-1] - model.P_dg[t] <= DG_RAMP)

    # Neutralidad energética diaria (blanda): solo en la última ventana.
    # s_neg captura el déficit SOC_fin(última hora) - soc0_dia.
    if es_ultima_ventana and soc0_dia is not None:
        ultima_t = T_v - 1
        model.constraints.add(
            model.s_neg >= soc0_dia - model.SOC[ultima_t]
        )

    solver = SolverFactory('highs')
    try:
        resultado = solver.solve(model, tee=False)
        if (resultado.solver.termination_condition
                != TerminationCondition.optimal):
            return None
    except Exception:
        return None

    # Devolver la solución de todas las horas de la ventana
    res = {}
    for t in range(T_v):
        p_bat = value(model.P_ch[t]) - value(model.P_dis[t])
        p_red = value(model.P_grid_buy[t]) - value(model.P_grid_sell[t])
        coste = (  value(model.P_grid_buy[t])  * precios_v[t] * DELTA_T
                 - value(model.P_grid_sell[t]) * FACTOR_VENTA * precios_v[t] * DELTA_T
                 + C_DG_B * value(model.P_dg[t]) * DELTA_T
                 + C_DEG  * (value(model.P_ch[t]) + value(model.P_dis[t])) * DELTA_T)
        res[t] = {
            'p_bat' : round(p_bat, 4),
            'soc'   : round(value(model.SOC[t]), 4),
            'p_red' : round(p_red, 4),
            'p_dg'  : round(value(model.P_dg[t]), 4),
            'coste' : coste,
        }
    return res

# ==============================================================
# BUCLE PRINCIPAL DE GENERACIÓN
# ==============================================================

dataset_completo = []
dataset_ia       = []
fallos           = 0

# Definir las ventanas del día
# Ventanas completas: inicio en 0, 3, 6, 9, 12, 15, 18 (paso=3)
# Última ventana: horas [21, 22, 23] (solo 3h disponibles)

ventanas = []
h = 0
while h < T_DIA:
    h_fin = min(h + T_VEN, T_DIA)   # no sobrepasar el día
    ventanas.append((h, h_fin))
    h += PASO

print(f"\nGenerando {DIAS} escenarios con horizonte T=4h")
print(f"  Estrategia: ventana deslizante de 4h, paso 3h")
print(f"  Ventanas por día: {len(ventanas)}")
for i, (h0, h1) in enumerate(ventanas):
    n_exec = h1 - h0
    print(f"    Ventana {i+1}: horas [{h0}..{h1-1}] "
          f"({h1-h0}h) — ejecuta {n_exec} hora(s)")
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

    for (h_ini, h_fin) in ventanas:

        res = optimizar_ventana(
            precios_v         = precios[h_ini:h_fin],
            pv_v              = pv[h_ini:h_fin],
            demanda_v         = demanda[h_ini:h_fin],
            soc_ini           = soc_actual,
            p_dg_prev         = p_dg_prev,
            es_ultima_ventana = ((h_ini, h_fin) == ventanas[-1]),
            soc0_dia          = soc0
        )

        if res is None:
            fallo_dia = True
            break

        n_horas_ventana = h_fin - h_ini

        # Ejecutar las primeras PASO horas (o todas si la ventana
        # es más corta que PASO — caso de la última ventana)
        n_ejecutar = min(PASO, n_horas_ventana)

        for k in range(n_horas_ventana):
            hora_global = h_ini + k
            p_bat_dia[hora_global] = res[k]['p_bat']
            soc_dia[hora_global]   = res[k]['soc']
            p_red_dia[hora_global] = res[k]['p_red']
            p_dg_dia[hora_global]  = res[k]['p_dg']
            coste_dia             += res[k]['coste']

        # Encadenar: SOC al final de la última hora ejecutada
        ultima_hora_ejecutada = n_ejecutar - 1
        soc_actual = res[ultima_hora_ejecutada]['soc']
        p_dg_prev  = res[ultima_hora_ejecutada]['p_dg']

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
        'horizonte'   : 4,
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

df_full.to_csv(os.path.join(DIR, "dataset_vpp_completo_T4h.csv"), index=False)
df_ia.to_csv(  os.path.join(DIR, "dataset_vpp_ia_T4h.csv"),       index=False)

# ==============================================================
# RESUMEN FINAL
# ==============================================================

escenarios_ok = DIAS - fallos

print("\n" + "=" * 58)
print("  DATASET T=4h GENERADO")
print("=" * 58)
print(f"  Horizonte por ventana : T = 4 h (última: 3 h)")
print(f"  Paso deslizante       : {PASO} horas")
print(f"  Horas ejecutadas/vent.: primeras {PASO} horas de cada ventana")
print(f"  Anticipación          : 1 hora buffer (4ª hora)")
print(f"  Estrategia SOC        : encadenado ventana a ventana")
print(f"  Escenarios solicitados: {DIAS}")
print(f"  Resueltos con éxito   : {escenarios_ok}")
print(f"  Fallos omitidos       : {fallos}")
print(f"  Train / Test          : {DIAS_TRAIN} / {DIAS_TEST}")
print()
print("  Archivos generados:")
print("    1) dataset_vpp_completo_T4h.csv")
print("    2) dataset_vpp_ia_T4h.csv")
print()
print("  Estructura dataset IA (igual que T=24h):")
print("    Entradas (73): soc_inicial + precio×24 + pv×24 + demanda×24")
print("    Salidas  (48): p_bat×24 + soc×24")
print("=" * 58)