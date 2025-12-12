import pandas as pd
import os
from sklearn.utils import shuffle

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
YEAR = 2022  # <--- ¡CAMBIA ESTO POR EL AÑO QUE QUIERAS! (2020, 2021, etc.)

# Rutas dinámicas (fr'...' permite usar llaves {} dentro de rutas de Windows)
archivo_entrada = fr'D:\SIG\csv\{YEAR}\Dataset_Completo_{YEAR}.csv' 
archivo_salida  = fr'D:\SIG\csv\{YEAR}\Dataset_{YEAR}_BALANCEADO_FINAL.csv'

# ==========================================
# 2. PROCESAMIENTO
# ==========================================

print(f"🚀 INICIANDO PROCESAMIENTO INTEGRAL AÑO {YEAR} (LIMPIEZA + BALANCEO)...\n")

try:
    # ---------------------------------------------------------
    # FASE 1: CARGA Y LIMPIEZA BÁSICA
    # ---------------------------------------------------------
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {archivo_entrada}")
    
    df = pd.read_csv(archivo_entrada)
    total_inicio = len(df)
    print(f"1️⃣  Archivo cargado ({YEAR}). Filas totales: {total_inicio}")

    # A. Eliminar Vacíos (Buena práctica general)
    df_limpio = df.dropna()
    eliminados_nulos = total_inicio - len(df_limpio)
    
    if eliminados_nulos > 0:
        print(f"    ⚠️ Se eliminaron {eliminados_nulos} filas con datos vacíos (huecos por nubes/bordes).")
    
    print(f"    ✅ Fase de Limpieza completada. Filas útiles: {len(df_limpio)}")

    # ---------------------------------------------------------
    # FASE 2: BALANCEO DE CLASES (UNDERSAMPLING)
    # ---------------------------------------------------------
    print("\n2️⃣  Iniciando Balanceo de Clases...")
    
    if 'class' not in df_limpio.columns:
        raise ValueError("El archivo no tiene la columna 'class'.")

    conteo = df_limpio['class'].value_counts()
    print(f"    Conteo previo:\n{conteo.to_string()}")

    # Separar por clases
    df_incendios = df_limpio[df_limpio['class'] == 1]
    df_no_incendios = df_limpio[df_limpio['class'] == 0]

    # Calcular el mínimo para igualar cantidades
    n_muestras = min(len(df_incendios), len(df_no_incendios))
    
    print(f"    ✂️ Recortando ambas clases a {n_muestras} muestras exactas.")

    # Muestreo aleatorio (semilla 42 para que sea repetible)
    df_inc_bal = df_incendios.sample(n=n_muestras, random_state=42)
    df_no_inc_bal = df_no_incendios.sample(n=n_muestras, random_state=42)

    # Unir y Mezclar (Shuffle) para que no estén ordenados
    df_final = pd.concat([df_inc_bal, df_no_inc_bal])
    df_final = shuffle(df_final, random_state=42)

    # ---------------------------------------------------------
    # FASE 3: GUARDAR RESULTADO
    # ---------------------------------------------------------
    # Verificar que la carpeta de salida exista (por seguridad)
    carpeta_salida = os.path.dirname(archivo_salida)
    if not os.path.exists(carpeta_salida):
        try:
            os.makedirs(carpeta_salida)
            print(f"    📁 Carpeta creada: {carpeta_salida}")
        except:
            pass

    df_final.to_csv(archivo_salida, index=False)

    print("\n" + "="*50)
    print(f"🎉 PROCESO {YEAR} FINALIZADO CON ÉXITO")
    print("="*50)
    print(f"📂 Archivo de Salida: {archivo_salida}")
    print(f"📊 Total de filas:    {len(df_final)}")
    print(f"⚖️  Balance final:")
    print(df_final['class'].value_counts().to_string())
    print("="*50)

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO: {e}")