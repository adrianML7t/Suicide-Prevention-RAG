import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = 'subjects.csv'           # El archivo que creamos antes
ARCHIVO_SALIDA = 'resultados_100.csv'    # Donde se guardará el análisis
MODELO_OLLAMA = 'llama3.3:70b'                 # CAMBIA ESTO por tu modelo (ej: 'mistral', 'gemma:2b', etc.)

# Nombres exactos de las columnas del CSV generado
COL_FRASE = 'frase'
COL_ETIQUETA = 'tipo'

def limpiar_etiqueta_csv(valor):
    """
    Asegura que la etiqueta del CSV sea un número entero 1 o 3.
    Maneja casos donde pandas lea '1' como string o int.
    """
    s = str(valor).strip()
    if '1' in s: return 1
    if '3' in s: return 3
    return None

def consultar_llm(texto):
    """
    Consulta al modelo local.
    Prompt optimizado para distinguir conjugaciones verbales en español (sujetos tácitos).
    """
    prompt_sistema = (
        "Eres un experto en gramática española. Tu tarea es identificar la persona gramatical.\n"
        "Reglas:\n"
        "1. Si la frase está en PRIMERA persona (yo, nosotros/as) o el verbo lo indica (ej: 'fui', 'comimos'), responde '1'.\n"
        "2. Si la frase está en TERCERA persona (él, ella, ellos, eso, nombre propio) o el verbo lo indica (ej: 'fue', 'el perro corrió'), responde '3'.\n"
        "3. IMPORTANTE: Responde ÚNICAMENTE con el dígito (sin puntos, sin texto extra)."
    )

    try:
        response = ollama.chat(model=MODELO_OLLAMA, messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': f"Frase: \"{texto}\""}
        ])
        
        # Limpieza de la respuesta del LLM
        contenido = response['message']['content'].strip()
        
        # Buscamos el número en la respuesta
        if '1' in contenido: return 1
        if '3' in contenido: return 3
        return 0 # No concluyente
        
    except Exception as e:
        print(f"Error con Ollama: {e}")
        return -1 # Error de conexión

def main():
    print(f"--- ANALIZANDO {ARCHIVO_ENTRADA} CON MODELO {MODELO_OLLAMA} ---")
    
    # 1. Cargar CSV
    try:
        df = pd.read_csv(ARCHIVO_ENTRADA)
    except FileNotFoundError:
        print("ERROR: No se encuentra el archivo 'frases.csv'. Asegúrate de crearlo primero.")
        return

    # 2. Preparar datos
    # Convertimos la columna 'tipo' a números enteros limpios para comparar
    df['y_real'] = df[COL_ETIQUETA].apply(limpiar_etiqueta_csv)
    
    # Lista para guardar predicciones
    predicciones = []
    
    # 3. Bucle de inferencia (con barra de progreso)
    print(f"Procesando {len(df)} frases...")
    for frase in tqdm(df[COL_FRASE], unit="frase"):
        pred = consultar_llm(frase)
        predicciones.append(pred)

    df['y_pred'] = predicciones

    # 4. Cálculo de métricas
    # Filtramos filas donde el LLM falló en responder (0 o -1) para no ensuciar la métrica
    df_valido = df[(df['y_pred'] == 1) | (df['y_pred'] == 3)]
    
    # Guardamos el resultado completo antes de imprimir métricas
    df.to_csv(ARCHIVO_SALIDA, index=False)

    print("\n" + "="*40)
    print("INFORME DE RENDIMIENTO")
    print("="*40)
    
    if len(df_valido) > 0:
        acc = accuracy_score(df_valido['y_real'], df_valido['y_pred'])
        print(f"PRECISIÓN (Accuracy): {acc:.2%}")
        print("-" * 40)
        
        # Matriz de confusión simple
        tn, fp, fn, tp = confusion_matrix(df_valido['y_real'], df_valido['y_pred'], labels=[1, 3]).ravel()
        print(f"Aciertos 1a Persona: {tn}")
        print(f"Aciertos 3a Persona: {tp}")
        print(f"Errores (Predijo 3 siendo 1): {fp}")
        print(f"Errores (Predijo 1 siendo 3): {fn}")
        
        # Mostrar errores específicos si los hay
        errores = df_valido[df_valido['y_real'] != df_valido['y_pred']]
        if not errores.empty:
            print("\n--- EJEMPLOS DE ERRORES DEL LLM ---")
            for index, row in errores.head(5).iterrows():
                print(f"Frase: \"{row[COL_FRASE]}\"")
                print(f"   -> Real: {row['y_real']} | Predicho: {row['y_pred']}")
                print("-" * 20)
    else:
        print("El modelo no devolvió respuestas válidas (1 o 3). Revisa si Ollama está corriendo.")

    print(f"\nResultados guardados en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()