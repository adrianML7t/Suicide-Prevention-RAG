import sys
import create_database as DB
from langchain_ollama import OllamaLLM

# --- CONFIGURACIÓN ---
DB_DIR1 = "chroma1/"
DB_DIR3 = "chroma3/"

def run_rag_test(model_name):
    # 1. Set LLM to input
    if model_name == "llama3":
        llm = OllamaLLM(model="llama3-sp")
    elif model_name == "com35":
        llm = OllamaLLM(model="com35-sp")
    elif model_name == "gpt120":
        llm = OllamaLLM(model="gpt-oss:120b-cloud")
    else:
        print(f"Error: Modelo '{model_name}' no reconocido.")
        return

    # 2. Definir base de datos en base a la consulta
    query_text = input("Escribe tu consulta: ")
    chroma_path = get_route(llm,query_text) #baja el rendimiento pasar el llm a una funcion?
    db = DB.get_db(chroma_path)

    # 3. Retrieval
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
  
    context_list = []
    for doc, score in results:
        # Limpiamos el texto y lo añadimos a la lista
        clean_content = doc.page_content.strip().replace("\n", " ")
        context_list.append(f"- {clean_content}")

    # Unimos todo en un solo string separado por saltos de línea
    full_context = "\n\n".join(context_list)

# 4. Prompt dinámico optimizado para Salud Mental
    prompt_text = f"""
    Instrucciones para el Asistente:
    Estás respondiendo a una persona que podría estar en crisis. Usa la siguiente información de las guías de salud (Contexto) para responderle con empatía.
    
    INFORMACIÓN DE LAS GUÍAS (CONTEXTO):
    {full_context}

    PREGUNTA DEL USUARIO:
    {query_text}

    TU RESPUESTA (basada EXCLUSIVAMENTE en el contexto anterior):
    """

    # 5. Llamada al LLM
    response = llm.invoke(prompt_text)

 # --- Mostrar resultados --- -> pasar a show_results
    print("\n" + "="*80)
    print(f"RESULTADOS PARA EL MODELO: {model_name}")
    print("="*80)
    print(f"Consulta: {query_text}")
    print("-" * 80)
    print(f"Respuesta:\n{response}")
    print("-" * 80)
    print(f"Contexto Utilizado (Score Top: {results[0][1] if results else 'N/A'}):\n{full_context}")
    print("="*80 + "\n")


def get_route(llm, query): # Mas adelante introducir aqui la clasificacion de riesgo ??
    print(f" Analizando intención de la consulta...")
        
    router_prompt = f"""
    Analiza la siguiente consulta sobre suicidio y clasifícala.
    
    Consulta: "{query}"
    
    Si el usuario pide ayuda para SÍ MISMO o describe sus propios sentimientos, responde: PROPIO
    Si el usuario pregunta cómo ayudar a OTRA PERSONA (hijo, amigo, pareja), responde: OTRO
    
    Responde SOLAMENTE con una de esas dos palabras (PROPIO u OTRO).
    """
    # El LLM decide
    decision = llm.invoke(router_prompt).strip().upper()
    
    # Lógica de selección de base de datos
    if "OTRO" in decision:
        print(">> Modo detectado: Ayuda a un TERCERO (Familiares/Amigos)")
        selected_db_path = DB_DIR3
    else:
        # Por seguridad, si duda, asumimos que es el propio usuario (PROPIO)
        print(">> Modo detectado: Ayuda PROPIA (Paciente)")
        selected_db_path = DB_DIR1
    return selected_db_path

if __name__ == "__main__":
    # Verificamos que se haya pasado un argumento por terminal
    if len(sys.argv) < 2:
        print("Uso: python nombre_del_script.py [llama3|com35|gpt120]")
    else:
        modelo_elegido = sys.argv[1]
        run_rag_test(modelo_elegido)