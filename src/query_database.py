import sys
import create_database as DB
from langchain_ollama import OllamaLLM

# --- CONFIGURACIÓN ---
DB_DIR = "chroma/"

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

    # 2. Definir base de datos
    db = DB.get_db()

    # 3. Retrieval
    query_text = input("Escribe tu consulta: ")
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

    TU RESPUESTA (Empática y basada EXCLUSIVAMENTE en el contexto anterior):
    """

    # 5. Llamada al LLM
    response = llm.invoke(prompt_text)

 # --- Mostrar resultados ---
    print("\n" + "="*80)
    print(f"RESULTADOS PARA EL MODELO: {model_name}")
    print("="*80)
    print(f"Consulta: {query_text}")
    print("-" * 80)
    print(f"Respuesta:\n{response}")
    print("-" * 80)
    # CORRECCIÓN: Usamos 'full_context' en lugar de 'context'
    print(f"Contexto Utilizado (Score Top: {results[0][1] if results else 'N/A'}):\n{full_context}")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Verificamos que se haya pasado un argumento por terminal
    if len(sys.argv) < 2:
        print("Uso: python nombre_del_script.py [llama3|com35|gpt120]")
    else:
        modelo_elegido = sys.argv[1]
        run_rag_test(modelo_elegido)