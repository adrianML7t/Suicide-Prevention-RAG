import sys
import create_database as DB
from langchain_ollama import OllamaLLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
DB_DIR1 = "chroma1/"
DB_DIR3 = "chroma3/"

def run_rag_test(model_name):
    # 1. Set LLM to input
    if model_name == "llama3-8b":
        llm = OllamaLLM(model="llama3-8b-sp")
    elif model_name == "com35":
        llm = OllamaLLM(model="com35-sp")
    elif model_name == "gpt120":
        llm = OllamaLLM(model="gpt-oss:120b-cloud")
    elif model_name == "llama3-70b":
        llm = OllamaLLM(model="llama3-70b-sp")
    else:
        print(f"Error: Modelo '{model_name}' no reconocido.")
        return

    while True:
        # 2. Choose database based on the query
        query_text = input("Escribe tu consulta: ")
        chroma_path = get_route(llm,query_text)
        db = DB.get_db(chroma_path)

        # 3. Retrieval
        results = db.similarity_search_with_relevance_scores(query_text, k=5)
    
        context_list = []
        for doc, score in results:
            # Clean the text and add it to the list
            clean_content = doc.page_content.strip().replace("\n", " ")
            context_list.append(f"- {clean_content}")

        # Join everything into a single string separated by blank lines
        full_context = "\n\n".join(context_list)

    # 4. Dynamic prompt optimized for Mental Health
        prompt_text = f"""
        Instrucciones para el Asistente:
        Estás respondiendo a una persona que podría estar en crisis de salud mental. Usa la siguiente información de las guías de salud (Contexto) para responderle con empatía.
        Asegurate de proponer recursos de ayuda apropiados, siempre proporcionando un número de contacto.
        
        INFORMACIÓN DE LAS GUÍAS (CONTEXTO):
        {full_context}

        PREGUNTA DEL USUARIO:
        {query_text}

        TU RESPUESTA (basada EXCLUSIVAMENTE en el contexto anterior):
        """

        # 5. Call to the LLM
        response = llm.invoke(prompt_text)

        print("\n" + "="*80)
        print("-" * 80)
        print(f"Respuesta:\n{response}")
        print("-" * 80)
       


def get_route(llm, query):
    print(f" Analizando intención de la consulta...")
        
    router_prompt = f"""
    Analiza la siguiente consulta sobre suicidio y clasifícala.
    
    Consulta: "{query}"
    
    Si el usuario pide ayuda para SÍ MISMO o describe sus propios sentimientos, responde: PROPIO
    Si el usuario pregunta cómo ayudar a OTRA PERSONA (hijo, amigo, pareja), responde: OTRO
    
    Responde SOLAMENTE con una de esas dos palabras (PROPIO u OTRO).
    """
    # The LLM decides
    decision = llm.invoke(router_prompt).strip().upper()
    
    # Database selection logic
    if "OTRO" in decision:
        print(">> Modo detectado: Ayuda a un TERCERO (Familiares/Amigos)")
        selected_db_path = DB_DIR3
    else:
        # For safety, if uncertain, assume the user themselves (PROPIO)
        print(">> Modo detectado: Ayuda PROPIA (Paciente)")
        selected_db_path = DB_DIR1
    return selected_db_path

if __name__ == "__main__":
    # Verify that a command-line argument was provided
    if len(sys.argv) < 2:
        print("Uso: python nombre_del_script.py [llama3|com35|gpt120]")
    else:
        modelo_elegido = sys.argv[1]
        run_rag_test(modelo_elegido)
