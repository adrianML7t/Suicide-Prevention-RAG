import create_database as DB
from langchain_ollama import OllamaLLM

DB_DIR = "chroma"

#Definir base de datos y LLM
db = DB.get_db()
llm = OllamaLLM(model="gpt-oss:120b-cloud")

#Retrieval. K es el numero de fragmentos que devuelve
query_text = "Consejos para personas en riesgo suicida"
results = db._similarity_search_with_relevance_scores(query_text, k=5)

#Muestra la mejor coincidencia
results = sorted(results, key=lambda x: x[1], reverse=True)

# Tomar el más relevante
top_doc, top_score = results[0]
context = top_doc.page_content.strip().replace("\n", " ") #En este caso al LLM le pasamos el mejor resultado

#Ajustar contexto y promts, y llamar a LLM

prompt_text = f"""
Teniendo en cuenta el siguiente contexto, responde a la pregunta como si fueras
un experto psicólogo en prevención del suicidio.
Si la respuesta no está en el contexto, di que no lo sabes.
Escribe en un formato que se pueda ver correctamenta en la terminal de python, evitando
tablas si es necesario.
Contexto:
{context}

Pregunta:
{query_text}
Respuesta:
            """

response = llm.invoke(prompt_text)


#---------------Mostrar resultados del LLM---------------------------#
print("\RESPUESTA GENERADA POR IA (RAG)")
print(f"Consulta: {query_text}")
print("="*80)
print(f"Respuesta:\n{response}")
print("-" * 80)
print(f"Contexto Utilizado (Relevancia: {top_score:.3f}):\n{context}")
print("="*80)