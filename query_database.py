import create_database as DB
from langchain_ollama import OllamaLLM

DB_DIR = "chroma"

#Definir base de datos y LLM
db = DB.get_db()
#llm = OllamaLLM(model="gpt-oss:120b-cloud")
#llm = OllamaLLM(model="llama3")
llm = OllamaLLM(model="command-r:35b")


#Retrieval. K es el numero de fragmentos que devuelve
query_text = "Me siento muy deprimido ultimamente"
results = db._similarity_search_with_relevance_scores(query_text, k=1)

#Muestra la mejor coincidencia
results = sorted(results, key=lambda x: x[1], reverse=True)

# Tomar el más relevante
top_doc, top_score = results[0]
context = top_doc.page_content.strip().replace("\n", " ") #En este caso al LLM le pasamos el mejor resultado

#Ajustar contexto y promts, y llamar a LLM

prompt_text = f"""
Responde a la pregunta como si fueras un experto psicólogo en prevención del suicidio. 
Tu respuesta debe estar basada en el contexto proporcionado, pero adaptala a la pregunta para que 
la respuesta sea directa.
Si la respuesta no está en el contexto, di que no lo sabes.
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
print(f"Contexto Utilizado:\n{context}")
print("="*80)