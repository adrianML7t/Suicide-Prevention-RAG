@echo off
:: 1. Eliminar los modelos existentes de llama y command sp
echo Eliminando modelos antiguos...
ollama rm llama3-70b-sp
:: 2. Crear el modelo nuevamente usando el Modelfile
echo Creando nuevos modelos adaptados
ollama create llama3-70b-sp -f Modelfile.llama70-sp

echo Proceso finalizado.
ollama list
pause