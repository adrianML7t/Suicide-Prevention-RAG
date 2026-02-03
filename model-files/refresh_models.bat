@echo off
:: 1. Eliminar los modelos existentes de llama y command sp
echo Eliminando modelos antiguos...
ollama rm llama3-sp
ollama rm com35-sp

:: 2. Crear el modelo nuevamente usando el Modelfile
echo Creando nuevos modelos adaptados
ollama create llama3-sp -f Modelfile.llama-sp
ollama create com35-sp -f Modelfile.com-sp

echo Proceso finalizado.
ollama list
pause