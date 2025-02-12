# Usa una imagen base con Python y CUDA (ajusta según tu entorno)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Crear un directorio de trabajo
WORKDIR /app

# Instalar vLLM
RUN pip install --no-cache-dir vllm

# Comando por defecto para ejecutar el servicio
CMD ["vllm", "serve", "Lauther/emb-multilingual-e5-large-instruct-3e", "--dtype", "auto", "--api-key", "", "--task", "embed"]


# sudo docker-compose build
# sudo docker-compose up -d
# sudo docker logs -f vllm_embed
