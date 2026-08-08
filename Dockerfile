FROM python:3.11-slim

# Cria um usuário não-root (UID 1000) — requisito de segurança do
# Hugging Face Spaces e boa prática geral de containers
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação e os artefatos do modelo
COPY app/ ./app/
COPY models/ ./models/

# Ajusta a posse dos arquivos para o usuário não-root antes de trocar de usuário
RUN chown -R appuser:appuser /app

USER appuser

# Porta exposta por padrão (documentação); em produção o Render injeta
# a porta real via variável de ambiente PORT
EXPOSE 7860

# Usa a variável PORT do Render se disponível; senão usa 7860 (padrão
# para testes locais ou Hugging Face Spaces)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
