FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY agents/RAG_agent/agents/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agents/RAG_agent/agents /app/agents
COPY agents/RAG_agent/rag_tools /app/rag_tools

WORKDIR /app

EXPOSE 8091

CMD ["uvicorn", "agents.agent:app", "--host", "0.0.0.0", "--port", "8091"]
