FROM python:3.11-slim

WORKDIR /app

COPY ./api/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./api /app/api
COPY ./documents /app/documents
COPY ./vector_store /app/vector_store

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]