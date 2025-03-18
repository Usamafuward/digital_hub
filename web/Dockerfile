FROM python:3.11-slim

WORKDIR /app

COPY ./web/requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./web /app/web
COPY ./static /app/static

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8002"]
