FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt-get remove -y build-essential && apt-get autoremove -y

EXPOSE 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
