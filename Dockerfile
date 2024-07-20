FROM python:3.11.9

RUN apt-get update && apt-get install -y postgresql-client

RUN mkdir /app

COPY ./api /app/api/
COPY ./data /app/data/
COPY ./aproved_models /app/aproved_models/
COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app/api/v1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]