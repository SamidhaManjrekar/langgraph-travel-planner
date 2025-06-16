FROM python:3.11-slim-buster

WORKDIR /app

RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.main:api_app", "--host", "0.0.0.0", "--port", "8001"]
