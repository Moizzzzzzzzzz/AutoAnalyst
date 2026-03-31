FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Lets you import the `autoanalyst` package when running inside the container.
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "autoanalyst.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
