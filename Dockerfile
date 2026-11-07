FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/processed ./data/processed
COPY data/raw/afl_players_info_raw.csv ./data/raw/afl_players_info_raw.csv
COPY models ./models

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "afl_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
