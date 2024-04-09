FROM python:3.9

WORKDIR /sse-dti

COPY requirements.txt .
COPY ./src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install torch-scatter torch-sparse

CMD ["python", "./src/main.py"]