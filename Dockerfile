FROM python:latest
 
WORKDIR /usr/src/text_embedding

COPY embedding_module.py .

COPY requirements.txt .

COPY configs/ ./configs/

COPY src/ ./src/
 
RUN pip install -r requirements.txt

CMD ["python", "embedding_module.py"]