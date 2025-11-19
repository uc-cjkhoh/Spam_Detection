FROM python:latest
 
WORKDIR /usr/src/text_embedding

COPY run_embedding.py .

COPY requirements.txt .

COPY configs/ ./configs/

COPY src/ ./src/
 
RUN pip install -r requirements.txt

CMD ["python", "run_embedding.py"]