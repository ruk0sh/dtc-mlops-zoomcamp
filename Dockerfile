FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

COPY ["hw4_predict_duration.py", "run.py"]
COPY ["requirements.txt", "requirements.txt"]

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run.py"]