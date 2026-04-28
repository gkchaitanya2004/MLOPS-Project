
from fastapi import FastAPI
import joblib
import torch
import re
from langdetect import detect
from src.embeddings import get_embeddings_single
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Summary
from prometheus_fastapi_instrumentator import Instrumentator
import time
import os
import psutil

def mem_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info()
    return mem_bytes.rss


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("Models/model.pth",weights_only=False, map_location=device)
model = model.to(device)

gaurdrail_sentences_processed = Counter('gaurdrail_sentences_processed', 'Number of sentences processed',['mode'])
gaurdrail_active_requests = Gauge('gaurdrail_active_requests', 'Number of active requests',['session_id'])
gaurdrail_memory_usage_bytes = Gauge('gaurdrail_memory_usage_bytes', 'Memory usage in bytes',['session_id'])
gaurdrail_request_latency_seconds = Summary('gaurdrail_request_latency_seconds', 'Latency of requests in seconds',['mode'])                               
Instrumentator().instrument(app).expose(app)

class TextInput(BaseModel):
    text: str


@app.post("/predict")
def get_preds(txt: TextInput):
        
    try:
        start_time = time.time()
        gaurdrail_active_requests.labels(session_id=str(os.getpid())).inc()

        text = txt.text
        lang = detect(text)

        if lang != 'hi' and lang != 'te':
            return {"error": "Unsupported language. Only Hindi and Telugu are supported.", "Detected Language": lang}

        text = text.strip()
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)


        embedding = get_embeddings_single(text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        embedding_tensor = embedding_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(embedding_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            conf = torch.softmax(outputs, dim=1)[0][pred].item()

        if lang == 'hi':
            le = joblib.load('Models/hi_label_encoder.joblib')
        else:
            le = joblib.load('Models/te_label_encoder.joblib')

        pred_label = le.inverse_transform([pred])[0]
        return {"predicted_label": pred_label, "confidence": conf, "Detected Language": lang}

    finally:
           end_time = time.time()
           latency = end_time - start_time
           mem_usage_bytes = mem_usage()
           gaurdrail_sentences_processed.labels(mode="api").inc()
           gaurdrail_request_latency_seconds.labels(mode="api").observe(latency)
           gaurdrail_memory_usage_bytes.labels(session_id=str(os.getpid())).set(mem_usage_bytes)
           gaurdrail_active_requests.labels(session_id=str(os.getpid())).dec()