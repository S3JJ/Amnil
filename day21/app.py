# app.py
import os

log_dir = "logs"

# Creating the folder if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

from fastapi import FastAPI, UploadFile, File
from models.model import predict, get_model_info
from PIL import Image
import io
import logging

import time
import psutil
import torch
import GPUtil

# Logging configuration
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

app = FastAPI(title="Image Classification using EfficientNetB0.")

def get_gpu_usage():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return {
                "gpu_usage": f"{gpus[0].load * 100:.2f}%",
                "gpu_specs": gpus[0].name
            }
        return {"gpu_usage": "No GPU", "gpu_specs": "None"}
    except:
        return {"gpu_usage": "Unavailable", "gpu_specs": "Unavailable"}
    
def get_system_specs():
    cpu_specs = os.popen("wmic cpu get name").read().strip().split("\n")[-1]
    ram_total = round(psutil.virtual_memory().total / (1024**3), 2)

    return {
        "cpu_specs": cpu_specs,
        "ram_specs": f"{ram_total} GB"
    }

def get_system_usage():
    return {
        "cpu_usage": f"{psutil.cpu_percent()}%",
        "ram_usage": f"{round(psutil.virtual_memory().used / (1024**3), 2)} GB"
    }

@app.get("/")
def home():
    return {"message": "This is the backend for image classification using EfficientNetB0 created using FastAPI."}

request_count = 0
start_time = time.time()

@app.post("/predict")
async def prediction_endpoint(file: UploadFile = File(...)):
    """
    Receives uploaded image, performs prediction and returns:
    - Class name
    - WNID
    - Confidence
    - Latency
    - Throughput
    - System resource usage: CPU, GPU, RAM
    - Model stats
    """

    global request_count

    # Start latency timer
    t0 = time.time()

    # Read and load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Run prediction from model.py
    prediction = predict(image)

    # Compute latency (ms)
    latency_ms = (time.time() - t0) * 1000

    # Update throughput
    request_count += 1
    total_time_running = time.time() - start_time
    throughput_rps = request_count / total_time_running

    # System Usage
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    # GPU memory (only if cuda available)
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # In MB

    # Model Info (size, params)
    model_info = get_model_info()

    # Response construction
    final_response = {
        **prediction,  # class_name, wnid, confidence

        "latency_ms": round(latency_ms, 2),
        "throughput_rps": round(throughput_rps, 4),

        "cpu_usage_percent": cpu_usage,
        "ram_usage_percent": ram_usage,
        "gpu_memory_mb": gpu_memory,

        "model_parameters": model_info["num_parameters"],
        "model_size_mb": model_info["model_size_mb"]
    }

    return final_response