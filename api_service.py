import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
import shutil
from pipeline import run_pipeline
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
PUBLIC_DIR = Path(config.get("public_dir", "./public"))
PUBLIC_DIR.mkdir(exist_ok=True)

def simple_type_detect(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ["pdf", "doc", "docx", "txt", "md"]:
        return "document"
    elif ext in ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]:
        return "image"
    elif ext in ["mp3", "wav", "flac", "aac", "ogg"]:
        return "audio"
    elif ext in ["mp4", "avi", "mov", "mkv", "flv"]:
        return "video"
    elif ext in ["ppt", "pptx"]:
        return "presentation"
    else:
        return "unknown"

app = FastAPI()

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    save_path = str(PUBLIC_DIR / filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_type = simple_type_detect(filename)
    return JSONResponse({
        "filename": filename,
        "saved_path": save_path,
        "file_type": file_type
    })

@app.post("/process")
def process_file(filename: str):
    input_path = str(PUBLIC_DIR / filename)
    if not os.path.exists(input_path):
        return JSONResponse({"error": f"File not found: {input_path}"}, status_code=404)
    try:
        run_pipeline(input_path)
        return JSONResponse({
            "status": "success",
            "input_file": filename,
            "output_dir": "/mnt/share/",  # 或其它实际输出目录
            "msg": "多模态处理已完成，请查看输出目录"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)

@app.post("/process_upload")
def process_upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    save_path = str(PUBLIC_DIR / filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        run_pipeline(save_path)
        return JSONResponse({
            "status": "success",
            "input_file": filename,
            "output_dir": "/mnt/share/",  # 或其它实际输出目录
            "msg": "多模态处理已完成，请查看输出目录"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
