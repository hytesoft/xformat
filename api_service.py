import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
import shutil
from multimodal_pipeline_dag import run_pipeline
import yaml
import uuid
import time

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


@app.post("/process")
def process_file(file: UploadFile = File(...)):
    # 上传即处理：保存文件、生成ID、自动run_pipeline
    ts = time.strftime('%Y%m%d%H%M%S')
    uniq = uuid.uuid4().hex[:8]
    orig_filename = file.filename or "uploaded_file"
    ext = orig_filename.rsplit('.', 1)[-1] if '.' in orig_filename else 'bin'
    file_id = f"{ts}{uniq}"
    new_filename = f"{file_id}.{ext}"
    save_path = str(PUBLIC_DIR / new_filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        run_pipeline(file_id)
        return JSONResponse({
            "status": "success",
            "id": file_id,
            "filename": new_filename,
            "output_dir": str(PUBLIC_DIR),
            "msg": "多模态处理已完成，请查看输出目录"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "msg": str(e)}, status_code=500)
