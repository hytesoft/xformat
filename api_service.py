import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path
import shutil

app = FastAPI()

PUBLIC_DIR = Path("./public")
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

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    save_path = PUBLIC_DIR / file.filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_type = simple_type_detect(file.filename)
    return JSONResponse({
        "filename": file.filename,
        "saved_path": str(save_path.resolve()),
        "file_type": file_type
    })
