from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from xformat.converter import convert
import io

app = FastAPI(title="xformat REST API", description="多格式文档智能转换API")

@app.post("/convert")
async def convert_endpoint(file: UploadFile = File(...), to: str = Form(...)):
    input_bytes = await file.read()
    result = convert(input_bytes, to)
    filename = file.filename.rsplit('.', 1)[0] + "." + to
    return StreamingResponse(io.BytesIO(result), media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename={filename}"
    })