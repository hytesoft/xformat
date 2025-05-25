from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from xformat.core import convert, convert_text, convert_url
import io

app = FastAPI(title="xformat REST API", description="多格式文档智能转换API")

@app.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
    to: str = Form(...),
    from_format: str = Form(None)
):
    """
    文件上传接口，支持文件格式转换
    """
    input_path = f"/tmp/{file.filename}"
    output_path = f"/tmp/converted.{to}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    convert(input_path, output_path, from_format=from_format, to_format=to)
    with open(output_path, "rb") as f:
        result = f.read()
    return StreamingResponse(io.BytesIO(result), media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename=converted.{to}"
    })

@app.post("/convert_text")
async def convert_text_api(
    content: str = Form(...),
    from_format: str = Form(...),
    to_format: str = Form(...)
):
    """
    文本内容直接转换接口
    """
    try:
        result = convert_text(content, from_format, to_format)
        return PlainTextResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/convert_url")
async def convert_url_api(
    url: str = Query(...),
    to_format: str = Query(...)
):
    """
    URL 网页内容转换接口
    """
    try:
        result = convert_url(url, to_format)
        return PlainTextResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)