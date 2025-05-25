FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    pandoc \
    texlive \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra \
    python3 \
    python3-pip \
    pdf2htmlex \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "xformat.api:app", "--host", "0.0.0.0", "--port", "8000"]