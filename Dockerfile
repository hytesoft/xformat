FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    pandoc \
    texlive \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra \
    python3 \
    python3-pip \
    pdf2htmlex

RUN pip3 install pypandoc pdfminer.six fastapi uvicorn

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]