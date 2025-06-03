FROM python:3.11-slim

# 替换为中科大Debian源，加速apt（适配bookworm及以后）
RUN echo "deb https://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free non-free-firmware\n\
deb https://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware\n\
deb https://mirrors.ustc.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware\n\
deb https://mirrors.ustc.edu.cn/debian-security/ bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list

# 设置pip默认源为清华镜像，所有pip操作都走国内源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update
WORKDIR /app
COPY . /app

RUN pip3 install --upgrade setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]