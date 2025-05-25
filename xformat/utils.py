import os

def ensure_file_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")