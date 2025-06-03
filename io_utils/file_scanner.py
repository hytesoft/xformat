import os
import json
from typing import List

def scan_files(input_path: str, output_json: str):
    """
    递归扫描输入目录或单文件，输出文件列表到 output_json。
    """
    file_list: List[str] = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                file_list.append(os.path.join(root, f))
    elif os.path.isfile(input_path):
        file_list.append(input_path)
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(file_list, f, ensure_ascii=False, indent=2)
