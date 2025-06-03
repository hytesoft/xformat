import json
from typing import Any

def write_output(data: Any, output_path: str):
    """
    将结构化数据写入 json 文件。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
