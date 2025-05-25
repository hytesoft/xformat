import os
import tempfile
import pypandoc
import subprocess
from typing import Union

def convert(input_data: Union[str, bytes], to_format: str) -> bytes:
    """
    统一格式转换接口
    :param input_data: 输入文件路径或字节流
    :param to_format: 目标格式（如 'pdf', 'md', 'pptx', 'html', 'txt'）
    :return: 转换后的文件字节流
    """
    # 1. 判断输入类型
    if isinstance(input_data, str) and os.path.isfile(input_data):
        in_path = input_data
        cleanup_in = False
    else:
        # 输入为字节流，写入临时文件
        tmp_in = tempfile.NamedTemporaryFile(delete=False)
        tmp_in.write(input_data)
        tmp_in.close()
        in_path = tmp_in.name
        cleanup_in = True

    # 2. 生成输出临时文件
    ext_map = {'md': 'md', 'markdown': 'md', 'html': 'html', 'pdf': 'pdf', 'pptx': 'pptx', 'txt': 'txt'}
    out_ext = ext_map.get(to_format.lower(), to_format.lower())
    out_path = tempfile.mktemp(suffix='.' + out_ext)

    # 3. 根据不同格式选择合适工具
    in_ext = os.path.splitext(in_path)[1][1:].lower()

    try:
        if to_format in ['md', 'markdown', 'html', 'pptx', 'pdf', 'txt', 'docx']:
            # 主要依赖Pandoc
            extra_args = []
            if to_format == 'pdf':
                extra_args = ['--pdf-engine=xelatex']
            pypandoc.convert_file(in_path, to_format, outputfile=out_path, extra_args=extra_args)
        elif in_ext == 'pdf' and to_format == 'html':
            subprocess.run(['pdf2htmlEX', in_path, out_path], check=True)
        elif in_ext == 'pdf' and to_format == 'txt':
            subprocess.run(['pdf2txt.py', in_path, '-o', out_path], check=True)
        else:
            raise NotImplementedError(f"暂不支持从 {in_ext} 到 {to_format} 的转换")
        with open(out_path, 'rb') as f:
            result = f.read()
    finally:
        if cleanup_in:
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
    return result