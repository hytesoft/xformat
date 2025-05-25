import os
from .backends import convert_with_pandoc, pdf_to_txt, pdf_to_html

SUPPORTED_FORMATS = ['md', 'markdown', 'pdf', 'html', 'txt', 'pptx']

# 支持矩阵：(from, to) : 后端类型
_CONVERT_MATRIX = {
    # Pandoc 支持
    ('md', 'pdf'): 'pandoc',
    ('markdown', 'pdf'): 'pandoc',
    ('md', 'html'): 'pandoc',
    ('markdown', 'html'): 'pandoc',
    ('html', 'pdf'): 'pandoc',
    ('html', 'md'): 'pandoc',
    ('html', 'markdown'): 'pandoc',
    ('md', 'txt'): 'pandoc',
    ('markdown', 'txt'): 'pandoc',
    ('md', 'pptx'): 'pandoc',
    ('markdown', 'pptx'): 'pandoc',
    ('pptx', 'pdf'): 'pandoc',
    ('pptx', 'md'): 'pandoc',
    ('pptx', 'markdown'): 'pandoc',
    ('pptx', 'html'): 'pandoc',
    # PDF → txt/html
    ('pdf', 'txt'): 'pdfminer',
    ('pdf', 'html'): 'pdf2htmlex',
}

def get_format(filename):
    ext = os.path.splitext(filename)[-1].lower().strip('.')
    if ext == 'md':
        return 'md'
    elif ext in ('markdown', 'mdown'):
        return 'markdown'
    elif ext == 'htm':
        return 'html'
    elif ext in SUPPORTED_FORMATS:
        return ext
    else:
        raise ValueError(f"不支持的文件扩展名: {filename}")

def convert(input_file, output_file, from_format=None, to_format=None):
    """
    通用格式转换接口
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param from_format: 输入格式（可省略，自动从文件名检测）
    :param to_format: 输出格式（必须指定）
    """
    if not to_format:
        raise ValueError("必须指定目标格式 to_format")
    if not from_format:
        from_format = get_format(input_file)
    to_format = to_format.lower()
    from_format = from_format.lower()
    if (from_format, to_format) not in _CONVERT_MATRIX:
        raise NotImplementedError(f"暂不支持从 {from_format} 到 {to_format} 的转换")
    backend = _CONVERT_MATRIX[(from_format, to_format)]
    if backend == 'pandoc':
        convert_with_pandoc(input_file, output_file, to_format)
    elif backend == 'pdfminer':
        pdf_to_txt(input_file, output_file)
    elif backend == 'pdf2htmlex':
        pdf_to_html(input_file, output_file)
    else:
        raise NotImplementedError(f"未知后端: {backend}")