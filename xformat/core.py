import os
from .backends import convert_with_pandoc, pdf_to_txt, pdf_to_html
import requests
from bs4 import BeautifulSoup
import time

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

def convert_text(input_text, from_format, to_format):
    """
    文本内容直接转换（仅支持文本格式: md, markdown, html, txt）
    :param input_text: 输入内容（字符串）
    :param from_format: 输入格式
    :param to_format: 输出格式
    :return: 输出内容（字符串）
    """
    from_format = from_format.lower()
    to_format = to_format.lower()
    if (from_format, to_format) not in _CONVERT_MATRIX:
        raise NotImplementedError(f"暂不支持从 {from_format} 到 {to_format} 的转换")
    backend = _CONVERT_MATRIX[(from_format, to_format)]
    if backend == 'pandoc':
        # 需要用临时文件中转
        import tempfile
        with tempfile.NamedTemporaryFile('w+', delete=False, suffix=f'.{from_format}') as fin:
            fin.write(input_text)
            fin.flush()
            with tempfile.NamedTemporaryFile('r', delete=False, suffix=f'.{to_format}') as fout:
                convert_with_pandoc(fin.name, fout.name, to_format)
                fout.seek(0)
                return fout.read()
    else:
        raise NotImplementedError(f"convert_text 仅支持 pandoc 文本格式转换")

def convert_url(url, to_format, proxy=None):
    """
    从URL抓取网页，去除body、br、div、span、css等元素和样式，转换为指定格式字符串
    支持代理和重试，返回 flag 标识
    :param url: 网页URL
    :param to_format: 目标格式（如 txt, md, html）
    :param proxy: 代理地址（可选）
    :return: (flag, result)
    flag: 0-直连成功，1-代理成功，2-代理失败
    """
    session = requests.Session()
    timeout = 10
    # 1. 直连尝试
    for i in range(3):
        try:
            resp = session.get(url, timeout=timeout)
            resp.encoding = resp.apparent_encoding
            html = resp.text
            flag = 0
            break
        except Exception:
            if i == 2:
                html = None
    else:
        html = None
    # 2. 代理尝试
    if html is None and proxy:
        proxies = {"http": proxy, "https": proxy}
        for i in range(3):
            try:
                resp = session.get(url, timeout=timeout, proxies=proxies)
                resp.encoding = resp.apparent_encoding
                html = resp.text
                flag = 1
                break
            except Exception:
                if i == 2:
                    html = None
        else:
            html = None
    # 3. 失败
    if html is None:
        flag = 2
        return flag, ""
    # 4. 清洗 html
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['body', 'br', 'div', 'span', 'style', 'script', 'link']):
        tag.decompose()
    for tag in soup.find_all(True):
        if 'style' in tag.attrs:
            del tag.attrs['style']
    clean_html = str(soup)
    # 5. 转换格式
    if to_format in ('txt', 'md', 'markdown'):
        result = convert_text(clean_html, 'html', to_format)
    elif to_format == 'html':
        result = clean_html
    else:
        result = ""
    return flag, result