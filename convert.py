import pypandoc
import subprocess

def convert(input_file, output_file, to_format):
    # Pandoc 直接支持的格式
    supported = ['md', 'markdown', 'html', 'txt', 'docx', 'pptx', 'pdf']
    ext = input_file.split('.')[-1]
    if to_format in supported and ext in supported:
        extra_args = []
        if to_format == 'pdf':
            extra_args = ['--pdf-engine=xelatex']
        pypandoc.convert_file(input_file, to_format, outputfile=output_file, extra_args=extra_args)
    elif ext == 'pdf' and to_format == 'html':
        # 用 pdf2htmlEX
        subprocess.run(['pdf2htmlEX', input_file, output_file])
    elif ext == 'pdf' and to_format == 'txt':
        # 用 pdfminer.six
        subprocess.run(['pdf2txt.py', input_file, '-o', output_file])
    else:
        raise NotImplementedError(f"Cannot convert {ext} to {to_format} automatically.")

if __name__ == "__main__":
    # 示例: Markdown转PDF
    convert('test.md', 'test.pdf', 'pdf')
    # 示例: PDF转HTML
    convert('test.pdf', 'test.html', 'html')