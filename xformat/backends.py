import pypandoc
import subprocess

def convert_with_pandoc(input_path, output_path, to_format):
    extra_args = []
    if to_format == 'pdf':
        extra_args = ['--pdf-engine=xelatex']
    pypandoc.convert_file(input_path, to_format, outputfile=output_path, extra_args=extra_args)

def pdf_to_txt(input_path, output_path):
    # 依赖 pdfminer.six
    subprocess.run(['pdf2txt.py', input_path, '-o', output_path], check=True)

def pdf_to_html(input_path, output_path):
    # 依赖 pdf2htmlex
    subprocess.run(['pdf2htmlEX', input_path, output_path], check=True)