import subprocess

def pdf_to_html(input_path, output_path):
    subprocess.run(['pdf2htmlEX', input_path, output_path], check=True)