import subprocess

def pdf_to_txt(input_path, output_path):
    subprocess.run(['pdf2txt.py', input_path, '-o', output_path], check=True)