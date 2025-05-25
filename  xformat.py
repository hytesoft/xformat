import os
import tempfile
import pypandoc

def convert(input_data, to_format):
    if isinstance(input_data, str) and os.path.isfile(input_data):
        in_path = input_data
        cleanup_in = False
    else:
        tmp_in = tempfile.NamedTemporaryFile(delete=False)
        tmp_in.write(input_data)
        tmp_in.close()
        in_path = tmp_in.name
        cleanup_in = True

    out_path = tempfile.mktemp(suffix='.' + to_format)
    pypandoc.convert_file(in_path, to_format, outputfile=out_path)
    with open(out_path, 'rb') as f:
        result = f.read()
    if cleanup_in:
        os.remove(in_path)
    os.remove(out_path)
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="xformat - 通用文档格式转换工具")
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--to', '-t', required=True, help='目标格式，例如 pdf, html, md, pptx, txt')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    args = parser.parse_args()
    with open(args.input, 'rb') as fin:
        content = fin.read()
    result = convert(content, args.to)
    with open(args.output, 'wb') as fout:
        fout.write(result)
    print(f"转换完成: {args.input} -> {args.output}")

   