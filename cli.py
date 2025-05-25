import argparse
from xformat.core import convert, SUPPORTED_FORMATS

def main():
    parser = argparse.ArgumentParser(description="xformat - 通用文档格式智能转换工具")
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    parser.add_argument('--from-format', help='输入格式(如md, pdf, html)，留空自动识别')
    parser.add_argument('--to-format', '-t', required=True, help=f'目标格式，支持：{SUPPORTED_FORMATS}')
    args = parser.parse_args()

    convert(args.input, args.output, args.from_format, args.to_format)
    print(f"转换完成: {args.input} → {args.output}")

if __name__ == "__main__":
    main()