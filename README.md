# xformat

**xformat** 是一个精致、专业的多格式文档转换工具，支持 Markdown、PDF、HTML、TXT、PPTX 等格式的高质量互转。适合个人办公、开发自动化、文档批量处理等多种场景。

## 特性

- 支持多种主流文档格式互转
- 统一参数接口，自动识别输入格式
- 支持 CLI 命令行、Python 包调用
- 自动选择最佳后端，转换质量高
- 易扩展，支持新格式和新引擎
- 错误提示友好，支持矩阵明确

## 安装

确保环境已安装 Pandoc、pdfminer.six、pdf2htmlex（可选）。

```bash
pip install pypandoc pdfminer.six
# 建议 apt install pandoc pdf2htmlex
```

## 快速开始

### 命令行用法

```bash
python cli.py --input example.md --output example.pdf --to-format pdf
```

### Python 包用法

```python
from xformat.core import convert

convert("example.md", "example.pdf", None, "pdf")
```

## 支持格式与支持矩阵

| 源格式      | 目标格式    | 支持情况 | 备注           |
| ----------- | ----------- | ------ | -------------- |
| md/markdown | pdf         | ✔️      | Pandoc         |
| md/markdown | html        | ✔️      | Pandoc         |
| md/markdown | txt         | ✔️      | Pandoc         |
| md/markdown | pptx        | ✔️      | Pandoc         |
| html        | pdf         | ✔️      | Pandoc         |
| html        | md/markdown | ✔️      | Pandoc         |
| pptx        | pdf         | ✔️      | Pandoc         |
| pptx        | md/markdown | ✔️      | Pandoc         |
| pptx        | html        | ✔️      | Pandoc         |
| pdf         | txt         | ✔️      | pdfminer.six   |
| pdf         | html        | ✔️      | pdf2htmlex     |
| 其它组合    |             | ❌      | 暂不支持       |

## TODO

- 支持批量目录、归档包转换
- REST API/Web UI
- 转换质量测试和对比

## 贡献

欢迎 issue、PR 和建议！

## License

MIT