# 多模态数据处理系统模块划分与详细要求

## 目录结构建议

```
multimodal_pipeline/
│
├── config/                  # 配置文件（全局参数、模型路径等）
│
├── io_utils/                # 输入输出与文件管理
│   └── file_scanner.py      # 批量扫描与类型检测
│   └── file_writer.py       # 结构化数据写入输出
│
├── segmenters/              # 文件切分模块
│   └── word_segmenter.py    # Word文档切分
│   └── ppt_segmenter.py     # PPT切分
│   └── pdf_segmenter.py     # PDF分页
│   └── markdown_segmenter.py# Markdown分段
│   └── excel_segmenter.py   # Excel分sheet/N行切
│   └── audio_segmenter.py   # 音频切片
│   └── video_segmenter.py   # 视频镜头/定长切分
│   └── image_segmenter.py   # 图片处理
│
├── recognizers/             # 识别模块
│   └── ocr_recognizer.py    # OCR识别图片文本
│   └── asr_recognizer.py    # ASR识别音频/视频语音
│   └── vision_captioner.py  # 视频画面理解/字幕生成
│
├── polisher/                # 文本整理与润色
│   └── llm_polisher.py      # 大模型润色/纠错/摘要
│
├── merger/                  # 合成与去重
│   └── content_merger.py    # 合成三类文本，去重归一
│
├── vectorizer/              # 向量化与入库
│   └── embedder.py          # 文本向量化
│   └── db_writer.py         # 向量数据库写入
│
├── postprocess/             # 质量提升与后处理
│   └── deduplicator.py      # 语义去重
│   └── confidence_filter.py # 置信度筛选&人工抽检接口
│   └── metadata_enricher.py # 元数据增强
│
├── main.py                  # 主控流程脚本
└── requirements.txt         # 依赖库
```

---

## 各模块详细要求

---

### 1. config/

- **功能**：集中管理参数、模型路径、API密钥等配置
- **输入**：无
- **输出**：配置对象
- **实现要点**：
  - 支持yaml/json/ini格式
  - 便于主流程和各模块调用

---

### 2. io_utils/

#### file_scanner.py

- **功能**：递归扫描输入目录，识别文件类型
- **输入**：输入目录路径
- **输出**：文件列表，每个文件带类型标签
- **要点**：支持python-magic和mimetypes双保险检测

#### file_writer.py

- **功能**：结构化数据输出（如jsonl/csv）
- **输入**：分段/识别/合成后的字典列表
- **输出**：结构化文件
- **要点**：统一输出格式，异常处理

---

### 3. segmenters/

每个文件类型一个segmenter，要求：

- **功能**：对指定文件类型切分为标准“段”
- **输入**：单个文件路径
- **输出**：段落列表，每段含 text_path, audio_path, image_paths
- **要点**：
  - 粒度可配置（如分页/定长/镜头/行数）
  - 切分结果写入缓存目录
  - 失败有日志

---

### 4. recognizers/

#### ocr_recognizer.py

- **功能**：批量识别图片文本
- **输入**：图片路径列表
- **输出**：对应文本内容/置信度
- **要点**：可切换OCR服务（PaddleOCR/百度/腾讯/大模型等）

#### asr_recognizer.py

- **功能**：批量识别音频
- **输入**：音频文件路径
- **输出**：转写文本/置信度
- **要点**：可选ASR模型，支持长音频切片拼接

#### vision_captioner.py

- **功能**：视频关键帧/片段画面描述生成
- **输入**：图片/视频片段路径
- **输出**：字幕/描述文本
- **要点**：支持大模型/字幕文件/自定义API

---

### 5. polisher/llm_polisher.py

- **功能**：用大模型对原始识别文本润色、纠错、分句、摘要
- **输入**：原始文本
- **输出**：润色后文本
- **要点**：可配置prompt，支持批量调用API或本地大模型

---

### 6. merger/content_merger.py

- **功能**：将文本、图片OCR、音频ASR、视频caption等多来源文本合成一段最终内容
- **输入**：各来源文本
- **输出**：合成文本
- **要点**：可配置合成顺序、分隔符、去重规则

---

### 7. vectorizer/

#### embedder.py

- **功能**：文本向量化
- **输入**：文本内容
- **输出**：向量
- **要点**：支持多种embedding模型

#### db_writer.py

- **功能**：写入向量数据库
- **输入**：向量、原文、meta
- **输出**：成功/失败
- **要点**：封装Milvus/Qdrant/Pinecone等常用数据库SDK

---

### 8. postprocess/

#### deduplicator.py

- **功能**：语义去重
- **输入**：文本列表
- **输出**：去重后文本列表

#### confidence_filter.py

- **功能**：根据置信度筛选段落，接口支持人工抽检
- **输入**：识别结果及分数
- **输出**：高质量片段/需复审列表

#### metadata_enricher.py

- **功能**：补充/规范化元数据
- **输入**：初始元数据
- **输出**：增强后的元数据

---

### 9. main.py

- **功能**：主流程调度，串联所有模块
- **输入**：输入目录、输出目录、配置参数
- **输出**：最终segments.json/向量入库报告
- **要点**：日志清晰、失败回滚、可断点续跑

---

## 实现范式要求

- 每个模块均以标准Python包结构实现，类/函数清晰、接口文档齐全
- 输入输出类型和字段严格对齐文档
- 主流程和每个模块均需有单元测试和异常处理
- 依赖通过requirements.txt统一管理
- 每个核心接口和数据结构配有示例说明

---

## 填鸭式开发建议

- 实现时AI同事只需关注本模块输入输出和实现要点，不必关心其它模块内部细节
- 所有数据流通过标准dict/list/json接口传递
- 配置和参数通过config统一管理
- 提供典型输入输出样例，便于自动化测试和主流程集成

---

如需具体模块实现样板，@Copilot 直接配合本结构开发即可。