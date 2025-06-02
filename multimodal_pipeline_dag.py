from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import sys
import yaml
import mimetypes

# 保证本地模块可被导入
sys.path.append('.')

def scan_file_task(**context):
    # 只处理单个文件，必须从 dag_run.conf 读取 input_file
    dag_run = context.get('dag_run')
    input_file = None
    if dag_run and hasattr(dag_run, 'conf') and dag_run.conf:
        input_file = dag_run.conf.get('input_file')
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf. No file to process.")
    # 输出路径
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    output_json = config.get("output_json", "./cache/file_list.json")
    from io_utils.file_scanner import scan_files
    scan_files(input_file, output_json)

def classify_file_task(**context):
    # 获取 input_file 路径
    dag_run = context.get('dag_run')
    input_file = None
    if dag_run and hasattr(dag_run, 'conf') and dag_run.conf:
        input_file = dag_run.conf.get('input_file')
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf. No file to process.")
    # 用 mimetypes 判断类型
    filetype, _ = mimetypes.guess_type(input_file)
    if not filetype:
        # 针对常见 office 文件补充判断
        if input_file.lower().endswith('.docx'):
            return 'segment_word'
        if input_file.lower().endswith('.pptx'):
            return 'segment_ppt'
        if input_file.lower().endswith('.xlsx'):
            return 'segment_excel'
        raise ValueError(f"无法识别文件类型: {input_file}")
    # 文本类类型判断
    if filetype.startswith('text/plain') or filetype.startswith('text/markdown') or filetype.startswith('text/html'):
        return 'segment_text'
    if filetype.startswith('application/pdf'):
        return 'segment_pdf'
    if filetype in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return 'segment_word'
    if filetype in ['application/vnd.openxmlformats-officedocument.presentationml.presentation']:
        return 'segment_ppt'
    if filetype in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return 'segment_excel'
    elif filetype.startswith('audio/'):
        return 'segment_audio'
    elif filetype.startswith('image/'):
        return 'segment_image'
    elif filetype.startswith('video/'):
        return 'segment_video'
    else:
        return 'segment_other'

def segment_text_task(**context):
    import os, json, re
    from bs4 import BeautifulSoup
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    # 判断文件类型
    import mimetypes
    filetype, _ = mimetypes.guess_type(input_file)
    output_path = "/mnt/share/out_text.json"  # 可根据实际挂载目录调整
    segments = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # html: 去标签、去样式、去脚本
        if filetype and filetype.startswith('text/html'):
            soup = BeautifulSoup(line, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        # markdown: 去除常见语法符号
        elif filetype and filetype.startswith('text/markdown'):
            text = re.sub(r'[#*`>\-\[\]()!_~]', '', line)
            text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)  # 去图片
            text = re.sub(r'\[.*?\]\(.*?\)', '', text)    # 去链接
            text = text.strip()
        # 纯文本：去除特殊控制字符
        else:
            text = re.sub(r'[\x00-\x1f\x7f]', '', line)
        if text:
            segments.append({
                "index": idx,
                "text": text,
                "audio": None,
                "image": []
            })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented {len(segments)} lines to {output_path}")

def segment_pdf_task(**context):
    import os, json
    import fitz  # PyMuPDF
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    output_path = "/mnt/share/out_pdf.json"  # 可根据实际挂载目录调整
    doc = fitz.open(input_file)
    segments = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"/mnt/share/pdf_page{page_num+1}_img{img_index+1}.{image_ext}"
            with open(image_filename, "wb") as img_f:
                img_f.write(image_bytes)
            images.append(image_filename)
        segments.append({
            "index": page_num,
            "text": text,
            "audio": None,
            "image": images
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented {len(segments)} pages to {output_path}")

def segment_word_task(**context):
    import os, json
    from docx import Document
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    output_path = "/mnt/share/out_word.json"
    doc = Document(input_file)
    segments = []
    img_idx = 0
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        images = []
        # 提取图片（docx图片一般在 runs 里）
        for run in para.runs:
            if 'graphic' in run._element.xml:
                # 这里只能做简单标记，复杂图片提取需用更底层包
                images.append(f"word_img_{img_idx+1}")
                img_idx += 1
        if text or images:
            segments.append({
                "index": idx,
                "text": text,
                "audio": None,
                "image": images
            })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented {len(segments)} paragraphs to {output_path}")

def segment_ppt_task(**context):
    import os, json
    from pptx import Presentation
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    output_path = "/mnt/share/out_ppt.json"
    prs = Presentation(input_file)
    segments = []
    img_idx = 0
    for idx, slide in enumerate(prs.slides):
        texts = []
        images = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text.strip())
            if shape.shape_type == 13:  # PICTURE
                images.append(f"ppt_img_{img_idx+1}")
                img_idx += 1
        text = "\n".join([t for t in texts if t])
        segments.append({
            "index": idx,
            "text": text,
            "audio": None,
            "image": images
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented {len(segments)} slides to {output_path}")

def segment_excel_task(**context):
    import os, json
    import pandas as pd
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    output_path = "/mnt/share/out_excel.json"
    xls = pd.ExcelFile(input_file)
    segments = []
    idx = 0
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        for row in df.itertuples(index=False):
            text = ' '.join([str(cell) for cell in row if pd.notnull(cell)])
            if text:
                segments.append({
                    "index": idx,
                    "text": text,
                    "audio": None,
                    "image": []
                })
                idx += 1
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented {len(segments)} rows to {output_path}")

def segment_audio_task(**context):
    import os, json, math
    import yaml
    from pydub import AudioSegment
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    segment_seconds = config.get('audio_segment_seconds', 30)
    output_dir = '/mnt/share/'
    output_json = os.path.join(output_dir, 'out_audio.json')
    audio = AudioSegment.from_file(input_file)
    duration_sec = len(audio) / 1000
    num_segments = math.ceil(duration_sec / segment_seconds)
    segments = []
    for idx in range(num_segments):
        start_ms = idx * segment_seconds * 1000
        end_ms = min((idx + 1) * segment_seconds * 1000, len(audio))
        segment_audio = audio[start_ms:end_ms]
        seg_path = os.path.join(output_dir, f'audio_seg_{idx+1}.wav')
        segment_audio.export(seg_path, format='wav')
        segments.append({
            "index": idx,
            "text": None,
            "audio": seg_path,
            "image": []
        })
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segmented audio into {num_segments} segments, config seconds={segment_seconds}, output: {output_json}")

def segment_video_task(**context):
    import os, json
    import yaml
    import cv2
    import tempfile
    import subprocess
    dag_run = context.get('dag_run')
    input_file = dag_run.conf.get('input_file') if dag_run and dag_run.conf else None
    if not input_file:
        raise ValueError("input_file must be provided via dag_run.conf.")
    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = '/mnt/share/'
    output_json = os.path.join(output_dir, 'out_video.json')
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_file}")
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    segments = []
    idx = 0
    last_hist = None
    frame_id = 0
    keyframe_gap = config.get('video_keyframe_min_gap', 10)
    last_keyframe = -keyframe_gap
    last_keyframe_time = 0.0
    video_basename = os.path.splitext(os.path.basename(input_file))[0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 计算当前帧的直方图
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        is_keyframe = False
        if last_hist is not None:
            diff = cv2.compareHist(hist, last_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > 0.5 and (frame_id - last_keyframe) >= keyframe_gap:
                is_keyframe = True
        else:
            is_keyframe = True
        if is_keyframe:
            img_path = os.path.join(output_dir, f'{video_basename}_keyframe_{idx+1}.jpg')
            cv2.imwrite(img_path, frame)
            # 计算关键帧时间（秒）
            keyframe_time = frame_id / fps if fps else 0.0
            # 音频切片（取上一个关键帧到当前关键帧的音频）
            if idx == 0:
                audio_start = 0.0
            else:
                audio_start = last_keyframe_time
            audio_end = keyframe_time
            audio_path = os.path.join(output_dir, f'{video_basename}_audio_{idx+1}.wav')
            # 用 ffmpeg 提取音频片段
            if audio_end > audio_start + 0.1:
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-ss', str(audio_start), '-to', str(audio_end),
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    audio_out = audio_path
                except Exception:
                    audio_out = None
            else:
                audio_out = None
            segments.append({
                "index": idx,
                "text": None,
                "audio": audio_out,
                "image": [img_path]
            })
            idx += 1
            last_keyframe = frame_id
            last_keyframe_time = keyframe_time
        last_hist = hist
        frame_id += 1
    cap.release()
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(segments)} keyframes+audio to {output_json}")

def audio_asr_and_refine_task(**context):
    import os, json
    import yaml
    import tempfile
    dag_run = context.get('dag_run')
    # 读取分段音频json
    audio_json = '/mnt/share/out_audio.json'
    if not os.path.exists(audio_json):
        raise FileNotFoundError(f"{audio_json} not found")
    with open(audio_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    # 读取配置，获取ASR和LLM参数
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 1. 语音识别（ASR）
    # 这里用 openai-whisper 命令行或 huggingface transformers/pipeline 方案，示例用 whisper
    refined_segments = []
    for seg in segments:
        audio_path = seg['audio']
        if not audio_path or not os.path.exists(audio_path):
            seg['text'] = None
            refined_segments.append(seg)
            continue
        # Whisper ASR
        try:
            import whisper
            model = whisper.load_model(config.get('asr_model', 'base'))
            result = model.transcribe(audio_path, language=config.get('asr_language', 'zh'))
            asr_text = result['text'].strip()
        except Exception as e:
            asr_text = ''
        # 2. LLM润色（假设有本地API或openai/gpt等，示例用伪代码）
        if asr_text:
            try:
                # 这里可替换为实际 LLM API 调用
                # refined_text = call_llm_refine_api(asr_text)
                refined_text = asr_text  # 占位，实际应调用 LLM
            except Exception:
                refined_text = asr_text
        else:
            refined_text = None
        seg['text'] = refined_text
        refined_segments.append(seg)
    # 输出新json
    output_json = '/mnt/share/out_audio_refined.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(refined_segments, f, ensure_ascii=False, indent=2)
    print(f"ASR+LLM refined audio segments written to {output_json}")

def video_multimodal_understanding_task(**context):
    import os, json, yaml
    dag_run = context.get('dag_run')
    # 读取视频分段json
    video_json = '/mnt/share/out_video.json'
    if not os.path.exists(video_json):
        raise FileNotFoundError(f"{video_json} not found")
    with open(video_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    # 读取配置，获取多模态模型参数
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    multimodal_api = config.get('multimodal_api', 'http://localhost:8000/multimodal_infer')
    multimodal_model = config.get('multimodal_model', 'qwen-vl')
    # 可选：先对音频做ASR
    try:
        import whisper
        asr_model = whisper.load_model(config.get('asr_model', 'base'))
    except Exception:
        asr_model = None
    refined_segments = []
    for seg in segments:
        img_path = seg['image'][0] if seg['image'] else None
        audio_path = seg['audio']
        asr_text = None
        if audio_path and asr_model and os.path.exists(audio_path):
            try:
                result = asr_model.transcribe(audio_path, language=config.get('asr_language', 'zh'))
                asr_text = result['text'].strip()
            except Exception:
                asr_text = None
        # 调用多模态大模型API（伪代码，需替换为实际API调用）
        multimodal_result = None
        if img_path and os.path.exists(img_path):
            try:
                # 你可以替换为实际的多模态API调用，如requests.post等
                # 示例：
                # resp = requests.post(multimodal_api, json={
                #     'image': open(img_path, 'rb').read(),
                #     'text': asr_text,
                #     'audio': open(audio_path, 'rb').read() if audio_path else None,
                #     'model': multimodal_model
                # })
                # multimodal_result = resp.json()
                multimodal_result = {
                    'summary': f"[DEMO] 图像+音频内容摘要 (index={seg['index']})",
                    'tags': ["demo_tag1", "demo_tag2"],
                    'asr_text': asr_text,
                    'model': multimodal_model
                }
            except Exception as e:
                multimodal_result = {'error': str(e)}
        seg['multimodal_result'] = multimodal_result
        # 可选：将asr_text补充到text字段
        seg['text'] = asr_text
        refined_segments.append(seg)
    # 输出新json
    output_json = '/mnt/share/out_video_multimodal.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(refined_segments, f, ensure_ascii=False, indent=2)
    print(f"Video multimodal understanding results written to {output_json}")

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id="multimodal_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    description="多模态批量数据切分与识别主流程"
) as dag:

    scan_file = PythonOperator(
        task_id="scan_file",
        python_callable=scan_file_task
    )

    classify_file = BranchPythonOperator(
        task_id="classify_file",
        python_callable=classify_file_task
    )

    segment_text = PythonOperator(
        task_id="segment_text",
        python_callable=segment_text_task
    )

    segment_pdf = PythonOperator(
        task_id="segment_pdf",
        python_callable=segment_pdf_task
    )

    segment_word = PythonOperator(
        task_id="segment_word",
        python_callable=segment_word_task
    )
    segment_ppt = PythonOperator(
        task_id="segment_ppt",
        python_callable=segment_ppt_task
    )
    segment_excel = PythonOperator(
        task_id="segment_excel",
        python_callable=segment_excel_task
    )

    segment_audio = PythonOperator(
        task_id="segment_audio",
        python_callable=segment_audio_task
    )

    segment_video = PythonOperator(
        task_id="segment_video",
        python_callable=segment_video_task
    )

    audio_asr_and_refine = PythonOperator(
        task_id="audio_asr_and_refine",
        python_callable=audio_asr_and_refine_task
    )

    video_multimodal_understanding = PythonOperator(
        task_id="video_multimodal_understanding",
        python_callable=video_multimodal_understanding_task
    )

    scan_file >> classify_file
    classify_file >> segment_text
    classify_file >> segment_pdf
    classify_file >> segment_word
    classify_file >> segment_ppt
    classify_file >> segment_excel
    classify_file >> segment_audio
    classify_file >> segment_video
    segment_audio >> audio_asr_and_refine
    segment_video >> video_multimodal_understanding
