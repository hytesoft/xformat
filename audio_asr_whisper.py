import os
import json
import yaml
from pathlib import Path

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def is_audio_file(path):
    if not isinstance(path, str):
        return False
    return path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"))

def asr_whisper(audio_path, model=None):
    import whisper
    if model is None:
        model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='zh')
    return result['text'].strip()

def process_json(json_path, model=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    changed = False
    for seg in segments:
        audio_val = seg.get('audio')
        if is_audio_file(audio_val) and os.path.exists(audio_val):
            try:
                text = asr_whisper(audio_val, model)
                seg['audio'] = text
                changed = True
                print(f"[ASR] {audio_val} -> {text[:30]}...")
            except Exception as e:
                print(f"[ASR ERROR] {audio_val}: {e}")
    if changed:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 已覆盖写回: {json_path}")
    else:
        print(f"[SKIP] 未发现可处理音频: {json_path}")

def main():
    config = load_config()
    public_dir = config.get('public_dir', './public_data')
    # 批量处理 public_dir 下所有 out_*.json 文件
    for file in os.listdir(public_dir):
        if file.endswith('.json') and file.startswith('out_'):
            json_path = os.path.join(public_dir, file)
            print(f"[PROCESS] {json_path}")
            process_json(json_path)

if __name__ == '__main__':
    main()
