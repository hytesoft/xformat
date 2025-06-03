from rich.console import Console
from rich.table import Table
import sys
from time import sleep

# 导入各节点函数（假设和 multimodal_pipeline_dag.py 在同一目录）
from multimodal_pipeline_dag import (
    scan_file_task, classify_file_task, segment_text_task, segment_pdf_task,
    segment_word_task, segment_ppt_task, segment_excel_task, segment_audio_task,
    segment_video_task, audio_asr_and_refine_task, video_multimodal_understanding_task
)

NODES = [
    ("scan_file", "文件扫描", scan_file_task),
    ("classify_file", "类型判断", classify_file_task),
    ("segment_text", "文本切分", segment_text_task),
    ("segment_pdf", "PDF切分", segment_pdf_task),
    ("segment_word", "Word切分", segment_word_task),
    ("segment_ppt", "PPT切分", segment_ppt_task),
    ("segment_excel", "Excel切分", segment_excel_task),
    ("segment_audio", "音频切分", segment_audio_task),
    ("audio_asr_and_refine", "音频ASR+润色", audio_asr_and_refine_task),
    ("segment_video", "视频切分", segment_video_task),
    ("video_multimodal_understanding", "视频多模态理解", video_multimodal_understanding_task),
]

console = Console()

def show_status(status_dict):
    table = Table(title="多模态处理流程进度")
    table.add_column("节点", style="cyan")
    table.add_column("状态", style="magenta")
    for node, label, _ in NODES:
        status = status_dict.get(node, "等待")
        if status == "完成":
            status_str = "[green]✓ 完成[/green]"
        elif status == "运行中":
            status_str = "[yellow]● 运行中[/yellow]"
        elif status == "出错":
            status_str = "[red]✗ 出错[/red]"
        else:
            status_str = "…"
        table.add_row(label, status_str)
    console.clear()
    console.print(table)

def run_pipeline(input_file):
    # 构造 context，兼容原有节点
    class DummyDagRun:
        def __init__(self, input_file):
            self.conf = {"input_file": input_file}
    context = {"dag_run": DummyDagRun(input_file)}
    status = {node: "等待" for node, _, _ in NODES}
    # 动态分支控制
    node_skip = set()
    for idx, (node, label, func) in enumerate(NODES):
        # 动态分支逻辑
        if node == "segment_text" or node == "segment_pdf" or node == "segment_word" or node == "segment_ppt" or node == "segment_excel" or node == "segment_audio" or node == "segment_video":
            # 只执行 classify_file_task 返回的分支
            if "classify_file_result" in context:
                if context["classify_file_result"] != node:
                    node_skip.add(node)
                    continue
        if node == "audio_asr_and_refine" and "segment_audio" in node_skip:
            node_skip.add(node)
            continue
        if node == "video_multimodal_understanding" and "segment_video" in node_skip:
            node_skip.add(node)
            continue
        status[node] = "运行中"
        show_status(status)
        try:
            if node == "classify_file":
                result = func(**context)
                context["classify_file_result"] = result
            else:
                func(**context)
            status[node] = "完成"
        except Exception as e:
            status[node] = "出错"
            show_status(status)
            console.print(f"[red]节点 {label} 出错: {e}[/red]")
            break
        show_status(status)
    print("流程执行完毕！")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python pipeline.py <input_file>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
