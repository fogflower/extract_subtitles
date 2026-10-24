#!/usr/bin/env python3
"""
视频字幕提取工具
- 从视频文件中提取嵌入式字幕（ffmpeg）
- 解析 VTT/SRT/ASS 字幕文件
- 输出 Markdown 格式（带时间戳 / 纯文本 / 段落模式）
"""

import subprocess
import sys
import os
import re
import json
import argparse
import tempfile
from pathlib import Path
from typing import Optional

# Windows 终端 UTF-8 编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# Whisper / OpenVINO 延迟导入
# ============================================================
_WHISPER_AVAILABLE = False
_WHISPER_ERROR = ""

try:
    import librosa
    import openvino_genai as ov_genai
    _WHISPER_AVAILABLE = True
except ImportError as e:
    _WHISPER_ERROR = str(e)


# ============================================================
# 1. 字幕流探测
# ============================================================

def probe_subtitle_streams(video_path: str) -> list[dict]:
    """用 ffprobe 探测视频中的字幕流"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-select_streams", "s",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[warn] ffprobe failed: {result.stderr}", file=sys.stderr)
            return []
        data = json.loads(result.stdout)
        return data.get("streams", [])
    except Exception as e:
        print(f"[warn] ffprobe error: {e}", file=sys.stderr)
        return []


# ============================================================
# 2. VTT / SRT 解析
# ============================================================

def parse_timestamp_vtt(ts: str) -> float:
    """VTT 时间戳 → 秒（支持 HH:MM:SS.mmm）"""
    # 去掉可能的额外毫秒分隔符 (.) 之后的内容，如 "00:01:23.456 --> ..."
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def format_timestamp(seconds: float) -> str:
    """秒 → HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clean_vtt_tags(text: str) -> str:
    """清洗 VTT 标签 (如 <c>, <v>, <00:00:01.000>, 样式标签)"""
    # 去掉时间戳标签 <00:00:01.000>
    text = re.sub(r"<\d{2}:\d{2}:\d{2}[\.:]\d{3}>", "", text)
    # 去掉其他 XML 风格标签（保留内容）
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    # 去掉 VTT 特有的 ruby / lang 等
    text = re.sub(r"</?c[\.\w]*>", "", text)
    text = re.sub(r"</?v[^>]*>", "", text)
    text = re.sub(r"</?ruby[^>]*>", "", text)
    text = re.sub(r"</?rt[^>]*>", "", text)
    text = re.sub(r"</?lang[^>]*>", "", text)
    # 解码 HTML 实体
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'")
    text = text.replace("&nbsp;", " ")
    return text.strip()


def parse_vtt(filepath: str) -> list[dict]:
    """解析 VTT 字幕文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 去除 WEBVTT 头部及元数据
    # 找到第一个空行后的内容
    lines = content.split("\n")
    blocks = []
    current_block: list[str] = []
    in_header = True

    for line in lines:
        if in_header:
            if line.strip() == "":
                # 可能是头部结束
                # 检查后面是否还有 STYLE / NOTE 块
                continue
            if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if line.strip() and not re.match(r"^\d{2}:", line) and not line.strip().isdigit():
                # 可能是 NOTE / STYLE 块，跳过
                continue
            in_header = False

        stripped = line.strip()
        # 分离每个 cue block（空行分隔）
        if stripped == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(stripped)

    if current_block:
        blocks.append(current_block)

    subtitles: list[dict] = []
    for block in blocks:
        # 跳过纯序号 block（如 "1"）
        # 找到时间戳行和文本行
        timestamp_line_idx = -1
        for i, line in enumerate(block):
            if "-->" in line:
                timestamp_line_idx = i
                break

        if timestamp_line_idx < 0:
            continue

        ts_line = block[timestamp_line_idx]
        # 解析起止时间
        parts = ts_line.split("-->")
        if len(parts) < 2:
            continue
        start = parse_timestamp_vtt(parts[0])
        end = parse_timestamp_vtt(parts[1].split()[0])  # 去掉可能的对齐等设置

        # 文本行：时间戳之后的全部行
        text_lines = block[timestamp_line_idx + 1:]
        text = " ".join(text_lines)
        text = clean_vtt_tags(text)

        if text:
            subtitles.append({
                "start": start,
                "end": end,
                "text": text
            })

    return subtitles


def parse_srt(filepath: str) -> list[dict]:
    """解析 SRT 字幕文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # SRT 格式: 序号 \n HH:MM:SS,mmm --> HH:MM:SS,mmm \n 文本 \n\n
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: list[dict] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # 跳过序号行
        ts_match = re.search(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", block)
        if not ts_match:
            continue

        start = parse_timestamp_vtt(ts_match.group(1))
        end = parse_timestamp_vtt(ts_match.group(2))

        # 找到时间戳行之后的内容
        ts_line_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line_idx = i
                break
        text = " ".join(lines[ts_line_idx + 1:]) if ts_line_idx >= 0 else ""
        text = clean_vtt_tags(text)

        if text:
            subtitles.append({"start": start, "end": end, "text": text})

    return subtitles


def detect_and_parse(filepath: str) -> list[dict]:
    """自动检测格式并解析字幕文件"""
    ext = Path(filepath).suffix.lower()
    if ext in (".vtt",):
        return parse_vtt(filepath)
    elif ext in (".srt",):
        return parse_srt(filepath)
    elif ext in (".ass", ".ssa"):
        return parse_ass(filepath)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")


def parse_ass(filepath: str) -> list[dict]:
    """解析 ASS/SSA 字幕文件（基础支持）"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    subtitles: list[dict] = []
    in_events = False

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("[Events]"):
            in_events = True
            continue
        if in_events and line.startswith("["):
            in_events = False
            continue
        if not in_events:
            continue
        if line.startswith("Format:") or line.startswith("Dialogue:"):
            if line.startswith("Dialogue:"):
                # Dialogue: Marked,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    start = parse_timestamp_vtt(parts[1])
                    end = parse_timestamp_vtt(parts[2])
                    text = parts[9].strip()
                    # 清洗 ASS 标签 {\xxx}
                    text = re.sub(r"\{[^}]*\}", "", text)
                    text = text.replace("\\N", "\n").replace("\\n", "\n")
                    text = text.strip()
                    if text:
                        subtitles.append({"start": start, "end": end, "text": text})

    return subtitles


# ============================================================
# 3. 从视频中提取字幕
# ============================================================

def extract_embedded_subtitle(video_path: str, stream_index: int = 0,
                              output_path: Optional[str] = None) -> str:
    """用 ffmpeg 提取视频中的嵌入式字幕流"""
    if output_path is None:
        base = Path(video_path).stem
        output_path = f"{base}.extracted_{stream_index}.srt"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", f"0:s:{stream_index}",
        "-c:s", "srt",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {result.stderr}")

    return output_path


# ============================================================
# 3b. Whisper AI 转录（OpenVINO GenAI + Intel GPU）
# ============================================================

# HuggingFace 预转换 OpenVINO 模型映射
# 格式: (repo_id, 是否为 english-only)
_HF_OV_MODELS = {
    "tiny":      ("OpenVINO/whisper-tiny-fp16-ov",       False),
    "tiny.en":   ("OpenVINO/whisper-tiny.en-fp16-ov",    True),
    "base":      ("OpenVINO/whisper-base-fp16-ov",       False),
    "base.en":   ("OpenVINO/whisper-base.en-fp16-ov",    True),
    "small":     ("OpenVINO/whisper-small-fp16-ov",      False),
    "small.en":  ("OpenVINO/whisper-small.en-fp16-ov",   True),
    "medium":    ("OpenVINO/whisper-medium-fp16-ov",     False),
    "medium.en": ("OpenVINO/whisper-medium.en-fp16-ov",  True),
    "large-v3":  ("OpenVINO/whisper-large-v3-fp16-ov",   False),
}

# 单例缓存
_whisper_pipeline_cache: dict = {}
_whisper_model_dir_cache: dict = {}


def _download_whisper_model(model_name: str) -> str:
    """
    下载预转换的 OpenVINO IR 模型到本地缓存目录。
    返回模型目录路径。
    """
    cache_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper-ov")
    local_dir = os.path.join(cache_root, model_name)

    # 检查是否已下载（目录存在且有 .xml 文件）
    if os.path.isdir(local_dir) and any(f.endswith(".xml") for f in os.listdir(local_dir)):
        print(f"[*] 模型已缓存: {local_dir}")
        return local_dir

    # 查找 HuggingFace repo
    repo_id, _ = _HF_OV_MODELS.get(model_name, (None, False))
    if repo_id is None:
        raise ValueError(
            f"未知的模型名: {model_name}。"
            f"可用: {', '.join(_HF_OV_MODELS.keys())}"
        )

    print(f"[*] 从 HuggingFace 下载模型: {repo_id}")
    print(f"[*] 保存到: {local_dir}")
    print(f"[*] 下载大小约 150-500MB，首次使用需要等待...")

    # 设置镜像（中国大陆用户）
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        # 自动检测：阿里云镜像可用
        pass

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(os.path.dirname(local_dir), exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2,
        )

        print(f"[*] 模型下载完成: {local_dir}")
        return local_dir

    except ImportError:
        print("[error] 需要安装 huggingface_hub 来下载模型:", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  或手动下载模型到: {local_dir}", file=sys.stderr)
        print(f"  huggingface-cli download {repo_id} --local-dir {local_dir}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[error] 模型下载失败: {e}", file=sys.stderr)
        print(f"[*] 可以手动执行:", file=sys.stderr)
        print(f"  pip install huggingface_hub", file=sys.stderr)
        print(f"  huggingface-cli download {repo_id} --local-dir {local_dir}", file=sys.stderr)
        sys.exit(1)


def load_whisper_pipeline(model_name: str = "base",
                          device: str = "GPU") -> tuple:
    """
    加载 Whisper pipeline（自动下载模型 + 单例缓存）

    Returns:
        (WhisperPipeline, actual_device)
    """
    cache_key = f"{model_name}:{device}"
    if cache_key in _whisper_pipeline_cache:
        return _whisper_pipeline_cache[cache_key]

    if not _WHISPER_AVAILABLE:
        print("[error] Whisper 不可用，请安装依赖:", file=sys.stderr)
        print(f"  pip install openvino-genai librosa huggingface_hub", file=sys.stderr)
        print(f"  原始错误: {_WHISPER_ERROR}", file=sys.stderr)
        sys.exit(1)

    # 下载模型
    model_dir = _download_whisper_model(model_name)

    print(f"[*] 加载 Whisper 模型: {model_name} (device={device})")

    try:
        # 先尝试指定设备
        use_device = device
        try:
            pipe = ov_genai.WhisperPipeline(model_dir, use_device)
        except RuntimeError as e:
            if use_device == "GPU":
                print(f"[warn] GPU 初始化失败: {e}", file=sys.stderr)
                print(f"[warn] 自动降级到 CPU ...", file=sys.stderr)
                use_device = "CPU"
                pipe = ov_genai.WhisperPipeline(model_dir, use_device)
            else:
                raise

        _whisper_pipeline_cache[cache_key] = (pipe, use_device)
        print(f"[*] 模型加载完成 (实际设备: {use_device})")
        return _whisper_pipeline_cache[cache_key]
    except Exception as e:
        print(f"[error] Whisper 模型加载失败: {e}", file=sys.stderr)
        sys.exit(1)


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """从视频中提取音频为 16kHz mono WAV（Whisper 输入格式）"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    print(f"[*] 从视频提取音频: {video_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",    # 16kHz 采样率
        "-ac", "1",        # 单声道
        "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"音频提取失败: {result.stderr}")

    # 获取音频时长信息
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format=duration",
        output_path
    ]
    try:
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(probe.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        print(f"[*] 音频提取完成: {output_path} (时长: {duration/60:.1f} 分钟)")
    except Exception:
        print(f"[*] 音频提取完成: {output_path}")

    return output_path


def transcribe_whisper(audio_path: str,
                       model_name: str = "base",
                       device: str = "GPU",
                       language: Optional[str] = None) -> tuple[list[dict], dict]:
    """
    用 Whisper 转录音频，返回字幕列表和元信息

    Returns:
        (subtitles, meta): subtitles 为 [{start, end, text}]，
                          meta 包含 {model, device, detected_language, duration}
    """
    pipe, actual_device = load_whisper_pipeline(model_name, device)

    print(f"[*] 读取音频文件: {audio_path}")
    raw_speech, sample_rate = librosa.load(audio_path, sr=16000)
    duration = len(raw_speech) / sample_rate
    print(f"[*] 音频时长: {duration/60:.1f} 分钟, 开始转录...")

    # 构建 generation config
    config = pipe.get_generation_config()
    config.task = "transcribe"
    config.return_timestamps = True  # 关键：返回时间戳

    if language:
        config.language = f"<|{language}|>"
        print(f"[*] 指定语言: {language}")

    # 转录
    audio_list = raw_speech.tolist()
    result = pipe.generate(audio_list, config)

    # 解析结果
    subtitles: list[dict] = []
    detected_language = "unknown"

    # WhisperDecodedResults 有 .chunks (list of WhisperDecodedResultChunk)
    # 每个 chunk 有 .text, .start_ts (float seconds), .end_ts
    if hasattr(result, "chunks") and result.chunks:
        for chunk in result.chunks:
            text = chunk.text.strip() if hasattr(chunk, "text") and chunk.text else ""
            if text:
                start = chunk.start_ts if hasattr(chunk, "start_ts") else 0.0
                end = chunk.end_ts if hasattr(chunk, "end_ts") else 0.0
                subtitles.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })
    elif hasattr(result, "texts") and result.texts:
        # 有 texts 但没有 chunks → 按全文处理
        full_text = result.texts[0].strip()
        if full_text:
            subtitles.append({
                "start": 0.0,
                "end": duration,
                "text": full_text,
            })
    elif hasattr(result, "text"):
        full_text = result.text.strip()
        if full_text:
            subtitles.append({
                "start": 0.0,
                "end": duration,
                "text": full_text,
            })

    # 尝试获取检测到的语言
    if hasattr(result, "language"):
        detected_language = result.language

    meta = {
        "model": model_name,
        "device": actual_device,
        "detected_language": detected_language,
        "duration": duration,
    }

    print(f"[*] 转录完成: {len(subtitles)} 条字幕, 检测语言: {detected_language}")
    return subtitles, meta


# ============================================================
# 4. 输出 Markdown
# ============================================================

def subtitles_to_markdown(subtitles: list[dict], title: str = "",
                          mode: str = "timestamp",
                          meta: Optional[dict] = None) -> str:
    """
    将字幕列表转为 Markdown

    mode:
      - "timestamp": 每条字幕带时间戳
      - "paragraph": 按短停顿（>2秒）合并为段落
      - "plain": 纯文本（无时间戳）

    meta: Whisper 转录元信息 {model, device, detected_language, duration}
    """
    out: list[str] = []

    if title:
        out.append(f"# {title}\n")

    # 元信息行
    meta_parts = [f"共 {len(subtitles)} 条字幕"]
    if meta:
        meta_parts.append(f"Whisper 转录 ({meta.get('model','?')}, {meta.get('device','?')})")
        if meta.get("detected_language", "unknown") != "unknown":
            meta_parts.append(f"检测语言: {meta['detected_language']}")
        dur = meta.get("duration", 0)
        if dur:
            meta_parts.append(f"音频时长: {dur/60:.1f} 分钟")
    else:
        meta_parts.append("由 extract_subtitles.py 生成")

    out.append(f"> {' | '.join(meta_parts)}\n")

    if mode == "plain":
        out.append("\n".join([s["text"] for s in subtitles]))
        return "\n".join(out)

    if mode == "paragraph":
        # 按 >2 秒的间隔合并段落
        paragraphs: list[list[dict]] = []
        current_para: list[dict] = []
        for i, sub in enumerate(subtitles):
            if current_para:
                gap = sub["start"] - current_para[-1]["end"]
                if gap > 2.0:  # 超过 2 秒视为新段落
                    paragraphs.append(current_para)
                    current_para = []
            current_para.append(sub)
        if current_para:
            paragraphs.append(current_para)

        for para in paragraphs:
            start_ts = format_timestamp(para[0]["start"])
            text = " ".join([s["text"] for s in para])
            out.append(f"- **`{start_ts}`** {text}")
        return "\n".join(out)

    # 默认: timestamp 模式
    for sub in subtitles:
        ts = format_timestamp(sub["start"])
        out.append(f"- **`{ts}`** {sub['text']}")

    return "\n".join(out)


# ============================================================
# 4b. 输出 SRT（剪映可导入）
# ============================================================

def subtitles_to_srt(subtitles: list[dict], output_path: str) -> str:
    """将字幕列表转为 SRT 格式（UTF-8），可直接导入剪映/CapCut"""
    def fmt_srt_ts(seconds: float) -> str:
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:   # 四舍五入进位
            ms = 999
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, sub in enumerate(subtitles, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(sub['start'])} --> {fmt_srt_ts(sub['end'])}")
        lines.append(sub["text"])
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


# ============================================================
# 5. CLI 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频字幕提取工具 — 提取字幕并输出 Markdown（支持 Whisper AI 转录）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s video.mp4                          # 提取第一个字幕流 → md
  %(prog)s video.mp4 -s 1                     # 提取第二个字幕流
  %(prog)s video.mp4 --subtitle-file sub.vtt  # 使用外部字幕文件
  %(prog)s video.mp4 -m plain                 # 纯文本输出
  %(prog)s video.mp4 -m paragraph             # 段落模式
  %(prog)s video.mp4 -l                       # 列出字幕流信息
  %(prog)s sub.vtt                             # 直接解析字幕文件
  %(prog)s video.mp4 --whisper                # 无字幕时用 Whisper AI 转录（Intel GPU）
  %(prog)s video.mp4 --whisper --whisper-model small  # 使用 small 模型
  %(prog)s video.mp4 --whisper --whisper-language en   # 指定语言
        """
    )

    parser.add_argument("input", help="视频文件路径 或 字幕文件路径")
    parser.add_argument("-s", "--stream", type=int, default=0,
                        help="字幕流索引 (默认: 0)")
    parser.add_argument("--subtitle-file", dest="sub_file",
                        help="指定外部字幕文件 (VTT/SRT/ASS)")
    parser.add_argument("-o", "--output", help="输出 md 文件路径")
    parser.add_argument("-m", "--mode", choices=["timestamp", "paragraph", "plain"],
                        default="timestamp",
                        help="输出模式: timestamp(默认), paragraph, plain")
    parser.add_argument("-t", "--title", default="",
                        help="Markdown 标题")
    parser.add_argument("-l", "--list", action="store_true",
                        help="仅列出视频中的字幕流信息")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留提取的中间文件（包括音频 WAV）")
    parser.add_argument("--srt", action="store_true",
                        help="同时输出 SRT 格式字幕（UTF-8，可直接导入剪映）")

    # Whisper 相关参数
    parser.add_argument("--whisper", action="store_true",
                        help="启用 Whisper AI 转录（视频无嵌入式字幕时自动触发）")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper 模型大小 (默认: base; tiny最快, small更准)")
    parser.add_argument("--whisper-device", default="GPU",
                        choices=["GPU", "CPU"],
                        help="推理设备 (默认: GPU; CPU 回退)")
    parser.add_argument("--whisper-language", default=None,
                        help="指定转录语言代码 (如 en, zh)，不指定则自动检测")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[error] 文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # --- 列出字幕流 ---
    if args.list:
        ext = Path(input_path).suffix.lower()
        if ext in (".vtt", ".srt", ".ass", ".ssa"):
            print("这是一个字幕文件，不是视频文件")
        else:
            streams = probe_subtitle_streams(input_path)
            if not streams:
                print("未找到嵌入式字幕流")
                print("提示: 使用 --whisper 通过 AI 从音频转录字幕")
            else:
                for s in streams:
                    lang = s.get("tags", {}).get("language", "unknown")
                    title_tag = s.get("tags", {}).get("title", "")
                    print(f"  流 #{s['index']} | {s['codec_name']} | 语言: {lang}"
                          + (f" | 标题: {title_tag}" if title_tag else ""))
        sys.exit(0)

    # --- 确定标题 ---
    title = args.title or Path(input_path).stem

    # --- 获取字幕数据 ---
    subtitles: list[dict] = []
    temp_files: list[str] = []
    whisper_meta: Optional[dict] = None

    try:
        if args.sub_file:
            # 使用指定的外部字幕文件
            print(f"[*] 解析外部字幕文件: {args.sub_file}")
            subtitles = detect_and_parse(args.sub_file)
        else:
            ext = Path(input_path).suffix.lower()
            if ext in (".vtt", ".srt", ".ass", ".ssa"):
                # 直接解析字幕文件
                print(f"[*] 解析字幕文件: {input_path}")
                subtitles = detect_and_parse(input_path)
            else:
                # ========= 视频文件处理 =========
                print(f"[*] 探测字幕流: {input_path}")
                streams = probe_subtitle_streams(input_path)

                if streams:
                    # --- 路径 A: 有嵌入式字幕 → 直接提取 ---
                    target_stream = None
                    for s in streams:
                        if s["index"] == args.stream:
                            target_stream = s
                            break
                    if target_stream is None:
                        print(f"[warn] 未找到字幕流 #{args.stream}，可用流:")
                        for s in streams:
                            print(f"  #{s['index']} | {s['codec_name']}")
                        sys.exit(1)

                    lang = target_stream.get("tags", {}).get("language", "unknown")
                    print(f"[*] 提取字幕流 #{args.stream} ({lang}) ...")
                    temp_srt = extract_embedded_subtitle(input_path, args.stream)
                    temp_files.append(temp_srt)
                    print(f"[*] 提取完成: {temp_srt}")
                    subtitles = detect_and_parse(temp_srt)

                elif args.whisper:
                    # --- 路径 B: 无字幕 + --whisper → AI 转录 ---
                    if not _WHISPER_AVAILABLE:
                        print("[error] Whisper AI 转录不可用，请安装依赖:", file=sys.stderr)
                        print("  pip install openvino-genai librosa", file=sys.stderr)
                        sys.exit(1)

                    print("[*] 视频无嵌入式字幕，启用 Whisper AI 转录 ...")
                    temp_wav = extract_audio(input_path)
                    temp_files.append(temp_wav)

                    subtitles, whisper_meta = transcribe_whisper(
                        temp_wav,
                        model_name=args.whisper_model,
                        device=args.whisper_device,
                        language=args.whisper_language,
                    )

                else:
                    # --- 路径 C: 无字幕 + 无 --whisper → 提示 ---
                    print("[warn] 视频中未找到嵌入式字幕流", file=sys.stderr)
                    print("[warn] 可用以下方式处理:", file=sys.stderr)
                    print("  1. 使用 Whisper AI 转录: 添加 --whisper 参数", file=sys.stderr)
                    print("     pip install openvino-genai librosa", file=sys.stderr)
                    print("  2. 指定外部字幕文件:  --subtitle-file <文件路径>", file=sys.stderr)
                    sys.exit(1)

        if not subtitles:
            print("[warn] 未解析到任何字幕内容", file=sys.stderr)
            sys.exit(1)

        print(f"[*] 共获得 {len(subtitles)} 条字幕")

        # --- 生成 Markdown ---
        md = subtitles_to_markdown(subtitles, title=title, mode=args.mode,
                                   meta=whisper_meta)

        # --- 输出 ---
        if args.output:
            output_path = args.output
        else:
            base = Path(input_path).stem
            output_path = f"{base}.subtitles.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"[✓] 字幕已输出到: {output_path}")
        print(f"    模式: {args.mode} | 条数: {len(subtitles)}")
        if args.srt:
            srt_path = str(Path(input_path).with_suffix(".srt"))
            subtitles_to_srt(subtitles, srt_path)
            print(f"[✓] SRT 字幕已输出到: {srt_path} (可直接导入剪映)")
        if whisper_meta:
            print(f"    转录: Whisper {whisper_meta['model']} ({whisper_meta['device']})"
                  + f" | 语言: {whisper_meta.get('detected_language','?')}")

    finally:
        # 清理临时文件
        if not args.keep_temp:
            for tf in temp_files:
                try:
                    os.remove(tf)
                    print(f"[*] 清理临时文件: {tf}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
