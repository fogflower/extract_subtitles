# extract_subtitles — 视频字幕提取 / AI 转录工具

从视频文件提取字幕，输出剪映可直接导入的 **SRT** 或文稿 **Markdown**。

## 功能

- **内嵌字幕提取**：视频自带字幕流时（如 mkv），用 ffmpeg 直接提取
- **Whisper AI 转录**：视频没有字幕时，用 Whisper（OpenVINO GenAI，支持 Intel GPU）语音识别生成带时间戳的字幕
- **字幕文件解析**：VTT / SRT / ASS / SSA → 干净的 Markdown 文稿（带时间戳 / 段落 / 纯文本三种模式）
- **SRT 输出**：`--srt` 一键生成 UTF-8 SRT，可直接导入剪映 / CapCut 编辑

## 环境要求

- Python 3.10+
- ffmpeg（在 PATH 中）
- 仅使用 Whisper 转录时需要：

```bash
pip install openvino-genai librosa huggingface_hub
```

## 用法

```bash
# 视频自带内嵌字幕 → 提取为 SRT + Markdown 文稿
python extract_subtitles.py video.mkv --srt

# 视频没有字幕 → Whisper AI 转录
python extract_subtitles.py video.mp4 --whisper --srt

# 中文视频指定语言更准
python extract_subtitles.py video.mp4 --whisper --whisper-language zh --srt

# 更准的模型（默认 base；tiny 最快，small 更准）
python extract_subtitles.py video.mp4 --whisper --whisper-model small --srt

# 直接解析已有字幕文件 → Markdown
python extract_subtitles.py subs.vtt
python extract_subtitles.py subs.vtt --srt

# 列出视频中的字幕流
python extract_subtitles.py video.mkv -l
```

首次使用 Whisper 会自动从 HuggingFace 下载模型（约 150–500 MB，缓存于 `~/.cache/whisper-ov/`）。

## 输出

| 输出 | 说明 |
|---|---|
| `*.srt`（--srt） | UTF-8 SRT 字幕，拖进剪映时间轴即可编辑 |
| `*.subtitles.md` | Markdown 文稿（timestamp / paragraph / plain 模式） |

## 剪映导入

新建草稿 → 导入视频 → 把 `.srt` 文件拖到时间轴上方（或「文本」→「字幕」→「导入字幕」），每条字幕即成为可编辑文本块。

## 配套脚本

- `vtt_to_md.py`：把 YouTube 自动字幕 .vtt 清洗为无时间戳的 Markdown 文稿
  ```bash
  python vtt_to_md.py subs.vtt            # 单个文件
  python vtt_to_md.py ./subtitle_dir      # 递归处理目录
  ```
