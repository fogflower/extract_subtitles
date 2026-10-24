"""
Convert VTT subtitle files to clean Markdown files (no timestamps).

Usage: python vtt_to_md.py <vtt_file_or_directory>
  - Pass a .vtt file to convert just that one
  - Pass a directory to convert all .vtt files found recursively
  - Each output .md file is placed next to its source .vtt
"""

import sys
import os
import re
import html
import glob as glob_module


def clean_vtt_line(line: str) -> str:
    """Remove VTT formatting tags from a text line, returning clean text."""
    # Remove timestamp tags like <00:00:01.719>
    line = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', line)
    # Unwrap <c>text</c> → text
    line = re.sub(r'<c\.color\w+\.bg_\w+>(.*?)</c>', r'\1', line)
    line = re.sub(r'<c>(.*?)</c>', r'\1', line)
    # Remove orphan <c> / </c> tags
    line = re.sub(r'</?c(?:\.\w+)?>', '', line)
    # Remove other HTML-like tags
    line = re.sub(r'<[^>]+>', '', line)
    # Decode HTML entities (&nbsp; &amp; etc.)
    line = html.unescape(line)
    # Collapse whitespace
    line = re.sub(r'\s+', ' ', line)
    return line.strip()


def is_timestamp_line(line: str) -> bool:
    """Check if line is a VTT timestamp like '00:00:01.079 --> 00:00:02.830...'"""
    return bool(re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->', line))


def is_header_line(line: str) -> bool:
    """Check if line is part of VTT header."""
    return line in ('WEBVTT', '') or line.startswith('Kind:') or line.startswith('Language:')


def vtt_to_clean_text(vtt_path: str) -> str:
    """Convert a VTT file to clean, readable text without timestamps."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    segments = []       # cleaned text segments in order
    last_clean = None   # for dedup

    for line in raw_lines:
        stripped = line.strip()

        # Skip empty lines, header, and timestamps
        if not stripped:
            continue
        if is_header_line(stripped):
            continue
        if is_timestamp_line(stripped):
            continue

        # Clean the text
        cleaned = clean_vtt_line(stripped)
        if not cleaned:
            continue

        # Deduplicate: skip if identical to the previous segment
        if cleaned == last_clean:
            continue
        last_clean = cleaned
        segments.append(cleaned)

    if not segments:
        return ""

    # --- Build flowing paragraphs ---
    # Strategy: join segments with space since VTT breaks mid-sentence.
    # Insert paragraph breaks at sentence-ending punctuation, and also
    # force breaks when a paragraph grows too long (for auto-captions
    # that lack punctuation).
    MAX_PARA_WORDS = 350  # force a break if paragraph exceeds this

    # Soft transition phrases that hint at topic shifts
    TRANSITIONS = (
        'so ', 'now ', 'but ', 'and then ', 'the other part',
        'finally ', 'secondly ', 'thirdly ', 'another ', 'next ',
        'in addition', 'additionally', 'with that', 'let me ',
        'i want to ', 'i think ', 'you know ', 'well ',
    )

    paragraphs = []
    current_para = segments[0]

    def para_word_count() -> int:
        return len(current_para.split())

    def should_break(seg: str) -> bool:
        """Return True if we should start a new paragraph before `seg`."""
        # Strong sentence-ending punctuation + next starts capital
        if current_para.rstrip().endswith(('.', '?', '!', '."', ':"', '.)', '.]')):
            if seg and seg[0].isupper() and len(seg.split()) > 2:
                return True
        # Paragraph is getting too long — look for any reasonable break
        if para_word_count() > MAX_PARA_WORDS:
            # Break if current ends with any punctuation
            if current_para.rstrip().endswith(('.', '?', '!', ',', ':', ';', '"', ')', ']')):
                return True
            # Break if next segment is a known transition phrase
            seg_lower = seg.lower() if seg else ''
            if any(seg_lower.startswith(t) for t in TRANSITIONS):
                return True
            # Break if next segment is longer than a snippet
            if len(seg.split()) >= 6:
                return True
        return False

    for seg in segments[1:]:
        if should_break(seg):
            paragraphs.append(current_para)
            current_para = seg
        else:
            current_para += ' ' + seg

    if current_para:
        paragraphs.append(current_para)

    return '\n\n'.join(paragraphs)


def convert_file(vtt_path: str) -> str:
    """Convert a single VTT file to MD. Returns the output path."""
    md_path = os.path.splitext(vtt_path)[0] + '.md'

    fname = os.path.basename(vtt_path)
    # Avoid UnicodeEncodeError on Windows console with full-width chars
    try:
        print(f"Converting: {fname}")
    except UnicodeEncodeError:
        print(f"Converting: {fname.encode('ascii', 'replace').decode('ascii')}")

    text = vtt_to_clean_text(vtt_path)

    if not text:
        print(f"  SKIPPED (no content extracted)")
        return md_path

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(text)

    para_count = text.count('\n\n') + 1
    word_count = len(text.split())
    try:
        print(f"  -> {md_path}")
    except UnicodeEncodeError:
        print(f"  -> {md_path.encode('ascii', 'replace').decode('ascii')}")
    print(f"     {word_count} words, ~{para_count} paragraphs")
    return md_path


def find_vtt_files(root: str) -> list:
    """Find all .vtt files under root directory."""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.vtt'):
                matches.append(os.path.join(dirpath, fn))
    return matches


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("No path provided. Searching for VTT files in current directory...")
        target = os.getcwd()
    else:
        target = sys.argv[1]

    if not os.path.exists(target):
        print(f"Path not found: {target}")
        sys.exit(1)

    # Collect VTT files
    if os.path.isfile(target):
        if target.lower().endswith('.vtt'):
            vtt_files = [target]
        else:
            print(f"Not a .vtt file: {target}")
            sys.exit(1)
    else:
        vtt_files = find_vtt_files(target)

    if not vtt_files:
        print("No .vtt files found.")
        return

    print(f"Found {len(vtt_files)} VTT file(s)\n")
    for vf in sorted(vtt_files):
        convert_file(vf)

    print(f"\nDone. {len(vtt_files)} file(s) processed.")


if __name__ == '__main__':
    main()
