from __future__ import annotations

import re
from pathlib import Path

from .models import TranscriptResult

ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def sanitize_filename(name: str, max_length: int = 35) -> str:
    """清理文件名中的非法字符与话题标签，确保跨平台文件系统安全。"""
    # 移除抖音话题标签 #xxx
    cleaned = re.sub(r"#\S+", "", name)
    cleaned = ILLEGAL_FILENAME_CHARS.sub("_", cleaned).strip(" ._")
    if not cleaned:
        return "未命名"
    return cleaned[:max_length].strip(" ._")


def find_existing_markdown(output_dir: Path, aweme_id: str) -> Path | None:
    """检查输出目录中是否已存在指定作品 ID 的 Markdown 文件（用于断点续转）。"""
    if not output_dir.exists() or not aweme_id:
        return None
    suffix_id = aweme_id[-6:]
    for md_file in output_dir.glob(f"*_{suffix_id}.md"):
        try:
            head = md_file.read_text(encoding="utf-8")[:500]
            if f"`{aweme_id}`" in head:
                return md_file
        except Exception:
            continue
    return None


def render_markdown(result: TranscriptResult) -> str:
    """将识别结果渲染为结构化的 Markdown 文稿。"""
    meta = result.metadata
    lines = [
        f"# {meta.title}",
        "",
        f"- **作者**：{meta.author}",
        f"- **作品 ID**：`{meta.aweme_id}`",
        f"- **原链接**：{meta.source_url}",
    ]
    if meta.create_time:
        lines.append(f"- **发布时间**：{meta.create_time}")
    lines.extend(
        [
            f"- **转换时间**：{result.converted_at}",
            f"- **识别耗时**：{result.elapsed_seconds:.2f}s",
            "",
        ]
    )

    if meta.description:
        lines.extend(
            [
                "## 视频原简介",
                "",
                *(f"> {line}" for line in meta.description.splitlines()),
                "",
            ]
        )

    body = result.formatted_text.strip() or "*（未识别到有效人声内容）*"
    lines.extend(
        [
            "## 语音转文字正文",
            "",
            body,
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown(result: TranscriptResult, output_dir: Path) -> Path:
    """将单个视频的转写结果保存为独立的 .md 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = result.metadata
    safe_author = sanitize_filename(meta.author, max_length=16)
    safe_title = sanitize_filename(meta.title, max_length=30)
    short_id = meta.aweme_id[-6:] if meta.aweme_id else "000000"

    filename = f"{safe_author}_{safe_title}_{short_id}.md"
    output_path = output_dir / filename
    output_path.write_text(render_markdown(result), encoding="utf-8")
    return output_path
