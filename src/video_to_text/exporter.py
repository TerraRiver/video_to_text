from __future__ import annotations

import re
from pathlib import Path

from .models import TranscriptResult, VideoMetadata
from .parser import detect_platform

ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def sanitize_filename(name: str, max_length: int = 35) -> str:
    """清理文件名中的非法字符与话题标签，确保跨平台文件系统安全。"""
    # 移除抖音/B站话题标签 #xxx
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
            head = md_file.read_text(encoding="utf-8")[:600]
            if f"`{aweme_id}`" in head:
                return md_file
        except Exception:
            continue
    return None


def render_markdown(result: TranscriptResult) -> str:
    """将识别与大模型校对结果渲染为结构化的 Markdown 文稿。"""
    meta = result.metadata
    platform_label = "B站 (bilibili)" if meta.platform == "bilibili" else "抖音 (douyin)"
    lines = [
        f"# {meta.title}",
        "",
        f"- **平台**：{platform_label}",
        f"- **作者**：{meta.author}",
        f"- **作品 ID**：`{meta.aweme_id}`",
        f"- **原链接**：{meta.source_url}",
    ]
    if meta.create_time:
        lines.append(f"- **发布时间**：{meta.create_time}")
    lines.append(f"- **转换时间**：{result.converted_at}")
    lines.append(f"- **识别耗时**：{result.elapsed_seconds:.2f}s")
    if result.checked_text and result.llm_model:
        lines.append(
            f"- **AI 校对模型**：`{result.llm_model}` ({result.llm_elapsed_seconds:.2f}s)"
        )
    lines.append("")

    if meta.description:
        lines.extend(
            [
                "## 视频原简介",
                "",
                *(f"> {line}" for line in meta.description.splitlines()),
                "",
            ]
        )

    raw_body = result.formatted_text.strip() or "*（未识别到有效人声内容）*"
    checked_body = result.checked_text.strip()

    if checked_body:
        lines.extend(
            [
                "## 语音转文字正文（AI 校对版）",
                "",
                checked_body,
                "",
                "<details>",
                "<summary>查看原始语音识别文本 (SenseVoiceSmall)</summary>",
                "",
                raw_body,
                "",
                "</details>",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 语音转文字正文",
                "",
                raw_body,
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


def is_markdown_already_checked(md_content: str) -> bool:
    """判断 Markdown 文件是否已经包含大模型校对结果。"""
    return "## 语音转文字正文（AI 校对版）" in md_content


def parse_markdown_for_check(md_path: Path) -> TranscriptResult:
    """解析已生成的 Markdown 文件，提取元信息与原始语音识别正文供大模型校对。"""
    content = md_path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    author_match = re.search(r"^-\s+\*\*作者\*\*：(.+)$", content, re.MULTILINE)
    aweme_match = re.search(r"^-\s+\*\*作品 ID\*\*：`([^`]+)`", content, re.MULTILINE)
    url_match = re.search(r"^-\s+\*\*原链接\*\*：(.+)$", content, re.MULTILINE)
    ctime_match = re.search(r"^-\s+\*\*发布时间\*\*：(.+)$", content, re.MULTILINE)
    conv_match = re.search(r"^-\s+\*\*转换时间\*\*：(.+)$", content, re.MULTILINE)
    elapsed_match = re.search(
        r"^-\s+\*\*识别耗时\*\*：([\d.]+)s", content, re.MULTILINE
    )

    desc_match = re.search(
        r"## 视频原简介\s*\n+((?:>.*(?:\n|$))+)", content, re.MULTILINE
    )
    description = ""
    if desc_match:
        desc_lines = [
            re.sub(r"^>\s?", "", line)
            for line in desc_match.group(1).strip().splitlines()
        ]
        description = "\n".join(desc_lines).strip()

    # 如果已有 <details> 折叠块，优先提取折叠块内的原始语音识别文本
    details_match = re.search(
        r"<summary>查看原始语音识别文本[^<]*</summary>\s*(.*?)\s*</details>",
        content,
        re.DOTALL,
    )
    if details_match:
        original_text = details_match.group(1).strip()
    else:
        body_match = re.search(
            r"## 语音转文字正文(?:（AI 校对版）)?\s*\n+(.*)",
            content,
            re.DOTALL,
        )
        original_text = body_match.group(1).strip() if body_match else ""

    if original_text == "*（未识别到有效人声内容）*":
        original_text = ""

    source_url = url_match.group(1).strip() if url_match else ""
    aweme_id = aweme_match.group(1).strip() if aweme_match else ""
    platform = (
        "bilibili"
        if (
            detect_platform(source_url) == "bilibili"
            or aweme_id.upper().startswith(("BV", "AV"))
        )
        else "douyin"
    )

    meta = VideoMetadata(
        source_url=source_url,
        aweme_id=aweme_id,
        title=title_match.group(1).strip() if title_match else md_path.stem,
        author=author_match.group(1).strip() if author_match else "未知作者",
        platform=platform,
        description=description,
        create_time=ctime_match.group(1).strip() if ctime_match else "",
    )

    return TranscriptResult(
        metadata=meta,
        raw_text=original_text,
        formatted_text=original_text,
        elapsed_seconds=float(elapsed_match.group(1)) if elapsed_match else 0.0,
        converted_at=conv_match.group(1).strip() if conv_match else "",
    )
