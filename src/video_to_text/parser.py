from __future__ import annotations

import re
from pathlib import Path

# 匹配抖音分享短链、网页长链与带 modal_id 的发现页/主页链接
DOUYIN_URL_PATTERN = re.compile(
    r"https?://(?:"
    r"v\.douyin\.com/[a-zA-Z0-9_\-]+/?|"
    r"www\.douyin\.com/(?:video|note)/\d+|"
    r"www\.douyin\.com/[^\s\"'<>]*?[?&]modal_id=\d+|"
    r"www\.iesdouyin\.com/share/video/\d+"
    r")"
)
AWEME_ID_PATTERN = re.compile(r"/(?:video|note)/(\d+)")


def extract_douyin_urls(text: str) -> list[str]:
    """从任意文本或抖音分享口令中提取所有抖音链接并保持顺序去重。"""
    urls: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for match in DOUYIN_URL_PATTERN.finditer(stripped):
            url = match.group(0).rstrip(".,;，。；！）)")
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def extract_urls_from_file(file_path: Path) -> list[str]:
    """从文件中读取文本并提取所有抖音链接。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到链接文件: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    return extract_douyin_urls(content)


def extract_aweme_id_from_url(url: str) -> str | None:
    """尝试从长链接或 URL 参数中提取作品 ID (aweme_id)。"""
    match = AWEME_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    modal_match = re.search(r"modal_id=(\d+)", url)
    if modal_match:
        return modal_match.group(1)
    return None
