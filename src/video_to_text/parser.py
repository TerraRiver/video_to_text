from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# 匹配抖音与 B 站分享短链、长链及分 P 链接
VIDEO_URL_PATTERN = re.compile(
    r"https?://(?:"
    # 抖音链接
    r"v\.douyin\.com/[a-zA-Z0-9_\-]+/?|"
    r"www\.douyin\.com/(?:video|note)/\d+|"
    r"www\.douyin\.com/[^\s\"'<>，。；！）)]*?[?&]modal_id=\d+|"
    r"www\.iesdouyin\.com/share/video/\d+|"
    # B 站链接 (b23.tv 短链 / bilibili.com BV或av长链)
    r"b23\.tv/[a-zA-Z0-9_\-]+/?|"
    r"(?:www\.|m\.)?bilibili\.com/video/(?:BV[a-zA-Z0-9]+|av\d+)(?:[/?][^\s\"'<>，。；！）)]*)?"
    r")"
)

AWEME_ID_PATTERN = re.compile(r"/(?:video|note)/(\d+)")
BILIBILI_BVID_PATTERN = re.compile(r"/video/(BV[a-zA-Z0-9]+|av\d+)", re.IGNORECASE)


def detect_platform(url: str) -> str:
    """根据 URL 识别所属平台 ('bilibili' 或 'douyin')。"""
    lowered = url.lower()
    if "bilibili.com" in lowered or "b23.tv" in lowered:
        return "bilibili"
    return "douyin"


def extract_bilibili_page_num(url: str) -> int:
    """从 B 站链接查询参数中提取分 P 页码 (?p=N)，默认为 1。"""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        p_vals = qs.get("p")
        if p_vals:
            p_int = int(p_vals[0])
            if p_int >= 1:
                return p_int
    except Exception:
        pass
    return 1


def clean_bilibili_url(url: str) -> str:
    """清理 B 站链接中的追踪参数 (如 vd_source, share_source 等)，仅保留核心路径与大于 1 的 ?p=N 参数。"""
    if "b23.tv" in url.lower():
        return url.rstrip("/")

    parsed = urlparse(url)
    bvid_match = BILIBILI_BVID_PATTERN.search(parsed.path)
    if not bvid_match:
        return url

    video_id = bvid_match.group(1)
    if video_id.lower().startswith("bv"):
        video_id = "BV" + video_id[2:]
    else:
        video_id = video_id.lower()

    clean_path = f"/video/{video_id}"
    page_num = extract_bilibili_page_num(url)
    clean_query = urlencode({"p": page_num}) if page_num > 1 else ""

    return urlunparse(
        (
            parsed.scheme or "https",
            "www.bilibili.com",
            clean_path,
            "",
            clean_query,
            "",
        )
    )


def normalize_extracted_url(url: str) -> str:
    """对提取出的 URL 进行尾部标点剥离与平台规范化。"""
    cleaned = url.rstrip(".,;，。；！）)】]")
    if detect_platform(cleaned) == "bilibili":
        return clean_bilibili_url(cleaned)
    return cleaned


def extract_video_urls(text: str) -> list[str]:
    """从任意文本或分享口令中提取所有抖音与 B 站链接并保持顺序去重。"""
    urls: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for match in VIDEO_URL_PATTERN.finditer(stripped):
            url = normalize_extracted_url(match.group(0))
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


# 保留向后兼容别名
extract_douyin_urls = extract_video_urls


def extract_urls_from_file(file_path: Path) -> list[str]:
    """从文件中读取文本并提取所有支持的视频链接。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到链接文件: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    return extract_video_urls(content)


def extract_video_id_from_url(url: str) -> str | None:
    """尝试从长链接或 URL 参数中提取作品唯一 ID (抖音 aweme_id 或 B 站 BV/av 号 + 分P)。"""
    platform = detect_platform(url)
    if platform == "bilibili":
        bvid_match = BILIBILI_BVID_PATTERN.search(url)
        if not bvid_match:
            return None
        vid = bvid_match.group(1)
        if vid.lower().startswith("bv"):
            vid = "BV" + vid[2:]
        else:
            vid = vid.lower()
        page_num = extract_bilibili_page_num(url)
        return f"{vid}_p{page_num}" if page_num > 1 else vid

    match = AWEME_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    modal_match = re.search(r"modal_id=(\d+)", url)
    if modal_match:
        return modal_match.group(1)
    return None


# 保留向后兼容别名
extract_aweme_id_from_url = extract_video_id_from_url
