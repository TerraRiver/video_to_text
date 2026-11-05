from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class VideoMetadata:
    """视频元数据（支持抖音与 B 站）及本地临时音频路径。"""

    source_url: str
    aweme_id: str
    title: str
    author: str
    platform: str = "douyin"
    description: str = ""
    create_time: str = ""
    media_url: str = ""
    backup_media_urls: list[str] = field(default_factory=list)
    audio_path: Path | None = None


@dataclass
class TranscriptResult:
    """单个视频语音转写与大模型校对结果。"""

    metadata: VideoMetadata
    raw_text: str
    formatted_text: str
    checked_text: str = ""
    llm_model: str = ""
    llm_elapsed_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    converted_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
