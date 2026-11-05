from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np

from video_to_text.downloader import DouyinDownloader, convert_media_to_wav16k
from video_to_text.exporter import (
    find_existing_markdown,
    render_markdown,
    sanitize_filename,
    save_markdown,
)
from video_to_text.models import TranscriptResult, VideoMetadata
from video_to_text.parser import (
    extract_aweme_id_from_url,
    extract_douyin_urls,
    extract_urls_from_file,
)
from video_to_text.transcriber import clean_sensevoice_text


def test_extract_douyin_urls(tmp_path: Path) -> None:
    sample_text = """
    # 这是注释行 https://v.douyin.com/ignored/
    4.89 复制打开抖音，看看【测试作者的作品】今日分享干货 https://v.douyin.com/iYAbCdEf/ 快来围观！
    重复口令测试 https://v.douyin.com/iYAbCdEf/
    网页端长链：https://www.douyin.com/video/7398765432101234567，真不错。
    图文长链：https://www.douyin.com/note/7311111111111111111
    发现页弹窗链接：https://www.douyin.com/discover?modal_id=7322222222222222222
    """
    urls = extract_douyin_urls(sample_text)
    assert urls == [
        "https://v.douyin.com/iYAbCdEf/",
        "https://www.douyin.com/video/7398765432101234567",
        "https://www.douyin.com/note/7311111111111111111",
        "https://www.douyin.com/discover?modal_id=7322222222222222222",
    ]

    links_file = tmp_path / "links.txt"
    links_file.write_text(sample_text, encoding="utf-8")
    assert extract_urls_from_file(links_file) == urls


def test_extract_aweme_id_from_url() -> None:
    assert (
        extract_aweme_id_from_url("https://www.douyin.com/video/7398765432101234567")
        == "7398765432101234567"
    )
    assert (
        extract_aweme_id_from_url(
            "https://www.douyin.com/discover?modal_id=7398765432101234567"
        )
        == "7398765432101234567"
    )
    assert extract_aweme_id_from_url("https://v.douyin.com/iYAbCdEf/") is None


def test_clean_sensevoice_text() -> None:
    raw = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>大家好，今天我们来聊聊如何使用Python批量处理抖音视频。这个方法非常简单实用！"
    cleaned = clean_sensevoice_text(raw)
    assert "<|" not in cleaned
    assert "大家好，今天我们来聊聊如何使用Python批量处理抖音视频。" in cleaned


def test_exporter_and_dedup(tmp_path: Path) -> None:
    assert sanitize_filename("作者/名字:测试*?#标签") == "作者_名字_测试"

    meta = VideoMetadata(
        source_url="https://v.douyin.com/iYAbCdEf/",
        aweme_id="7398765432101234567",
        title="测试视频标题 #人工智能 #编程",
        author="测试博主",
        description="测试视频标题 #人工智能 #编程\n第二行简介",
        create_time="2026-11-05 12:00:00",
    )
    result = TranscriptResult(
        metadata=meta,
        raw_text="<|zh|>这是一段测试语音转写内容。",
        formatted_text="这是一段测试语音转写内容。",
        elapsed_seconds=1.23,
    )

    md_text = render_markdown(result)
    assert "# 测试视频标题 #人工智能 #编程" in md_text
    assert "`7398765432101234567`" in md_text
    assert "这是一段测试语音转写内容。" in md_text

    saved_path = save_markdown(result, tmp_path)
    assert saved_path.exists()
    assert saved_path.name == "测试博主_测试视频标题_234567.md"

    existing = find_existing_markdown(tmp_path, "7398765432101234567")
    assert existing == saved_path
    assert find_existing_markdown(tmp_path, "9999999999999999999") is None


def test_extract_media_url_from_detail() -> None:
    detail = {
        "aweme_id": "12345",
        "video": {
            "play_addr": {
                "url_list": [
                    "https://v26-web.douyinvod.com/video/tos/cn/test1.mp4",
                    "https://v3-web.douyinvod.com/video/tos/cn/test2.mp4",
                ]
            }
        },
    }
    assert (
        DouyinDownloader._extract_media_url_from_detail(detail)
        == "https://v26-web.douyinvod.com/video/tos/cn/test1.mp4"
    )


def test_convert_media_to_wav16k(tmp_path: Path) -> None:
    """测试使用内置 PyAV 将 44.1kHz 双声道音频重采样为 16kHz 单声道 WAV。"""
    src_audio = tmp_path / "input_44k.wav"
    dst_wav = tmp_path / "output_16k.wav"

    sample_rate = 44100
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    tone = (np.sin(2 * math.pi * 440 * t) * 16000).astype(np.int16)
    stereo = np.vstack([tone, tone])  # shape (2, N)

    with av.open(str(src_audio), mode="w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=sample_rate)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    convert_media_to_wav16k(src_audio, dst_wav)
    assert dst_wav.exists()

    with av.open(str(dst_wav)) as check_container:
        audio_stream = check_container.streams.audio[0]
        assert audio_stream.rate == 16000
        assert audio_stream.channels == 1
