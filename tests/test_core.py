from __future__ import annotations

import json
import math
from pathlib import Path

import av
import httpx
import numpy as np

from video_to_text.downloader import DouyinDownloader, convert_media_to_wav16k
from video_to_text.exporter import (
    find_existing_markdown,
    is_markdown_already_checked,
    parse_markdown_for_check,
    render_markdown,
    sanitize_filename,
    save_markdown,
)
from video_to_text.llm_checker import LLMChecker, LLMConfig, split_text_into_chunks
from video_to_text.models import TranscriptResult, VideoMetadata
from video_to_text.parser import (
    clean_bilibili_url,
    detect_platform,
    extract_aweme_id_from_url,
    extract_douyin_urls,
    extract_urls_from_file,
    extract_video_id_from_url,
    extract_video_urls,
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


def test_llm_config_and_chunking(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "LLM_API_KEY='sk-test-12345'\n"
        "LLM_BASE_URL=https://api.example.com/v1/ # 行尾注释测试\n"
        "LLM_MODEL=qwen-plus # 默认模型\n",
        encoding="utf-8",
    )
    cfg = LLMConfig.from_env(env_file)
    assert cfg.is_configured
    assert cfg.api_key == "sk-test-12345"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "qwen-plus"

    # 验证占位符 sk-xxxx 与空前缀 sk- 被识别为未配置
    placeholder_cfg = LLMConfig(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    assert not placeholder_cfg.is_configured
    assert not LLMConfig(api_key="sk-").is_configured

    text = "第一段内容。\n\n第二段内容比较长。\n\n第三段内容。"
    chunks = split_text_into_chunks(text, max_chars=12)
    assert len(chunks) == 3
    assert chunks[0] == "第一段内容。"


def test_llm_checker_and_markdown_roundtrip(tmp_path: Path) -> None:
    meta = VideoMetadata(
        source_url="https://v.douyin.com/iUZFlH_XNeU/",
        aweme_id="7703448870348475683",
        title="被唐朝打败的突厥，跟今天的土耳其，到底是什么关系？ #土耳其 #突厥",
        author="胖哥扯淡",
        description="被唐朝打败的突厥，跟今天的土耳其，到底是什么关系？ #土耳其 #突厥",
        create_time="2026-11-02 17:55:00",
    )
    raw_asr = "突厥韩国兴起之时，西卫把长乐公主嫁给了突厥的首领。"
    result = TranscriptResult(
        metadata=meta,
        raw_text=raw_asr,
        formatted_text=raw_asr,
        elapsed_seconds=5.0,
    )

    # 1. 先保存未经过 AI 校对的初稿
    md_path = save_markdown(result, tmp_path)
    initial_md = md_path.read_text(encoding="utf-8")
    assert not is_markdown_already_checked(initial_md)

    # 2. 解析存量 Markdown 并使用 MockTransport 模拟大模型校对
    parsed_result = parse_markdown_for_check(md_path)
    assert parsed_result.metadata.aweme_id == "7703448870348475683"
    assert parsed_result.formatted_text == raw_asr

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]
        assert "被唐朝打败的突厥" in user_msg
        assert "突厥韩国兴起之时" in user_msg
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "突厥汗国兴起之时，西魏把长乐公主嫁给了突厥的首领。"
                        }
                    }
                ]
            },
        )

    cfg = LLMConfig(
        api_key="sk-mock",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    checker = LLMChecker(cfg, transport=httpx.MockTransport(mock_handler))
    checker.check_result(parsed_result)

    assert (
        parsed_result.checked_text
        == "突厥汗国兴起之时，西魏把长乐公主嫁给了突厥的首领。"
    )

    # 3. 写入校对后的 Markdown，验证包含 AI 校对版与 <details> 原始文本折叠块
    checked_md = render_markdown(parsed_result)
    md_path.write_text(checked_md, encoding="utf-8")
    assert is_markdown_already_checked(checked_md)
    assert "## 语音转文字正文（AI 校对版）" in checked_md
    assert "突厥汗国兴起之时，西魏把长乐公主嫁给了突厥的首领。" in checked_md
    assert "<summary>查看原始语音识别文本 (SenseVoiceSmall)</summary>" in checked_md
    assert raw_asr in checked_md

    # 4. 再次解析已校对的 Markdown（如 --force 重新校对时），应提取出 <details> 内的原始文本
    reparsed = parse_markdown_for_check(md_path)
    assert reparsed.formatted_text == raw_asr

    # 5. 验证无有效人声的文稿在解析时将 formatted_text 置为空字符串
    empty_result = TranscriptResult(
        metadata=meta,
        raw_text="",
        formatted_text="",
        elapsed_seconds=1.0,
    )
    empty_md_path = tmp_path / "empty_voice.md"
    empty_md_path.write_text(render_markdown(empty_result), encoding="utf-8")
    parsed_empty = parse_markdown_for_check(empty_md_path)
    assert parsed_empty.formatted_text == ""


def test_bilibili_urls_and_platform_detection() -> None:
    sample_mixed = """
    抖音口令：4.89 复制打开抖音 https://v.douyin.com/iYAbCdEf/
    B站短链：【【何同学】我做了一个自己打字的键盘...】 https://b23.tv/abcDEF1，快来看！
    B站长链带追踪参数：https://www.bilibili.com/video/BV1xx411c7mD/?spm_id_from=333.1007&vd_source=abcdef12345
    B站多P链接：https://www.bilibili.com/video/BV1xx411c7mD?p=2&share_source=copy_web
    """
    urls = extract_video_urls(sample_mixed)
    assert urls == [
        "https://v.douyin.com/iYAbCdEf/",
        "https://b23.tv/abcDEF1",
        "https://www.bilibili.com/video/BV1xx411c7mD",
        "https://www.bilibili.com/video/BV1xx411c7mD?p=2",
    ]

    assert detect_platform("https://v.douyin.com/iYAbCdEf/") == "douyin"
    assert detect_platform("https://b23.tv/abcDEF1") == "bilibili"
    assert detect_platform("https://www.bilibili.com/video/BV1xx411c7mD") == "bilibili"

    assert (
        extract_video_id_from_url("https://www.bilibili.com/video/BV1xx411c7mD")
        == "BV1xx411c7mD"
    )
    assert (
        extract_video_id_from_url("https://www.bilibili.com/video/BV1xx411c7mD?p=2")
        == "BV1xx411c7mD_p2"
    )
    assert extract_video_id_from_url("https://b23.tv/abcDEF1") is None
    assert (
        clean_bilibili_url(
            "https://m.bilibili.com/video/bv1xx411c7mD/?p=1&vd_source=123"
        )
        == "https://www.bilibili.com/video/BV1xx411c7mD"
    )


def test_bilibili_state_and_playinfo_parsing() -> None:
    initial_state = {
        "bvid": "BV1xx411c7mD",
        "videoData": {
            "bvid": "BV1xx411c7mD",
            "title": "深入理解计算机系统精讲",
            "desc": "本课程带你精读 CSAPP 核心章节",
            "pubdate": 1730000000,
            "owner": {"name": "硬核计算机UP主"},
            "pages": [
                {"page": 1, "part": "导论与概述"},
                {"page": 2, "part": "信息的表示和处理"},
            ],
        },
        "tags": [
            {"tag_name": "计算机科学"},
            {"tag_name": "CSAPP"},
        ],
    }
    playinfo = {
        "data": {
            "dash": {
                "audio": [
                    {
                        "id": 30216,
                        "bandwidth": 64000,
                        "baseUrl": "https://upos-sz-mirrorcos.bilivideo.com/audio_low.m4s",
                        "backupUrl": [],
                    },
                    {
                        "id": 30280,
                        "bandwidth": 192000,
                        "baseUrl": "https://upos-sz-mirrorcos.bilivideo.com/audio_high.m4s",
                        "backupUrl": [
                            "https://upos-sz-mirrorhw.bilivideo.com/audio_high_bak.m4s"
                        ],
                    },
                ]
            }
        }
    }

    meta = DouyinDownloader._parse_bilibili_state_and_playinfo(
        source_url="https://www.bilibili.com/video/BV1xx411c7mD?p=2",
        final_url="https://www.bilibili.com/video/BV1xx411c7mD?p=2",
        page_num=2,
        initial_state=initial_state,
        playinfo=playinfo,
    )

    assert meta.platform == "bilibili"
    assert meta.aweme_id == "BV1xx411c7mD_p2"
    assert meta.author == "硬核计算机UP主"
    assert meta.title == "深入理解计算机系统精讲 (P2 信息的表示和处理)"
    assert "#计算机科学 #CSAPP" in meta.description
    assert (
        meta.media_url
        == "https://upos-sz-mirrorcos.bilivideo.com/audio_high.m4s"
    )
    assert meta.backup_media_urls == [
        "https://upos-sz-mirrorhw.bilivideo.com/audio_high_bak.m4s"
    ]


def test_llm_concurrent_chunk_order() -> None:
    """验证多线程并发分块校对时（后发的分块先返回），最终拼接结果依然严格保持原始段落顺序。"""
    import time as _time

    def delayed_mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][1]["content"]
        if "第一段原始文本" in user_msg:
            _time.sleep(0.08)
            reply = "第一段校对后文本"
        elif "第二段原始文本" in user_msg:
            _time.sleep(0.03)
            reply = "第二段校对后文本"
        else:
            reply = "第三段校对后文本"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply}}]},
        )

    cfg = LLMConfig(
        api_key="sk-mock",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        max_chunk_chars=12,
        max_concurrency=50,
    )
    checker = LLMChecker(cfg, transport=httpx.MockTransport(delayed_mock_handler))
    raw_input = "第一段原始文本\n\n第二段原始文本\n\n第三段原始文本"
    checked_text, _ = checker.check_text(raw_input)
    assert checked_text == "第一段校对后文本\n\n第二段校对后文本\n\n第三段校对后文本"

    # 验证多视频 check_batch 全局并发池同时校对多篇文稿
    res1 = TranscriptResult(
        metadata=VideoMetadata(
            source_url="u1", aweme_id="1", title="视频1", author="A1"
        ),
        raw_text=raw_input,
        formatted_text=raw_input,
    )
    res2 = TranscriptResult(
        metadata=VideoMetadata(
            source_url="u2", aweme_id="2", title="视频2", author="A2"
        ),
        raw_text="第一段原始文本\n\n第二段原始文本",
        formatted_text="第一段原始文本\n\n第二段原始文本",
    )
    errors = checker.check_batch([res1, res2])
    assert errors == {0: None, 1: None}
    assert res1.checked_text == "第一段校对后文本\n\n第二段校对后文本\n\n第三段校对后文本"
    assert res2.checked_text == "第一段校对后文本\n\n第二段校对后文本"
