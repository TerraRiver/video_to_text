from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import av
import httpx
from playwright.sync_api import BrowserContext, Page, Response, sync_playwright
from rich.console import Console

from .models import VideoMetadata
from .parser import extract_aweme_id_from_url

console = Console()

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


def convert_media_to_wav16k(input_path: Path, output_wav_path: Path) -> Path:
    """使用内置 PyAV (FFmpeg 库) 将视频/音频文件重采样为 16kHz 单声道 WAV，无需系统安装 ffmpeg。"""
    with av.open(str(input_path)) as in_container:
        audio_streams = [s for s in in_container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError(f"媒体文件中未找到音轨: {input_path}")
        in_stream = audio_streams[0]

        with av.open(str(output_wav_path), mode="w", format="wav") as out_container:
            out_stream = out_container.add_stream("pcm_s16le", rate=16000)
            out_stream.layout = "mono"
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=16000,
            )

            for frame in in_container.decode(in_stream):
                frame.pts = None
                resampled_frames = resampler.resample(frame)
                for r_frame in resampled_frames:
                    for packet in out_stream.encode(r_frame):
                        out_container.mux(packet)

            # Flush resampler & encoder
            for r_frame in resampler.resample(None):
                for packet in out_stream.encode(r_frame):
                    out_container.mux(packet)
            for packet in out_stream.encode(None):
                out_container.mux(packet)

    return output_wav_path


def open_login_session(profile_dir: Path) -> None:
    """打开可见的持久化浏览器窗口供用户扫码登录抖音，关闭窗口或按回车后自动保存登录态。"""
    profile_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold cyan]正在启动持久化浏览器 (配置目录: {profile_dir})...[/bold cyan]"
    )
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 820},
            args=["--disable-blink-features=AutomationControlled"],
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://www.douyin.com/", wait_until="domcontentloaded")
        console.print(
            "\n[bold green]请在弹出的浏览器窗口中使用手机抖音扫码登录。[/bold green]\n"
            "[yellow]登录成功后，直接关闭浏览器窗口（或在终端按 Ctrl+C）即可保存登录状态。[/yellow]\n"
        )
        try:
            # 等待用户关闭浏览器窗口或按 Ctrl+C
            page.wait_for_event("close", timeout=300_000)
        except (Exception, KeyboardInterrupt):
            pass
        finally:
            context.close()
    console.print("[bold green]✓ 抖音登录状态已保存到本地持久化目录！[/bold green]")


class DouyinDownloader:
    """基于 Playwright 持久化上下文的抖音视频解析与音频下载器。"""

    def __init__(
        self,
        profile_dir: Path,
        temp_dir: Path,
        headless: bool = True,
        timeout_ms: int = 30_000,
    ) -> None:
        self.profile_dir = profile_dir
        self.temp_dir = temp_dir
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._context: BrowserContext | None = None

    def __enter__(self) -> DouyinDownloader:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 820},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        # 注入反自动化检测脚本
        self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._context:
            self._context.close()
        if self._playwright:
            self._playwright.stop()

    def process_url(self, url: str) -> VideoMetadata:
        """访问抖音分享链接，解析视频元信息并下载转码为 16kHz WAV 音频。"""
        if not self._context:
            raise RuntimeError("DouyinDownloader 未初始化上下文")

        page = self._context.new_page()
        captured_detail: dict[str, Any] = {}
        captured_media_urls: list[str] = []

        def handle_response(response: Response) -> None:
            resp_url = response.url
            # 1. 捕获抖音网页端详情 API
            if "/aweme/v1/web/aweme/detail/" in resp_url and response.status == 200:
                try:
                    data = response.json()
                    aweme_detail = data.get("aweme_detail")
                    if isinstance(aweme_detail, dict):
                        captured_detail.update(aweme_detail)
                except Exception:
                    pass
            # 2. 捕获直接加载的音视频流地址作为后备
            content_type = response.headers.get("content-type", "")
            if any(
                k in content_type for k in ("video/mp4", "audio/mp4", "audio/mpeg")
            ) and any(
                d in resp_url
                for d in ("douyinvod.com", "zjcdn.com", "/aweme/v1/play/")
            ):
                if resp_url not in captured_media_urls:
                    captured_media_urls.append(resp_url)

        page.on("response", handle_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            # 等待详情接口或 video 元素出现
            self._wait_for_video_data(page, captured_detail)

            metadata = self._build_metadata(
                source_url=url,
                page=page,
                captured_detail=captured_detail,
                captured_media_urls=captured_media_urls,
            )

            if not metadata.media_url:
                raise RuntimeError(
                    "未能从页面提取到视频/音频流地址，可能是触发了验证码或作品已失效。可尝试运行 `uv run video-to-text login` 更新登录态，或加 `--no-headless` 查看页面。"
                )

            # 下载媒体并转码为 16kHz WAV，无论转码成功与否均清理原始临时视频
            raw_media_path = self._download_stream(
                media_url=metadata.media_url,
                aweme_id=metadata.aweme_id,
            )
            wav_path = self.temp_dir / f"{metadata.aweme_id}.wav"
            try:
                convert_media_to_wav16k(raw_media_path, wav_path)
            finally:
                raw_media_path.unlink(missing_ok=True)
            metadata.audio_path = wav_path
            return metadata
        finally:
            page.close()

    def _wait_for_video_data(
        self, page: Page, captured_detail: dict[str, Any]
    ) -> None:
        """等待详情数据或页面 <video> 标签加载完成。"""
        deadline = time.time() + 12.0
        while time.time() < deadline:
            if captured_detail:
                return
            try:
                video_src = page.evaluate(
                    """() => {
                        const v = document.querySelector('video');
                        if (!v) return '';
                        if (v.src && !v.src.startsWith('blob:')) return v.src;
                        const s = v.querySelector('source');
                        return s ? s.src : '';
                    }"""
                )
                if video_src:
                    return
            except Exception:
                pass
            page.wait_for_timeout(400)

    def _build_metadata(
        self,
        source_url: str,
        page: Page,
        captured_detail: dict[str, Any],
        captured_media_urls: list[str],
    ) -> VideoMetadata:
        """综合 API 拦截数据、RENDER_DATA 与 DOM 提取视频元信息。"""
        final_url = page.url
        aweme_id = (
            str(captured_detail.get("aweme_id") or "")
            or extract_aweme_id_from_url(final_url)
            or extract_aweme_id_from_url(source_url)
            or hashlib.md5(source_url.encode("utf-8")).hexdigest()[:12]
        )

        # 如果详情接口未命中，尝试从页面 RENDER_DATA 解析
        if not captured_detail:
            rendered = self._extract_from_render_data(page)
            if rendered:
                captured_detail.update(rendered)

        desc = str(captured_detail.get("desc") or "").strip()
        author = str(
            (captured_detail.get("author") or {}).get("nickname") or ""
        ).strip()
        create_ts = captured_detail.get("create_time")
        create_time_str = ""
        if isinstance(create_ts, (int, float)) and create_ts > 0:
            create_time_str = datetime.fromtimestamp(create_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # 提取音视频直链
        media_url = self._extract_media_url_from_detail(captured_detail)

        # DOM 兜底提取标题、作者与视频源
        dom_info = self._extract_from_dom(page)
        if not desc:
            desc = dom_info.get("title") or f"抖音视频_{aweme_id}"
        if not author:
            author = dom_info.get("author") or "未知作者"
        if not media_url:
            media_url = dom_info.get("video_src") or (
                captured_media_urls[0] if captured_media_urls else ""
            )

        if media_url.startswith("//"):
            media_url = "https:" + media_url

        # 提取首行或前 40 字作为干净标题
        first_line = desc.splitlines()[0].strip() if desc else f"抖音视频_{aweme_id}"
        clean_title = first_line[:50] if first_line else f"抖音视频_{aweme_id}"

        return VideoMetadata(
            source_url=source_url,
            aweme_id=aweme_id,
            title=clean_title,
            author=author,
            description=desc,
            create_time=create_time_str,
            media_url=media_url,
        )

    @staticmethod
    def _extract_media_url_from_detail(detail: dict[str, Any]) -> str:
        """从 aweme_detail 字典中提取最佳视频或音频播放直链。"""
        video_info = detail.get("video") or {}
        # 优先取 play_addr.url_list
        play_addr = video_info.get("play_addr") or {}
        url_list = play_addr.get("url_list") or []
        if url_list:
            return str(url_list[0])

        # 备选 bit_rate 列表
        bit_rates = video_info.get("bit_rate") or []
        for br in bit_rates:
            br_urls = (br.get("play_addr") or {}).get("url_list") or []
            if br_urls:
                return str(br_urls[0])

        # 图文作品背景音乐兜底
        music_info = detail.get("music") or {}
        music_urls = (music_info.get("play_url") or {}).get("url_list") or []
        if music_urls:
            return str(music_urls[0])

        return ""

    @staticmethod
    def _extract_from_render_data(page: Page) -> dict[str, Any]:
        """尝试从页面的 RENDER_DATA 脚本标签解析 awemeDetail。"""
        try:
            raw_render = page.evaluate(
                """() => {
                    const el = document.querySelector('script#RENDER_DATA');
                    return el ? el.textContent : '';
                }"""
            )
            if not raw_render:
                return {}
            data = json.loads(unquote(raw_render))
            for val in data.values():
                if isinstance(val, dict) and "aweme" in val:
                    detail = (val.get("aweme") or {}).get("detail")
                    if isinstance(detail, dict):
                        return detail
        except Exception:
            pass
        return {}

    @staticmethod
    def _extract_from_dom(page: Page) -> dict[str, str]:
        """从页面 DOM 元素兜底提取标题、作者与 <video> 源地址。"""
        try:
            return page.evaluate(
                """() => {
                    const getText = (selectors) => {
                        for (const sel of selectors) {
                            const el = document.querySelector(sel);
                            if (el && el.textContent && el.textContent.trim()) {
                                return el.textContent.trim();
                            }
                        }
                        return '';
                    };
                    const title = getText([
                        '[data-e2e="detail-video-info"] h1',
                        '[data-e2e="video-desc"]',
                        'title'
                    ]).replace(/\\s*-\\s*抖音$/, '');

                    const author = getText([
                        '[data-e2e="detail-video-info"] [data-e2e="video-author-name"]',
                        '[data-e2e="user-info"] a',
                        '.author-card-user-name'
                    ]).replace(/^@/, '');

                    let videoSrc = '';
                    const videoEl = document.querySelector('video');
                    if (videoEl) {
                        if (videoEl.src && !videoEl.src.startsWith('blob:')) {
                            videoSrc = videoEl.src;
                        } else {
                            const sources = Array.from(videoEl.querySelectorAll('source'));
                            for (const s of sources) {
                                if (s.src && !s.src.startsWith('blob:')) {
                                    videoSrc = s.src;
                                    break;
                                }
                            }
                        }
                    }
                    return { title, author, video_src: videoSrc };
                }"""
            )
        except Exception:
            return {}

    def _download_stream(self, media_url: str, aweme_id: str) -> Path:
        """携带当前浏览器 Cookie 流式下载音视频到本地临时目录。"""
        assert self._context is not None
        cookies = {c["name"]: c["value"] for c in self._context.cookies()}
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://www.douyin.com/",
            "Accept": "*/*",
        }
        target_path = self.temp_dir / f"{aweme_id}.mp4"
        with httpx.Client(
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=60.0,
        ) as client:
            with client.stream("GET", media_url) as response:
                response.raise_for_status()
                with target_path.open("wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
        return target_path
