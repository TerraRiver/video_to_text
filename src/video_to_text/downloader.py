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
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .models import VideoMetadata
from .parser import (
    detect_platform,
    extract_bilibili_page_num,
    extract_video_id_from_url,
)

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


def open_login_session(profile_dir: Path, platform: str = "all") -> None:
    """打开可见的持久化浏览器窗口供用户扫码登录抖音和/或 B 站。"""
    profile_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold cyan]正在启动持久化浏览器 (配置目录: {profile_dir})...[/bold cyan]"
    )
    platform_norm = platform.lower().strip()
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

        first_page = context.pages[0] if context.pages else context.new_page()
        if platform_norm == "bilibili":
            first_page.goto("https://www.bilibili.com/", wait_until="domcontentloaded")
            target_desc = "B 站 (bilibili.com)"
        elif platform_norm == "douyin":
            first_page.goto("https://www.douyin.com/", wait_until="domcontentloaded")
            target_desc = "抖音 (douyin.com)"
        else:
            first_page.goto("https://www.douyin.com/", wait_until="domcontentloaded")
            bili_page = context.new_page()
            bili_page.goto("https://www.bilibili.com/", wait_until="domcontentloaded")
            target_desc = "抖音 与 B 站（已分别打开两个标签页）"

        console.print(
            f"\n[bold green]请在弹出的浏览器窗口中扫码登录 {target_desc}。[/bold green]\n"
            "[yellow]登录成功后，直接关闭浏览器窗口（或在终端按 Ctrl+C）即可保存登录状态。[/yellow]\n"
        )
        try:
            # 等待所有页面被用户关闭或按 Ctrl+C
            while context.pages:
                context.pages[0].wait_for_event("close", timeout=300_000)
        except (Exception, KeyboardInterrupt):
            pass
        finally:
            context.close()
    console.print("[bold green]✓ 登录状态已保存到本地持久化目录！[/bold green]")


class DouyinDownloader:
    """基于 Playwright 持久化上下文的多平台（抖音 + B 站）视频解析与音频下载器。"""

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
        """根据 URL 平台类型自动分发解析（抖音或 B 站），并下载转码为 16kHz WAV 音频。"""
        platform = detect_platform(url)
        if platform == "bilibili":
            return self._process_bilibili_url(url)
        return self._process_douyin_url(url)

    def _process_douyin_url(self, url: str) -> VideoMetadata:
        """访问抖音分享链接，解析视频元信息并下载转码为 16kHz WAV 音频。"""
        if not self._context:
            raise RuntimeError("Downloader 未初始化上下文")

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
            self._wait_for_video_data(page, captured_detail)

            metadata = self._build_metadata(
                source_url=url,
                page=page,
                captured_detail=captured_detail,
                captured_media_urls=captured_media_urls,
            )

            if not metadata.media_url:
                raise RuntimeError(
                    "未能从抖音页面提取到视频/音频流地址，可能是触发了验证码或作品已失效。可尝试运行 `uv run video-to-text login` 更新登录态，或加 `--no-headless` 查看页面。"
                )

            raw_media_path = self._download_stream(
                media_url=metadata.media_url,
                video_id=metadata.aweme_id,
                referer="https://www.douyin.com/",
                backup_urls=metadata.backup_media_urls,
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

    def _process_bilibili_url(self, url: str) -> VideoMetadata:
        """访问 B 站分享短链或 BV 链接，提取元信息与 DASH 纯音轨 (.m4s) 并转码为 16kHz WAV。"""
        if not self._context:
            raise RuntimeError("Downloader 未初始化上下文")

        page = self._context.new_page()
        captured_playinfo: dict[str, Any] = {}
        captured_m4s_urls: list[str] = []

        def handle_bili_response(response: Response) -> None:
            resp_url = response.url
            if (
                "/x/player/wbi/playurl" in resp_url or "/x/player/playurl" in resp_url
            ) and response.status == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and data.get("data"):
                        captured_playinfo.update(data)
                except Exception:
                    pass
            elif ".m4s" in resp_url and any(
                k in resp_url for k in ("-1-30280.m4s", "-1-30232.m4s", "-1-30216.m4s")
            ):
                if resp_url not in captured_m4s_urls:
                    captured_m4s_urls.append(resp_url)

        page.on("response", handle_bili_response)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._wait_for_bilibili_data(page, captured_playinfo)

            metadata = self._build_bilibili_metadata(
                source_url=url,
                page=page,
                captured_playinfo=captured_playinfo,
                captured_m4s_urls=captured_m4s_urls,
            )

            if not metadata.media_url:
                raise RuntimeError(
                    "未能从 B 站页面提取到音频流地址，可能是该视频为付费/充电专属或触发了验证。可尝试运行 `uv run video-to-text login -p bilibili` 登录 B 站账号。"
                )

            raw_media_path = self._download_stream(
                media_url=metadata.media_url,
                video_id=metadata.aweme_id,
                referer="https://www.bilibili.com/",
                backup_urls=metadata.backup_media_urls,
                ext=".m4s",
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

    def _wait_for_bilibili_data(
        self, page: Page, captured_playinfo: dict[str, Any]
    ) -> None:
        """等待 B 站页面的 window.__playinfo__ 或播放地址接口就绪。"""
        deadline = time.time() + 12.0
        while time.time() < deadline:
            if captured_playinfo:
                return
            try:
                has_playinfo = page.evaluate(
                    "() => Boolean(window.__playinfo__ && window.__playinfo__.data)"
                )
                if has_playinfo:
                    return
            except Exception:
                pass
            page.wait_for_timeout(400)

    def _build_bilibili_metadata(
        self,
        source_url: str,
        page: Page,
        captured_playinfo: dict[str, Any],
        captured_m4s_urls: list[str],
    ) -> VideoMetadata:
        """从 B 站页面的 window.__INITIAL_STATE__ 与 window.__playinfo__ 构建元数据。"""
        final_url = page.url
        page_num = extract_bilibili_page_num(final_url)
        if page_num == 1:
            page_num = extract_bilibili_page_num(source_url)

        initial_state: dict[str, Any] = {}
        playinfo: dict[str, Any] = {}
        try:
            state_and_play = page.evaluate(
                """() => ({
                    initialState: window.__INITIAL_STATE__ || {},
                    playinfo: window.__playinfo__ || {}
                })"""
            )
            if isinstance(state_and_play, dict):
                initial_state = state_and_play.get("initialState") or {}
                playinfo = state_and_play.get("playinfo") or {}
        except Exception:
            pass

        if not playinfo and captured_playinfo:
            playinfo = captured_playinfo

        meta = self._parse_bilibili_state_and_playinfo(
            source_url=source_url,
            final_url=final_url,
            page_num=page_num,
            initial_state=initial_state,
            playinfo=playinfo,
        )

        # 若 window.__playinfo__ 为空对象或未包含有效音轨，自动回退尝试网络拦截到的 captured_playinfo
        if not meta.media_url and captured_playinfo and captured_playinfo is not playinfo:
            fallback_url, fallback_backups = self._extract_bilibili_audio_urls(
                captured_playinfo
            )
            if fallback_url:
                meta.media_url = fallback_url
                meta.backup_media_urls = fallback_backups

        # DOM 兜底补充标题与 UP 主名称
        if not meta.title or meta.title.startswith("B站视频_"):
            try:
                dom_title = page.evaluate(
                    """() => {
                        const h1 = document.querySelector('h1.video-title, .video-title');
                        return h1 && h1.textContent ? h1.textContent.trim() : document.title.replace(/_哔哩哔哩.*$/, '').trim();
                    }"""
                )
                if dom_title:
                    meta.title = dom_title[:60]
            except Exception:
                pass

        if meta.author == "未知UP主":
            try:
                dom_author = page.evaluate(
                    """() => {
                        const el = document.querySelector('.up-name, .username');
                        return el && el.textContent ? el.textContent.trim() : '';
                    }"""
                )
                if dom_author:
                    meta.author = dom_author
            except Exception:
                pass

        if not meta.media_url and captured_m4s_urls:
            meta.media_url = captured_m4s_urls[0]

        return meta

    @classmethod
    def _parse_bilibili_state_and_playinfo(
        cls,
        source_url: str,
        final_url: str,
        page_num: int,
        initial_state: dict[str, Any],
        playinfo: dict[str, Any],
    ) -> VideoMetadata:
        """纯函数解析 B 站 __INITIAL_STATE__ 与 __playinfo__ 数据结构（便于单元测试验证）。"""
        video_data = initial_state.get("videoData") or {}
        bvid = (
            str(video_data.get("bvid") or initial_state.get("bvid") or "").strip()
            or extract_video_id_from_url(final_url)
            or extract_video_id_from_url(source_url)
            or hashlib.md5(source_url.encode("utf-8")).hexdigest()[:12]
        )
        # 确保分 P 后缀正确附加且不重复
        base_bvid = bvid.split("_p")[0]
        video_id = f"{base_bvid}_p{page_num}" if page_num > 1 else base_bvid

        main_title = str(video_data.get("title") or "").strip()
        # 检查多 P 子标题
        pages = video_data.get("pages") or []
        part_title = ""
        if isinstance(pages, list) and len(pages) > 1:
            for p_item in pages:
                if isinstance(p_item, dict) and p_item.get("page") == page_num:
                    part_title = str(p_item.get("part") or "").strip()
                    break

        if main_title and part_title and part_title != main_title:
            full_title = f"{main_title} (P{page_num} {part_title})"
        elif main_title:
            full_title = (
                f"{main_title} (P{page_num})" if page_num > 1 else main_title
            )
        else:
            full_title = f"B站视频_{video_id}"

        owner = video_data.get("owner") or initial_state.get("upData") or {}
        author = str(owner.get("name") or "").strip() or "未知UP主"

        pubdate = video_data.get("pubdate") or video_data.get("ctime")
        create_time_str = ""
        if isinstance(pubdate, (int, float)) and pubdate > 0:
            create_time_str = datetime.fromtimestamp(pubdate).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        desc = str(video_data.get("desc") or "").strip()
        tags = initial_state.get("tags") or []
        tag_names: list[str] = []
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, dict):
                    t_name = str(t.get("tag_name") or "").strip()
                    if t_name and not t_name.startswith("#"):
                        tag_names.append(f"#{t_name}")
        tags_line = " ".join(tag_names)
        if desc and tags_line:
            full_desc = f"{desc}\n{tags_line}"
        elif tags_line:
            full_desc = f"{full_title}\n{tags_line}"
        else:
            full_desc = desc or full_title

        primary_url, backup_urls = cls._extract_bilibili_audio_urls(playinfo)

        return VideoMetadata(
            source_url=source_url,
            aweme_id=video_id,
            title=full_title[:65],
            author=author,
            platform="bilibili",
            description=full_desc,
            create_time=create_time_str,
            media_url=primary_url,
            backup_media_urls=backup_urls,
        )

    @staticmethod
    def _extract_bilibili_audio_urls(
        playinfo: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """从 B 站 __playinfo__ 中提取最佳音质 DASH 纯音频流 baseUrl 与 backupUrl 列表。"""
        data = playinfo.get("data") or playinfo.get("result") or {}
        dash = data.get("dash") or {}
        audio_list = dash.get("audio") or []
        if isinstance(audio_list, list) and audio_list:
            # 按带宽/码率从高到低排序，优先选音质最好的流
            sorted_audios = sorted(
                (a for a in audio_list if isinstance(a, dict)),
                key=lambda x: int(x.get("bandwidth") or 0),
                reverse=True,
            )
            for audio_item in sorted_audios:
                base_url = str(
                    audio_item.get("baseUrl") or audio_item.get("base_url") or ""
                ).strip()
                backups_raw = (
                    audio_item.get("backupUrl")
                    or audio_item.get("backup_url")
                    or []
                )
                backups = [str(u).strip() for u in backups_raw if u]
                if base_url:
                    return base_url, backups
                if backups:
                    return backups[0], backups[1:]

        # 兜底支持老式 durl MP4 流
        durl_list = data.get("durl") or []
        if isinstance(durl_list, list) and durl_list:
            first_durl = durl_list[0]
            if isinstance(first_durl, dict):
                primary = str(first_durl.get("url") or "").strip()
                backups = [
                    str(u).strip()
                    for u in (first_durl.get("backup_url") or [])
                    if u
                ]
                return primary, backups

        return "", []

    def _wait_for_video_data(
        self, page: Page, captured_detail: dict[str, Any]
    ) -> None:
        """等待抖音详情数据或页面 <video> 标签加载完成。"""
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
        """综合 API 拦截数据、RENDER_DATA 与 DOM 提取抖音视频元信息。"""
        final_url = page.url
        aweme_id = (
            str(captured_detail.get("aweme_id") or "")
            or extract_video_id_from_url(final_url)
            or extract_video_id_from_url(source_url)
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
            platform="douyin",
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

    def _download_stream(
        self,
        media_url: str,
        video_id: str,
        referer: str = "https://www.douyin.com/",
        backup_urls: list[str] | None = None,
        ext: str = ".mp4",
    ) -> Path:
        """携带当前浏览器 Cookie 与指定 Referer 流式下载音视频（支持主备地址自动切换）。"""
        assert self._context is not None
        cookies = {c["name"]: c["value"] for c in self._context.cookies()}
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": referer,
            "Accept": "*/*",
        }
        target_path = self.temp_dir / f"{video_id}{ext}"
        candidate_urls = [media_url, *(backup_urls or [])]

        last_exc: Exception | None = None
        with httpx.Client(
            headers=headers,
            cookies=cookies,
            follow_redirects=True,
            timeout=60.0,
        ) as client:
            for candidate in candidate_urls:
                if not candidate:
                    continue
                try:
                    with client.stream("GET", candidate) as response:
                        response.raise_for_status()
                        total_bytes_header = response.headers.get("content-length")
                        total_bytes = (
                            int(total_bytes_header)
                            if total_bytes_header and total_bytes_header.isdigit()
                            else None
                        )
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[cyan]{task.description}[/cyan]"),
                            BarColumn(bar_width=32),
                            TaskProgressColumn(),
                            DownloadColumn(),
                            TransferSpeedColumn(),
                            TimeRemainingColumn(),
                            console=console,
                            transient=False,
                        ) as progress:
                            task_id = progress.add_task(
                                "   ↓ 下载媒体流", total=total_bytes
                            )
                            with target_path.open("wb") as f:
                                for chunk in response.iter_bytes(chunk_size=65536):
                                    f.write(chunk)
                                    progress.update(task_id, advance=len(chunk))
                    return target_path
                except Exception as exc:
                    last_exc = exc
                    target_path.unlink(missing_ok=True)
                    continue

        raise RuntimeError(f"媒体流下载失败: {last_exc}")
