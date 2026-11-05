from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .downloader import DouyinDownloader, open_login_session
from .exporter import (
    find_existing_markdown,
    is_markdown_already_checked,
    parse_markdown_for_check,
    render_markdown,
    save_markdown,
)
from .llm_checker import LLMChecker, LLMConfig
from .models import TranscriptResult, VideoMetadata
from .parser import (
    detect_platform,
    extract_urls_from_file,
    extract_video_id_from_url,
    extract_video_urls,
)
from .transcriber import SenseVoiceTranscriber

app = typer.Typer(
    name="video-to-text",
    help="批量将抖音与 B 站分享链接/口令转换为 Markdown 文本 (Playwright 持久化登录 + SenseVoiceSmall + LLM 保真校对)",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("login")
def login_cmd(
    platform: Annotated[
        str,
        typer.Option(
            "--platform",
            "-p",
            help="要登录的平台：all (同时打开抖音和B站)、douyin 或 bilibili",
        ),
    ] = "all",
    profile_dir: Annotated[
        Path,
        typer.Option(
            "--profile-dir",
            help="Playwright 持久化浏览器配置目录",
        ),
    ] = Path(".browser_profile"),
) -> None:
    """打开浏览器窗口扫码登录抖音和/或 B 站，永久保存登录状态到 .browser_profile 目录。"""
    open_login_session(profile_dir=profile_dir.resolve(), platform=platform)


@app.command("run")
def run_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="包含抖音或 B 站分享口令/链接的文本文件路径",
        ),
    ] = Path("links.txt"),
    url: Annotated[
        list[str] | None,
        typer.Option(
            "--url",
            "-u",
            help="直接传入单条或多条抖音/B站分享口令或链接（可多次指定）",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Markdown 文稿输出目录",
        ),
    ] = Path("output"),
    profile_dir: Annotated[
        Path,
        typer.Option(
            "--profile-dir",
            help="Playwright 持久化浏览器配置目录",
        ),
    ] = Path(".browser_profile"),
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--no-headless",
            help="是否使用无头模式运行浏览器（遇到验证码时可加 --no-headless 查看）",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="强制重新下载并转换已存在的文件",
        ),
    ] = False,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="指定推理设备 (如 mps / cpu / cuda:0，默认自动检测)",
        ),
    ] = None,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="是否在语音转写后调用大模型进行保真错别字校对（需在 .env 配置 LLM_API_KEY）",
        ),
    ] = True,
) -> None:
    """批量读取抖音或 B 站分享链接/口令，提取音频并转换为独立 Markdown 文件。"""
    urls: list[str] = []
    if url:
        for item in url:
            urls.extend(extract_video_urls(item))
    elif input_file.exists():
        urls.extend(extract_urls_from_file(input_file))
    else:
        console.print(
            f"[bold red]未找到链接文件 {input_file}，且未通过 -u 传入链接。[/bold red]\n"
            f"请将抖音或 B 站分享口令粘贴到 [cyan]{input_file}[/cyan] 中，或使用 [cyan]uv run video-to-text run -u '分享链接'[/cyan]。"
        )
        raise typer.Exit(code=1)

    # 去重
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        console.print(
            "[yellow]未在输入中提取到任何有效的抖音或 B 站链接，请检查 links.txt 内容。[/yellow]"
        )
        raise typer.Exit(code=0)

    console.print(
        f"[bold green]共提取到 {len(unique_urls)} 条视频链接，准备开始批量处理...[/bold green]"
    )

    llm_config = LLMConfig.from_env()
    llm_checker: LLMChecker | None = None
    if llm:
        if llm_config.is_configured:
            llm_checker = LLMChecker(llm_config)
            console.print(
                f"[cyan]✓ 已启用大模型保真校对 (模型: [bold]{llm_config.model}[/bold])[/cyan]"
            )
        else:
            console.print(
                "[dim]ℹ 未检测到 .env 中的 LLM_API_KEY 配置，将仅输出 SenseVoiceSmall 原始识别文本（如需开启 AI 校对，请配置 .env）。[/dim]"
            )

    temp_dir = Path(".tmp_media").resolve()
    output_dir = output_dir.resolve()
    profile_dir = profile_dir.resolve()

    transcriber = SenseVoiceTranscriber(device=device)

    success_items: list[tuple[str, str, Path]] = []
    skipped_items: list[tuple[str, Path]] = []
    failed_items: list[tuple[str, str]] = []
    downloaded_tasks: list[VideoMetadata] = []
    seen_video_ids: set[str] = set()

    try:
        # ==================== 阶段一：批量解析视频并提取音频 ====================
        console.rule("[bold magenta]阶段一：批量解析视频信息并提取音频[/bold magenta]")
        with DouyinDownloader(
            profile_dir=profile_dir,
            temp_dir=temp_dir,
            headless=headless,
        ) as downloader:
            for idx, item_url in enumerate(unique_urls, start=1):
                platform_tag = (
                    "B站" if detect_platform(item_url) == "bilibili" else "抖音"
                )
                console.rule(
                    f"[bold cyan]下载进度 ({idx}/{len(unique_urls)}) [{platform_tag}][/bold cyan]"
                )
                console.print(f"[dim]原始链接:[/dim] {item_url}")

                # 如果链接本身已带作品 ID，先检查是否在本批次中重复或已转换过
                url_video_id = extract_video_id_from_url(item_url)
                if url_video_id and url_video_id in seen_video_ids:
                    console.print(
                        f"[yellow]↷ 当前批次中已包含相同作品 ID ({url_video_id})，跳过重复链接[/yellow]"
                    )
                    continue
                if not force and url_video_id:
                    existing_md = find_existing_markdown(output_dir, url_video_id)
                    if existing_md:
                        console.print(
                            f"[yellow]↷ 已存在转换结果，跳过下载:[/yellow] {existing_md.name}"
                        )
                        skipped_items.append((item_url, existing_md))
                        seen_video_ids.add(url_video_id)
                        continue

                try:
                    console.print(
                        f"正在通过 Playwright 解析{platform_tag}视频信息并提取音频..."
                    )
                    metadata = downloader.process_url(item_url)

                    # 解析出真实作品 ID 后检查是否与当前批次之前的任务重复（如短链与长链指向同一视频）
                    if metadata.aweme_id in seen_video_ids:
                        console.print(
                            f"[yellow]↷ 当前批次中已包含相同作品 ID ({metadata.aweme_id})，跳过重复任务[/yellow]"
                        )
                        continue

                    # 检查输出目录是否已存在该作品 ID 的文稿（针对短链重定向后得到 ID 的情况）
                    if not force and metadata.aweme_id:
                        existing_md = find_existing_markdown(
                            output_dir, metadata.aweme_id
                        )
                        if existing_md:
                            console.print(
                                f"[yellow]↷ 已存在转换结果，跳过:[/yellow] {existing_md.name}"
                            )
                            if metadata.audio_path and metadata.audio_path.exists():
                                metadata.audio_path.unlink(missing_ok=True)
                            skipped_items.append((item_url, existing_md))
                            seen_video_ids.add(metadata.aweme_id)
                            continue

                    if metadata.aweme_id:
                        seen_video_ids.add(metadata.aweme_id)
                    console.print(
                        f"   [green]✓ 音频就绪:[/green] [bold]{metadata.author}[/bold] - {metadata.title}"
                    )
                    downloaded_tasks.append(metadata)
                except Exception as exc:
                    console.print(f"   [bold red]✗ 下载/解析失败:[/bold red] {exc}")
                    failed_items.append((item_url, str(exc)))

        # 此时 Playwright 浏览器上下文已自动关闭并释放内存
        # ==================== 阶段二：统一加载模型并批量语音转文字 ====================
        transcribed_items: list[tuple[TranscriptResult, Path]] = []
        if downloaded_tasks:
            console.rule(
                f"[bold magenta]阶段二：批量语音转文字 (共 {len(downloaded_tasks)} 个音频)[/bold magenta]"
            )
            for idx, metadata in enumerate(downloaded_tasks, start=1):
                console.rule(
                    f"[bold cyan]语音识别进度 ({idx}/{len(downloaded_tasks)}) - {metadata.author}[/bold cyan]"
                )
                console.print(f"[bold]{metadata.title}[/bold]")
                audio_path = metadata.audio_path
                try:
                    result = transcriber.transcribe(metadata)
                    # 先将原始识别结果落盘保存为 .md（即使后续跳过或中断，识别初稿也不会丢失）
                    md_path = save_markdown(result, output_dir=output_dir)
                    console.print(
                        f"   [green]✓ 语音识别完成 ({result.elapsed_seconds:.2f}s)，初稿已保存:[/green] {md_path.name}"
                    )
                    transcribed_items.append((result, md_path))
                except Exception as exc:
                    console.print(f"   [bold red]✗ 转写失败:[/bold red] {exc}")
                    failed_items.append((metadata.source_url, str(exc)))
                finally:
                    if audio_path and audio_path.exists():
                        audio_path.unlink(missing_ok=True)

        # ==================== 阶段三：全局大并发大模型保真校对 ====================
        if transcribed_items and llm_checker is not None:
            console.rule(
                f"[bold magenta]阶段三：全局并发大模型保真校对 (共 {len(transcribed_items)} 篇文稿, 模型: {llm_config.model})[/bold magenta]"
            )
            results_only = [item[0] for item in transcribed_items]
            batch_errors = llm_checker.check_batch(results_only)

            for (result, md_path), err in zip(
                transcribed_items, batch_errors.values()
            ):
                if err is not None:
                    console.print(
                        f"   [yellow]⚠ 《{result.metadata.title}》AI 校对失败，已保留原始识别初稿: {err}[/yellow]"
                    )
                else:
                    md_path = save_markdown(result, output_dir=output_dir)
                    console.print(
                        f"   [bold green]✓ 已更新 AI 校对版:[/bold green] {md_path}"
                    )
                success_items.append(
                    (result.metadata.author, result.metadata.title, md_path)
                )
        else:
            for result, md_path in transcribed_items:
                success_items.append(
                    (result.metadata.author, result.metadata.title, md_path)
                )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    _print_summary(success_items, skipped_items, failed_items)


@app.command("check")
def check_cmd(
    target: Annotated[
        Path,
        typer.Argument(
            help="待校对的 Markdown 文件路径或包含 .md 文件的目录（默认 ./output）",
        ),
    ] = Path("output"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="对已经包含 AI 校对版的内容强制重新校对",
        ),
    ] = False,
) -> None:
    """对已生成的 Markdown 文稿批量执行全局并发大模型保真校对（无需重新下载视频和语音识别）。"""
    llm_config = LLMConfig.from_env()
    if not llm_config.is_configured:
        console.print(
            "[bold red]未检测到有效的大模型配置！[/bold red]\n"
            "请先复制配置文件：[cyan]cp .env.example .env[/cyan]，并在 [cyan].env[/cyan] 中填入 `LLM_API_KEY`。"
        )
        raise typer.Exit(code=1)

    target = target.resolve()
    if target.is_file():
        md_files = [target]
    elif target.is_dir():
        md_files = sorted(target.glob("*.md"))
    else:
        console.print(f"[bold red]找不到目标文件或目录: {target}[/bold red]")
        raise typer.Exit(code=1)

    if not md_files:
        console.print(f"[yellow]在 {target} 中未找到任何 .md 文件。[/yellow]")
        raise typer.Exit(code=0)

    checker = LLMChecker(llm_config)
    checked_count = 0
    skipped_count = 0
    failed_count = 0
    pending_items: list[tuple[TranscriptResult, Path]] = []

    for md_path in md_files:
        raw_md = md_path.read_text(encoding="utf-8")
        if not force and is_markdown_already_checked(raw_md):
            console.print(
                f"[yellow]↷ 跳过已校对文稿 (加 --force 可重校):[/yellow] {md_path.name}"
            )
            skipped_count += 1
            continue

        try:
            result = parse_markdown_for_check(md_path)
            if not result.formatted_text.strip():
                console.print(
                    f"[yellow]↷ 跳过无有效人声识别内容文稿:[/yellow] {md_path.name}"
                )
                skipped_count += 1
                continue
            pending_items.append((result, md_path))
        except Exception as exc:
            console.print(f"[bold red]✗ 解析失败 ({md_path.name}):[/bold red] {exc}")
            failed_count += 1

    if pending_items:
        console.rule(
            f"[bold cyan]开始全局并发校对 (共 {len(pending_items)} 篇文稿, 模型: {llm_config.model})[/bold cyan]"
        )
        results_only = [item[0] for item in pending_items]
        batch_errors = checker.check_batch(results_only)
        for (result, md_path), err in zip(pending_items, batch_errors.values()):
            if err is not None:
                console.print(f"[bold red]✗ 校对失败 ({md_path.name}):[/bold red] {err}")
                failed_count += 1
            else:
                md_path.write_text(render_markdown(result), encoding="utf-8")
                console.print(
                    f"[bold green]✓ 校对并更新成功:[/bold green] {md_path}"
                )
                checked_count += 1

    console.rule("[bold]AI 校对汇总[/bold]")
    console.print(
        f"[green]成功: {checked_count}[/green] | [yellow]跳过: {skipped_count}[/yellow] | [red]失败: {failed_count}[/red]"
    )


def _print_summary(
    success_items: list[tuple[str, str, Path]],
    skipped_items: list[tuple[str, Path]],
    failed_items: list[tuple[str, str]],
) -> None:
    console.rule("[bold]批量转换汇总[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("状态", width=10)
    table.add_column("数量", justify="right", width=8)
    table.add_column("说明")
    table.add_row("[green]成功[/green]", str(len(success_items)), "已生成独立 .md 文稿")
    table.add_row("[yellow]跳过[/yellow]", str(len(skipped_items)), "已存在同名作品 ID 文稿")
    table.add_row("[red]失败[/red]", str(len(failed_items)), "解析或下载异常")
    console.print(table)

    if failed_items:
        console.print("\n[bold red]失败详情：[/bold red]")
        for u, err in failed_items:
            console.print(f" - {u}: {err}")


def main_entry() -> None:
    app()
