from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .downloader import DouyinDownloader, open_login_session
from .exporter import find_existing_markdown, save_markdown
from .parser import extract_aweme_id_from_url, extract_douyin_urls, extract_urls_from_file
from .transcriber import SenseVoiceTranscriber

app = typer.Typer(
    name="video-to-text",
    help="批量将抖音分享链接/口令转换为 Markdown 文本 (Playwright 持久化登录 + 阿里 SenseVoiceSmall)",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("login")
def login_cmd(
    profile_dir: Annotated[
        Path,
        typer.Option(
            "--profile-dir",
            help="Playwright 持久化浏览器配置目录",
        ),
    ] = Path(".browser_profile"),
) -> None:
    """打开浏览器窗口扫码登录抖音，永久保存登录状态到 .browser_profile 目录。"""
    open_login_session(profile_dir=profile_dir.resolve())


@app.command("run")
def run_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="包含抖音分享口令或链接的文本文件路径",
        ),
    ] = Path("links.txt"),
    url: Annotated[
        list[str] | None,
        typer.Option(
            "--url",
            "-u",
            help="直接传入单条或多条抖音分享口令/链接（可多次指定）",
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
) -> None:
    """批量读取抖音分享链接/口令，提取音频并转换为独立 Markdown 文件。"""
    urls: list[str] = []
    if url:
        for item in url:
            urls.extend(extract_douyin_urls(item))
    elif input_file.exists():
        urls.extend(extract_urls_from_file(input_file))
    else:
        console.print(
            f"[bold red]未找到链接文件 {input_file}，且未通过 -u 传入链接。[/bold red]\n"
            f"请将抖音分享口令粘贴到 [cyan]{input_file}[/cyan] 中，或使用 [cyan]uv run video-to-text run -u '分享口令'[/cyan]。"
        )
        raise typer.Exit(code=1)

    # 去重
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        console.print("[yellow]未在输入中提取到任何有效的抖音链接，请检查 links.txt 内容。[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"[bold green]共提取到 {len(unique_urls)} 条抖音链接，准备开始批量处理...[/bold green]"
    )

    temp_dir = Path(".tmp_media").resolve()
    output_dir = output_dir.resolve()
    profile_dir = profile_dir.resolve()

    transcriber = SenseVoiceTranscriber(device=device)

    success_items: list[tuple[str, str, Path]] = []
    skipped_items: list[tuple[str, Path]] = []
    failed_items: list[tuple[str, str]] = []

    try:
        with DouyinDownloader(
            profile_dir=profile_dir,
            temp_dir=temp_dir,
            headless=headless,
        ) as downloader:
            for idx, item_url in enumerate(unique_urls, start=1):
                console.rule(f"[bold cyan]处理进度 ({idx}/{len(unique_urls)})[/bold cyan]")
                console.print(f"[dim]原始链接:[/dim] {item_url}")

                # 如果链接本身已带 aweme_id，先检查是否已转换过
                url_aweme_id = extract_aweme_id_from_url(item_url)
                if not force and url_aweme_id:
                    existing_md = find_existing_markdown(output_dir, url_aweme_id)
                    if existing_md:
                        console.print(
                            f"[yellow]↷ 已存在转换结果，跳过:[/yellow] {existing_md.name}"
                        )
                        skipped_items.append((item_url, existing_md))
                        continue

                audio_path: Path | None = None
                try:
                    console.print("1. 正在通过 Playwright 解析视频信息并提取音频...")
                    metadata = downloader.process_url(item_url)
                    audio_path = metadata.audio_path

                    # 解析出真实 aweme_id 后再次检查是否已存在（针对短链重定向后得到 ID 的情况）
                    if not force and metadata.aweme_id:
                        existing_md = find_existing_markdown(output_dir, metadata.aweme_id)
                        if existing_md:
                            console.print(
                                f"[yellow]↷ 已存在转换结果，跳过:[/yellow] {existing_md.name}"
                            )
                            skipped_items.append((item_url, existing_md))
                            continue

                    console.print(
                        f"   [green]✓ 解析成功:[/green] [bold]{metadata.author}[/bold] - {metadata.title}"
                    )
                    console.print("2. 正在使用 SenseVoiceSmall 进行语音转文字...")
                    result = transcriber.transcribe(metadata)
                    md_path = save_markdown(result, output_dir=output_dir)
                    console.print(
                        f"   [bold green]✓ 已生成文稿 ({result.elapsed_seconds:.2f}s):[/bold green] {md_path}"
                    )
                    success_items.append((metadata.author, metadata.title, md_path))
                except Exception as exc:
                    console.print(f"   [bold red]✗ 处理失败:[/bold red] {exc}")
                    failed_items.append((item_url, str(exc)))
                finally:
                    # 及时清理当前视频的临时音频文件
                    if audio_path and audio_path.exists():
                        audio_path.unlink(missing_ok=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    _print_summary(success_items, skipped_items, failed_items)


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
