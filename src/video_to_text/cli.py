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
from .parser import extract_aweme_id_from_url, extract_douyin_urls, extract_urls_from_file
from .transcriber import SenseVoiceTranscriber

app = typer.Typer(
    name="video-to-text",
    help="批量将抖音分享链接/口令转换为 Markdown 文本 (Playwright 持久化登录 + SenseVoiceSmall + LLM 保真校对)",
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
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="是否在语音转写后调用大模型进行保真错别字校对（需在 .env 配置 LLM_API_KEY）",
        ),
    ] = True,
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
                    console.print(
                        f"   [green]✓ 语音识别完成 ({result.elapsed_seconds:.2f}s)[/green]"
                    )

                    if llm_checker is not None:
                        console.print(
                            f"3. 正在调用大模型 ([bold]{llm_config.model}[/bold]) 进行保真错别字校对..."
                        )
                        try:
                            llm_checker.check_result(result)
                            console.print(
                                f"   [green]✓ AI 保真校对完成 ({result.llm_elapsed_seconds:.2f}s)[/green]"
                            )
                        except Exception as llm_exc:
                            console.print(
                                f"   [yellow]⚠ 大模型校对失败，已自动降级保留原始识别文本: {llm_exc}[/yellow]"
                            )

                    md_path = save_markdown(result, output_dir=output_dir)
                    console.print(f"   [bold green]✓ 已生成文稿:[/bold green] {md_path}")
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
    """对已生成的 Markdown 文稿单独执行大模型保真错别字校对（无需重新下载视频和语音识别）。"""
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

    console.print(
        f"[bold cyan]准备使用模型 [bold]{llm_config.model}[/bold] 对 {len(md_files)} 个 Markdown 文件执行保真校对...[/bold cyan]"
    )

    for idx, md_path in enumerate(md_files, start=1):
        console.rule(f"[cyan]校对进度 ({idx}/{len(md_files)}) - {md_path.name}[/cyan]")
        raw_md = md_path.read_text(encoding="utf-8")
        if not force and is_markdown_already_checked(raw_md):
            console.print(
                f"[yellow]↷ 该文稿已包含 AI 校对版，跳过（如需重校请加 --force）[/yellow]"
            )
            skipped_count += 1
            continue

        try:
            result = parse_markdown_for_check(md_path)
            if not result.formatted_text.strip():
                console.print("[yellow]↷ 该文稿无有效人声识别内容，跳过校对[/yellow]")
                skipped_count += 1
                continue
            checker.check_result(result)
            md_path.write_text(render_markdown(result), encoding="utf-8")
            console.print(
                f"[bold green]✓ 校对并更新成功 ({result.llm_elapsed_seconds:.2f}s):[/bold green] {md_path}"
            )
            checked_count += 1
        except Exception as exc:
            console.print(f"[bold red]✗ 校对失败:[/bold red] {exc}")
            failed_count += 1

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
