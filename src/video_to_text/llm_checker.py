from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .models import TranscriptResult, VideoMetadata

# 屏蔽 httpx / httpcore 默认打印的 INFO HTTP Request 日志，避免干扰进度条显示
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

console = Console()

SYSTEM_PROMPT = """你是一位专业的中文速记与文稿校对专家。
用户将提供一段短视频的语音识别（ASR）初稿，以及该视频的标题、作者和简介标签作为背景参考。
请对语音识别初稿进行【保真校对】，严格遵守以下规则：
1. 仅修正同音错别字、专有名词错误（如历史人名、朝代、地名、术语、品牌名）以及明显的标点符号或断句错误（例如句号断在了半句话中间）。
2. 严格保留原作者的口语风格、语气词、叙事顺序和全部细节内容，严禁删减、缩写、总结或大幅改写句式。
3. 保持合理的自然段落划分（每段约 120~200 字），段落之间空一行，便于阅读。
4. 直接输出校对后的正文文本，不要输出任何开场白、解释说明、修改列表或 Markdown 标题。"""


def load_dotenv_file(dotenv_path: Path = Path(".env")) -> dict[str, str]:
    """轻量解析本地 .env 配置文件（不覆盖已存在的系统环境变量）。"""
    env_vars: dict[str, str] = {}
    if not dotenv_path.exists():
        return env_vars

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 若未用引号包裹，剥离行尾 # 注释
        if not (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value.split("#", 1)[0].strip()
        else:
            value = value[1:-1].strip()
        if key:
            env_vars[key] = value
    return env_vars


@dataclass
class LLMConfig:
    """OpenAI 兼容接口配置。"""

    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout_seconds: float = 90.0
    max_chunk_chars: int = 2400
    max_concurrency: int = 50

    @classmethod
    def from_env(cls, dotenv_path: Path = Path(".env")) -> LLMConfig:
        file_vars = load_dotenv_file(dotenv_path)

        def get_val(key: str, default: str = "") -> str:
            return (os.environ.get(key) or file_vars.get(key) or default).strip()

        raw_concurrency = get_val("LLM_MAX_CONCURRENCY", "50")
        try:
            concurrency = max(1, int(raw_concurrency))
        except ValueError:
            concurrency = 50

        return cls(
            api_key=get_val("LLM_API_KEY", ""),
            base_url=get_val("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
            model=get_val("LLM_MODEL", "deepseek-chat"),
            max_concurrency=concurrency,
        )

    @property
    def is_configured(self) -> bool:
        return bool(
            self.api_key
            and self.api_key != "sk-"
            and not self.api_key.startswith("sk-xxxx")
            and self.base_url
            and self.model
        )


def split_text_into_chunks(text: str, max_chars: int = 2400) -> list[str]:
    """按自然段落边界将长文本切分为不超过 max_chars 的分块，防止大模型截断或缩写。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paras: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_paras and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current_paras))
            current_paras = [para]
            current_len = len(para)
        else:
            current_paras.append(para)
            current_len += len(para) + (2 if len(current_paras) > 1 else 0)

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


class LLMChecker:
    """基于 OpenAI 兼容接口的大模型保真校对器（支持多分块并发请求）。"""

    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    def check_result(self, result: TranscriptResult) -> TranscriptResult:
        """对单个 TranscriptResult 进行大模型保真校对并填充 checked_text。"""
        errors = self.check_batch([result])
        if 0 in errors and errors[0] is not None:
            raise errors[0]
        return result

    def check_batch(
        self, results: list[TranscriptResult]
    ) -> dict[int, Exception | None]:
        """将多个视频的所有文本分块汇总到同一个全局并发池中同时校对（最高 max_concurrency 并发）。
        返回每个 result 下标对应的异常（成功则为 None）。
        """
        if not self.config.is_configured:
            raise RuntimeError(
                "未配置大模型 API Key，请在 .env 文件中设置 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL。"
            )

        # 收集所有视频的所有分块任务: (result_idx, chunk_idx, total_parts_of_video, chunk_text, metadata)
        all_tasks: list[tuple[int, int, int, str, VideoMetadata]] = []
        chunks_per_result: dict[int, int] = {}

        for r_idx, res in enumerate(results):
            clean_input = res.formatted_text.strip()
            if not clean_input:
                chunks_per_result[r_idx] = 0
                continue
            v_chunks = split_text_into_chunks(
                clean_input, max_chars=self.config.max_chunk_chars
            )
            chunks_per_result[r_idx] = len(v_chunks)
            for c_idx, chunk_str in enumerate(v_chunks, start=1):
                all_tasks.append(
                    (r_idx, c_idx, len(v_chunks), chunk_str, res.metadata)
                )

        if not all_tasks:
            return {i: None for i in range(len(results))}

        total_all_chunks = len(all_tasks)
        max_workers = max(1, min(self.config.max_concurrency, total_all_chunks))
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        def _process_chunk(
            client: httpx.Client,
            r_idx: int,
            c_idx: int,
            v_total: int,
            chunk_str: str,
            meta: VideoMetadata,
        ) -> tuple[int, int, str, float]:
            t0 = time.perf_counter()
            user_prompt = self._build_user_prompt(
                chunk=chunk_str,
                metadata=meta,
                part_index=c_idx,
                total_parts=v_total,
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"大模型返回结果为空: {data}")
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            ).strip()
            return r_idx, c_idx, (content or chunk_str), (time.perf_counter() - t0)

        # 按 (r_idx, c_idx) 存储结果，按 r_idx 记录错误与耗时
        chunk_outputs: dict[tuple[int, int], str] = {}
        elapsed_per_result: dict[int, float] = {i: 0.0 for i in range(len(results))}
        errors: dict[int, Exception | None] = {i: None for i in range(len(results))}

        with httpx.Client(
            headers=headers,
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(
                max_connections=max(100, max_workers * 2),
                max_keepalive_connections=max(20, max_workers),
            ),
            transport=self._transport,
        ) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}[/cyan]"),
                BarColumn(bar_width=32),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"   ⚡ 全局并发校对 ({len(results)} 篇文稿 / 共 {total_all_chunks} 块, 并发度: {max_workers})",
                    total=total_all_chunks,
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_key = {
                        executor.submit(
                            _process_chunk,
                            client,
                            r_idx,
                            c_idx,
                            v_total,
                            chunk_str,
                            meta,
                        ): (r_idx, c_idx)
                        for r_idx, c_idx, v_total, chunk_str, meta in all_tasks
                    }
                    for future in as_completed(future_to_key):
                        r_idx, c_idx = future_to_key[future]
                        try:
                            _, _, checked_str, chunk_elapsed = future.result()
                            chunk_outputs[(r_idx, c_idx)] = checked_str
                            elapsed_per_result[r_idx] = max(
                                elapsed_per_result[r_idx], chunk_elapsed
                            )
                        except Exception as exc:
                            if errors[r_idx] is None:
                                errors[r_idx] = exc
                        finally:
                            progress.update(task_id, advance=1)

        # 按每个视频的分块顺序组装结果
        for r_idx, res in enumerate(results):
            n_chunks = chunks_per_result.get(r_idx, 0)
            if n_chunks == 0 or errors[r_idx] is not None:
                continue
            assembled = [
                chunk_outputs[(r_idx, c_idx)] for c_idx in range(1, n_chunks + 1)
            ]
            res.checked_text = "\n\n".join(assembled)
            res.llm_model = self.config.model
            res.llm_elapsed_seconds = elapsed_per_result[r_idx]

        return errors

    def check_text(
        self,
        text: str,
        metadata: VideoMetadata | None = None,
    ) -> tuple[str, float]:
        """对给定语音识别文本结合视频元信息进行并发分段保真校对，返回 (校对后文本, 耗时秒数)。"""
        if not self.config.is_configured:
            raise RuntimeError(
                "未配置大模型 API Key，请在 .env 文件中设置 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL。"
            )

        clean_input = text.strip()
        if not clean_input:
            return "", 0.0

        chunks = split_text_into_chunks(
            clean_input, max_chars=self.config.max_chunk_chars
        )
        total_parts = len(chunks)

        start_time = time.perf_counter()
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        def _process_single_chunk(
            client: httpx.Client, idx: int, chunk: str
        ) -> tuple[int, str]:
            user_prompt = self._build_user_prompt(
                chunk=chunk,
                metadata=metadata,
                part_index=idx,
                total_parts=total_parts,
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"大模型返回结果为空: {data}")
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            ).strip()
            return idx, (content or chunk)

        ordered_results: dict[int, str] = {}
        max_workers = max(1, min(self.config.max_concurrency, total_parts))

        with httpx.Client(
            headers=headers,
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(
                max_connections=max(100, max_workers * 2),
                max_keepalive_connections=max(20, max_workers),
            ),
            transport=self._transport,
        ) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}[/cyan]"),
                BarColumn(bar_width=32),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"   ⚡ 并发校对分块 (并发度: {max_workers})",
                    total=total_parts,
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_process_single_chunk, client, idx, chunk)
                        for idx, chunk in enumerate(chunks, start=1)
                    ]
                    for future in as_completed(futures):
                        idx, checked_chunk = future.result()
                        ordered_results[idx] = checked_chunk
                        progress.update(task_id, advance=1)

        checked_chunks = [ordered_results[i] for i in range(1, total_parts + 1)]
        elapsed = time.perf_counter() - start_time
        return "\n\n".join(checked_chunks), elapsed

    @staticmethod
    def _build_user_prompt(
        chunk: str,
        metadata: VideoMetadata | None,
        part_index: int,
        total_parts: int,
    ) -> str:
        context_lines: list[str] = []
        if metadata:
            if metadata.title:
                context_lines.append(f"- 视频标题：{metadata.title}")
            if metadata.author:
                context_lines.append(f"- 视频作者：{metadata.author}")
            if metadata.description and metadata.description != metadata.title:
                context_lines.append(f"- 视频简介与标签：{metadata.description}")

        part_note = (
            f"（当前为第 {part_index}/{total_parts} 部分）" if total_parts > 1 else ""
        )
        context_block = (
            "【视频背景参考信息】\n" + "\n".join(context_lines) + "\n\n"
            if context_lines
            else ""
        )
        return (
            f"{context_block}"
            f"【待保真校对的语音识别初稿{part_note}】\n"
            f"{chunk}"
        )
