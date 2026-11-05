from __future__ import annotations

import contextlib
import io
import re
import time
from collections.abc import Callable
from pathlib import Path

import torch
from rich.console import Console

from .models import TranscriptResult, VideoMetadata

console = Console()

# 匹配 SenseVoice 特殊标签，如 <|zh|> <|NEUTRAL|> <|Speech|> <|woitn|>
SENSEVOICE_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")


def clean_sensevoice_text(raw_text: str) -> str:
    """移除 SenseVoice 输出的控制标签并整理自然段落。"""
    cleaned = SENSEVOICE_TAG_PATTERN.sub("", raw_text).strip()
    # 合并多余空白
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return format_paragraphs(cleaned)


def format_paragraphs(text: str, max_chars_per_para: int = 160) -> str:
    """按句号/问号/感叹号将长段文字切分为易读的 Markdown 段落。"""
    if not text:
        return ""
    # 在句末标点后切分句子
    sentences = re.split(r"(?<=[。！？!?])\s*", text)
    paragraphs: list[str] = []
    current_para: list[str] = []
    current_len = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        current_para.append(s)
        current_len += len(s)
        if current_len >= max_chars_per_para:
            paragraphs.append("".join(current_para))
            current_para = []
            current_len = 0

    if current_para:
        paragraphs.append("".join(current_para))

    return "\n\n".join(paragraphs)


def detect_best_device() -> str:
    """检测当前环境最佳推理设备（优先使用 Apple Silicon MPS 或 CUDA，回退 CPU）。"""
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SenseVoiceTranscriber:
    """基于阿里 FunASR (iic/SenseVoiceSmall) 的本地语音转文字引擎。"""

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: str | None = None,
        post_process_hook: Callable[[str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or detect_best_device()
        # 预留后处理钩子（方便后续接入 LLM 润色）
        self.post_process_hook = post_process_hook
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        console.print(
            f"[cyan]正在加载阿里语音识别模型 [bold]{self.model_id}[/bold] (设备: {self.device}，首次运行会自动下载模型权重，请稍候)...[/cyan]"
        )
        # 屏蔽 funasr 导入时打印的无害 Notice: ffmpeg is not installed 提示
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
        try:
            self._model = AutoModel(
                model=self.model_id,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000, "device": "cpu"},
                device=self.device,
                disable_update=True,
            )
        except Exception:
            # 如果 MPS 设备在部分算子上不兼容，自动回退到 CPU
            if self.device != "cpu":
                console.print(
                    f"[yellow]设备 {self.device} 初始化回退至 cpu...[/yellow]"
                )
                self.device = "cpu"
                self._model = AutoModel(
                    model=self.model_id,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000, "device": "cpu"},
                    device="cpu",
                    disable_update=True,
                )
            else:
                raise

    def transcribe(self, metadata: VideoMetadata) -> TranscriptResult:
        """对单个视频提取出的 16kHz WAV 音频进行语音转文字。"""
        if not metadata.audio_path or not Path(metadata.audio_path).exists():
            raise FileNotFoundError(
                f"找不到待识别的音频文件: {metadata.audio_path}"
            )

        self._ensure_model_loaded()
        assert self._model is not None

        start_time = time.perf_counter()
        generate_kwargs = {
            "input": str(metadata.audio_path),
            "cache": {},
            "language": "auto",
            "use_itn": True,
            "batch_size_s": 300,
            "merge_vad": True,
            "merge_length_s": 15,
        }
        try:
            res = self._model.generate(**generate_kwargs)
        except Exception:
            if self.device != "cpu":
                console.print(
                    f"[yellow]推理阶段设备 {self.device} 算子不兼容，正在回退至 cpu 重试...[/yellow]"
                )
                self.device = "cpu"
                self._model = None
                self._ensure_model_loaded()
                assert self._model is not None
                res = self._model.generate(**generate_kwargs)
            else:
                raise
        elapsed = time.perf_counter() - start_time

        raw_text = ""
        if isinstance(res, list) and res:
            raw_text = str(res[0].get("text") or "")

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            processed = rich_transcription_postprocess(raw_text)
            formatted_text = format_paragraphs(SENSEVOICE_TAG_PATTERN.sub("", processed).strip())
        except Exception:
            formatted_text = clean_sensevoice_text(raw_text)

        if self.post_process_hook is not None and formatted_text:
            formatted_text = self.post_process_hook(formatted_text)

        return TranscriptResult(
            metadata=metadata,
            raw_text=raw_text,
            formatted_text=formatted_text,
            elapsed_seconds=elapsed,
        )
