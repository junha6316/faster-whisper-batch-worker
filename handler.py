"""
RunPod Serverless Handler with process-level concurrency support

WhisperModel + ThreadPoolExecutor로 GPU 병렬 처리 최적화.
concurrency_modifier로 RunPod이 동시에 여러 Job을 Pull 가능.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from typing import Any

import runpod

from batch_transcriber import BatchTranscriber

# audio_url 다운로드 타임아웃 (초)
DOWNLOAD_TIMEOUT = int(os.environ.get("AUDIO_DOWNLOAD_TIMEOUT", "60"))

# beam search 기본값 (기존 batch_size 기본 16 → min(16, 5) = 5 와 동일한 실효값)
DEFAULT_BEAM_SIZE = 5

# beam_size 상한. 이 위로는 품질이 사실상 오르지 않고 디코딩 시간만 선형으로
# 늘어서 GPU 과금이 그대로 커진다.
MAX_BEAM_SIZE = 10

# deprecated batch_size 별칭에만 적용하는 상한. 예전 동작을 그대로 보존한다.
LEGACY_BATCH_SIZE_CAP = 5

SUPPORTED_OUTPUT_FORMATS = ("text", "srt", "vtt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 글로벌 transcriber 인스턴스 (cold start 시 한 번만 초기화)
transcriber: BatchTranscriber | None = None


def get_transcriber() -> BatchTranscriber:
    """Lazy initialization of transcriber"""
    global transcriber
    if transcriber is None:
        model_name = os.environ.get("WHISPER_MODEL", "turbo")
        logger.info("Initializing transcriber with model: %s", model_name)
        transcriber = BatchTranscriber(model_name=model_name)
    return transcriber


def _materialize_inputs(
    audio_b64_list: list[str],
    audio_urls: list[str],
) -> list[str]:
    """
    base64 / URL 입력을 임시 파일로 저장하고 경로 리스트를 반환한다.

    순서는 base64 입력 먼저, 그 다음 URL 입력이다.
    중간에 실패하면 이미 만든 임시 파일을 지우고 예외를 올린다.
    """
    temp_paths: list[str] = []

    try:
        for audio_b64 in audio_b64_list:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_paths.append(temp_file.name)
            temp_file.write(base64.b64decode(audio_b64))
            temp_file.close()

        for url in audio_urls:
            suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".wav"
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_paths.append(temp_file.name)
            try:
                with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
                    shutil.copyfileobj(response, temp_file)
            finally:
                temp_file.close()

        return temp_paths

    except Exception:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def _format_timestamp(seconds: float, decimal_marker: str) -> str:
    """초 단위 시간을 HH:MM:SS<decimal_marker>mmm 형태로 만든다."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def build_srt(segments: list[dict[str, Any]]) -> str:
    """세그먼트 리스트로 SRT 자막을 만든다 (HH:MM:SS,mmm --> HH:MM:SS,mmm)."""
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = _format_timestamp(segment["start"], ",")
        end = _format_timestamp(segment["end"], ",")
        text = segment["text"].strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n\n")
    return "".join(blocks)


def build_vtt(segments: list[dict[str, Any]]) -> str:
    """세그먼트 리스트로 WebVTT 자막을 만든다 (HH:MM:SS.mmm --> HH:MM:SS.mmm)."""
    blocks: list[str] = ["WEBVTT\n\n"]
    for segment in segments:
        start = _format_timestamp(segment["start"], ".")
        end = _format_timestamp(segment["end"], ".")
        text = segment["text"].strip()
        blocks.append(f"{start} --> {end}\n{text}\n\n")
    return "".join(blocks)


def _add_subtitle_outputs(
    results: list[dict[str, Any]],
    output_formats: list[str],
) -> None:
    """요청된 형식에 따라 각 결과에 srt / vtt 키를 채운다 (in-place)."""
    want_srt = "srt" in output_formats
    want_vtt = "vtt" in output_formats
    if not want_srt and not want_vtt:
        return

    for result in results:
        segments = result.get("segments")
        if segments is None:
            continue
        if want_srt:
            result["srt"] = build_srt(segments)
        if want_vtt:
            result["vtt"] = build_vtt(segments)


def _clamp_beam_size(beam_size: int, cap: int, job_id: str) -> int:
    """beam_size를 1..cap으로 자른다. 잘렸으면 경고를 남긴다."""
    clamped = max(1, min(beam_size, cap))
    if clamped != beam_size:
        logger.warning(
            "Job %s: beam_size=%d clamped to %d (allowed range 1..%d)",
            job_id, beam_size, clamped, cap,
        )
    return clamped


def _resolve_beam_size(job_input: dict[str, Any], job_id: str) -> int:
    """beam_size를 결정한다. batch_size는 deprecated 별칭으로 받는다."""
    if job_input.get("beam_size") is not None:
        return _clamp_beam_size(int(job_input["beam_size"]), MAX_BEAM_SIZE, job_id)

    legacy_batch_size = job_input.get("batch_size")
    if legacy_batch_size is not None:
        # 예전 동작을 그대로 유지한다: batch_size는 5로 캡됐다.
        # 캡을 풀면 batch_size=16을 보내던 클라이언트가 코드 변경 없이
        # 디코딩 3배 느려지고 GPU 과금이 그만큼 늘어난다.
        logger.warning(
            "Job %s: 'batch_size' is deprecated, use 'beam_size' instead", job_id
        )
        return _clamp_beam_size(int(legacy_batch_size), LEGACY_BATCH_SIZE_CAP, job_id)

    return DEFAULT_BEAM_SIZE


def _as_list(plural_value: Any, singular_value: Any, name: str) -> list[str]:
    """
    복수형(audio_urls) / 단수형(audio_url) 입력을 리스트로 정규화한다.

    복수형이 비어 있으면 단수형을 쓴다. 리스트도 문자열도 아니면 ValueError.
    """
    value = plural_value if plural_value else singular_value
    if value is None or value == []:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"'{name}' must be a string or a list of strings, got {type(value).__name__}")


def _resolve_output_formats(job_input: dict[str, Any]) -> list[str]:
    """output_formats를 정규화한다. 문자열 하나만 준 경우도 허용."""
    raw = job_input.get("output_formats", ["text"])
    if isinstance(raw, str):
        raw = [raw]
    return [str(fmt).lower() for fmt in raw]


def concurrency_modifier(current_concurrency: int) -> int:
    """
    RunPod concurrency modifier.

    RUNPOD_MAX_CONCURRENCY 환경변수로 최대 동시 Job 수 제어.
    기본값 1 = 기존 동작 유지 (한 번에 1개 Job만 처리).
    """
    max_concurrency = int(os.environ.get("RUNPOD_MAX_CONCURRENCY", "1"))
    return max_concurrency


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless handler

    Input format (audio_* 중 최소 하나 필요):
    {
        "input": {
            "audio_url": "https://...",      # 단일 파일 URL
            "audio_urls": ["https://..."],   # 여러 파일 URL batch
            "audio_base64": "...",           # 단일 파일 (하위호환)
            "audio_base64_list": ["..."],    # 여러 파일 batch
            "beam_size": 5,                  # beam search 크기 (기본 5, 상한 10)
            "batch_size": 5,                 # deprecated: beam_size 별칭
            "language": "ko",                # 언어 코드. 없으면 자동 감지
            "vad_filter": true,              # 무음 구간 제거 (기본 true)
            "word_timestamps": false,        # 단어 타임스탬프 (기본 false)
            "output_formats": ["text"]       # text / srt / vtt
        }
    }

    results 순서는 base64 입력 먼저, 그 다음 URL 입력이다.

    Output format:
    {
        "results": [
            {
                "transcription": "...",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "text": "...",
                        "avg_logprob": -0.25,
                        "no_speech_prob": 0.01,
                        "words": [           # word_timestamps=true 일 때만
                            {
                                "start": 0.0,
                                "end": 0.4,
                                "word": "안녕",
                                "probability": 0.98
                            }
                        ]
                    }
                ],
                "inference_time": 1.23,
                "detected_language": "ko",
                "language_probability": 0.99,
                "segment_count": 5,
                "srt": "...",                # output_formats에 srt 포함 시
                "vtt": "..."                 # output_formats에 vtt 포함 시
            },
            ...
        ]
    }
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    # 입력 파싱은 try 안에서 한다. 잘못된 타입(audio_urls: 42, beam_size: "fast" 등)이
    # 핸들러 밖으로 나가면 runpod SDK가 traceback과 호스트 정보를 응답에 실어버린다.
    try:
        # 단일 또는 배열 입력 지원 (base64 / URL)
        audio_b64_list = _as_list(
            job_input.get("audio_base64_list"), job_input.get("audio_base64"), "audio_base64"
        )
        audio_urls = _as_list(job_input.get("audio_urls"), job_input.get("audio_url"), "audio_url")

        file_count = len(audio_b64_list) + len(audio_urls)
        if file_count == 0:
            return {
                "error": "No audio data provided. Use 'audio_url', 'audio_urls', "
                "'audio_base64', or 'audio_base64_list'"
            }

        output_formats = _resolve_output_formats(job_input)
        unsupported = [fmt for fmt in output_formats if fmt not in SUPPORTED_OUTPUT_FORMATS]
        if unsupported:
            return {
                "error": f"Unsupported output_formats: {unsupported}. "
                f"Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"
            }

        beam_size = _resolve_beam_size(job_input, job_id)
        # language 미지정 시 None → faster-whisper가 자동 감지
        language = job_input.get("language")
        vad_filter = bool(job_input.get("vad_filter", True))
        word_timestamps = bool(job_input.get("word_timestamps", False))
    except (TypeError, ValueError) as e:
        logger.error("Job %s: invalid input parameter: %s", job_id, e)
        return {"error": f"Invalid input parameter: {e}"}

    logger.info(
        "Job %s: processing %d audio file(s), language=%s, beam_size=%d, "
        "vad_filter=%s, word_timestamps=%s, output_formats=%s",
        job_id, file_count, language or "auto", beam_size,
        vad_filter, word_timestamps, output_formats,
    )

    # base64 / URL → temp files
    temp_paths: list[str] = []

    try:
        temp_paths = await asyncio.to_thread(
            _materialize_inputs,
            audio_b64_list,
            audio_urls,
        )

        # Transcription 실행 (to_thread로 event loop 해제 → 다른 job 동시 수신 가능)
        trans = get_transcriber()
        results = await asyncio.to_thread(
            trans.transcribe_batch,
            temp_paths,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        _add_subtitle_outputs(results, output_formats)

        logger.info("Job %s: completed, %d result(s)", job_id, len(results))
        return {"results": results}

    except Exception as e:
        logger.error("Job %s: failed with error: %s", job_id, e)
        return {"error": str(e)}

    finally:
        # 임시 파일 정리
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


# RunPod serverless 시작
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier,
})
