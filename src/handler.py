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
            "batch_size": 16,                # beam_size로 사용됨 (최대 5로 cap)
            "language": "ko"                 # 언어 코드
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
                        "no_speech_prob": 0.01
                    }
                ],
                "inference_time": 1.23,
                "detected_language": "ko",
                "language_probability": 0.99,
                "segment_count": 5
            },
            ...
        ]
    }
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    # 단일 또는 배열 입력 지원 (base64 / URL)
    audio_b64_list = job_input.get("audio_base64_list", [])
    if not audio_b64_list and job_input.get("audio_base64"):
        audio_b64_list = [job_input["audio_base64"]]

    audio_urls = job_input.get("audio_urls", [])
    if not audio_urls and job_input.get("audio_url"):
        audio_urls = [job_input["audio_url"]]

    file_count = len(audio_b64_list) + len(audio_urls)
    if file_count == 0:
        return {
            "error": "No audio data provided. Use 'audio_url', 'audio_urls', "
            "'audio_base64', or 'audio_base64_list'"
        }

    batch_size = job_input.get("batch_size", 16)
    language = job_input.get("language", "ko")

    logger.info("Job %s: processing %d audio file(s), language=%s", job_id, file_count, language)

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
            batch_size=batch_size,
            language=language,
        )

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
