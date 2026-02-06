"""
RunPod Serverless Handler with process-level concurrency support

WhisperModel + ThreadPoolExecutor로 GPU 병렬 처리 최적화.
concurrency_modifier로 RunPod이 동시에 여러 Job을 Pull 가능.
"""

import base64
import logging
import os
import tempfile
from typing import Any

import runpod

from batch_transcriber import BatchTranscriber

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


def concurrency_modifier(current_concurrency: int) -> int:
    """
    RunPod concurrency modifier.

    RUNPOD_MAX_CONCURRENCY 환경변수로 최대 동시 Job 수 제어.
    기본값 1 = 기존 동작 유지 (한 번에 1개 Job만 처리).
    """
    max_concurrency = int(os.environ.get("RUNPOD_MAX_CONCURRENCY", "1"))
    return max_concurrency


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless handler

    Input format:
    {
        "input": {
            "audio_base64": "...",           # 단일 파일 (하위호환)
            "audio_base64_list": ["..."],    # 여러 파일 batch
            "batch_size": 16,                # beam_size (최대 5로 cap)
            "language": "ko"                 # 언어 코드
        }
    }

    Output format:
    {
        "results": [
            {
                "transcription": "...",
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

    # 단일 또는 배열 입력 지원
    audio_list = job_input.get("audio_base64_list", [])
    if not audio_list and job_input.get("audio_base64"):
        audio_list = [job_input["audio_base64"]]

    if not audio_list:
        return {"error": "No audio data provided. Use 'audio_base64' or 'audio_base64_list'"}

    batch_size = job_input.get("batch_size", 16)
    language = job_input.get("language", "ko")

    logger.info("Job %s: processing %d audio file(s), language=%s", job_id, len(audio_list), language)

    # base64 → temp files
    temp_paths = []

    try:
        for audio_b64 in audio_list:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(base64.b64decode(audio_b64))
            temp_file.close()
            temp_paths.append(temp_file.name)

        # Transcription 실행
        trans = get_transcriber()
        results = trans.transcribe_batch(
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
