"""
RunPod Serverless Handler for insanely-faster-whisper

HuggingFace Transformers pipeline + Flash Attention 2 기반
기존 faster-whisper worker와 동일한 API 인터페이스 유지
"""

import base64
import os
import tempfile
from typing import Any

import runpod

from transcriber import InsanelyFastTranscriber

transcriber: InsanelyFastTranscriber | None = None


def get_transcriber() -> InsanelyFastTranscriber:
    global transcriber
    if transcriber is None:
        transcriber = InsanelyFastTranscriber()
    return transcriber


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless handler

    Input format:
    {
        "input": {
            "audio_base64": "...",           # 단일 파일
            "audio_base64_list": ["..."],    # 여러 파일 batch
            "batch_size": 16,                # chunk batch 크기
            "language": "ko"
        }
    }

    Output format:
    {
        "results": [
            {
                "transcription": "...",
                "inference_time": 1.23,
                "detected_language": "ko",
                "language_probability": null,
                "segment_count": 5
            },
            ...
        ]
    }
    """
    job_input = job.get("input", {})

    audio_list = job_input.get("audio_base64_list", [])
    if not audio_list and job_input.get("audio_base64"):
        audio_list = [job_input["audio_base64"]]

    if not audio_list:
        return {"error": "No audio data provided. Use 'audio_base64' or 'audio_base64_list'"}

    batch_size = job_input.get("batch_size", 16)
    language = job_input.get("language", "ko")

    temp_paths = []

    try:
        for audio_b64 in audio_list:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(base64.b64decode(audio_b64))
            temp_file.close()
            temp_paths.append(temp_file.name)

        trans = get_transcriber()
        results = trans.transcribe_batch(
            temp_paths,
            batch_size=batch_size,
            language=language,
        )

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}

    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
