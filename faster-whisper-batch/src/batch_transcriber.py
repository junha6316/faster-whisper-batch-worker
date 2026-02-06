"""
WhisperModel + ThreadPoolExecutor를 사용한 프로세스 레벨 동시 처리 transcriber

CTranslate2는 GPU 연산 중 Python GIL을 해제하므로,
num_workers + ThreadPoolExecutor 조합으로 단일 프로세스 내 진정한 GPU 병렬 처리 가능.
- 모델 가중치는 GPU에서 공유 (VRAM 중복 없음)
- 활성화 버퍼만 worker 수에 비례하여 증가 (~0.5-1.5GB/worker)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class BatchTranscriber:
    def __init__(self, model_name: str = "turbo"):
        """
        WhisperModel 기반 transcriber 초기화

        환경변수:
            CT2_NUM_WORKERS: CTranslate2 worker 수 (기본 2)
            CT2_COMPUTE_TYPE: 연산 타입 (기본 float16)
            TRANSCRIBE_MAX_THREADS: ThreadPoolExecutor 최대 스레드 수 (기본 2)
        """
        num_workers = int(os.environ.get("CT2_NUM_WORKERS", "2"))
        compute_type = os.environ.get("CT2_COMPUTE_TYPE", "float16")
        self.max_threads = int(os.environ.get("TRANSCRIBE_MAX_THREADS", "2"))

        logger.info(
            "Initializing WhisperModel: model=%s, num_workers=%d, compute_type=%s, max_threads=%d",
            model_name, num_workers, compute_type, self.max_threads,
        )

        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type=compute_type,
            num_workers=num_workers,
        )
        self.model_name = model_name

    def _transcribe_one(
        self,
        audio_path: str,
        language: str,
        beam_size: int,
    ) -> dict:
        """단일 파일 transcription (스레드에서 실행)"""
        start_time = time.perf_counter()

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=False,
            )

            segment_list = list(segments)
            text = " ".join([s.text.strip() for s in segment_list])
            inference_time = time.perf_counter() - start_time

            return {
                "transcription": text,
                "inference_time": inference_time,
                "detected_language": info.language,
                "language_probability": info.language_probability,
                "segment_count": len(segment_list),
            }
        except Exception as e:
            inference_time = time.perf_counter() - start_time
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return {
                "error": str(e),
                "inference_time": inference_time,
            }

    def transcribe_batch(
        self,
        audio_paths: list[str],
        batch_size: int = 16,
        language: str = "ko",
    ) -> list[dict]:
        """
        여러 오디오 파일을 병렬 처리

        파일이 1개이거나 max_threads <= 1이면 순차 처리.
        그 외에는 ThreadPoolExecutor로 병렬 submit.

        Args:
            audio_paths: 오디오 파일 경로 리스트
            batch_size: beam_size로 사용 (기본 16 → 5로 권장)
            language: 언어 코드 (기본 ko)

        Returns:
            각 파일의 transcription 결과 리스트 (입력 순서 보장)
        """
        beam_size = min(batch_size, 5)

        # 순차 처리 폴백: 파일 1개이거나 스레드 비활성
        if len(audio_paths) <= 1 or self.max_threads <= 1:
            return [
                self._transcribe_one(path, language, beam_size)
                for path in audio_paths
            ]

        # ThreadPoolExecutor 병렬 처리
        num_threads = min(self.max_threads, len(audio_paths))
        results: list[dict | None] = [None] * len(audio_paths)

        logger.info(
            "Starting parallel transcription: %d files, %d threads",
            len(audio_paths), num_threads,
        )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_idx = {
                executor.submit(self._transcribe_one, path, language, beam_size): idx
                for idx, path in enumerate(audio_paths)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Thread exception for file index %d: %s", idx, e)
                    results[idx] = {"error": str(e), "inference_time": 0.0}

        return results

    def transcribe_single(
        self,
        audio_path: str,
        batch_size: int = 16,
        language: str = "ko",
    ) -> dict:
        """단일 파일 transcription (transcribe_batch의 편의 메서드)"""
        results = self.transcribe_batch([audio_path], batch_size, language)
        return results[0] if results else {}
