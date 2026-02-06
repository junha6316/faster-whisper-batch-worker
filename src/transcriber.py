"""
insanely-faster-whisper 스타일 transcription

HuggingFace Transformers pipeline + Flash Attention 2 사용
"""

import os
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class InsanelyFastTranscriber:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get(
            "WHISPER_MODEL", "openai/whisper-large-v3-turbo"
        )
        self.device = "cuda:0"
        self.torch_dtype = torch.float16

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe_batch(
        self,
        audio_paths: list[str],
        batch_size: int = 24,
        language: str = "ko",
    ) -> list[dict]:
        results = []

        for audio_path in audio_paths:
            start_time = time.perf_counter()

            output = self.pipe(
                audio_path,
                chunk_length_s=30,
                batch_size=batch_size,
                generate_kwargs={"language": language},
                return_timestamps=True,
            )

            inference_time = time.perf_counter() - start_time
            text = output.get("text", "").strip()
            chunks = output.get("chunks", [])

            results.append({
                "transcription": text,
                "inference_time": inference_time,
                "detected_language": language,
                "segment_count": len(chunks),
            })

        return results

    def transcribe_single(
        self,
        audio_path: str,
        batch_size: int = 24,
        language: str = "ko",
    ) -> dict:
        results = self.transcribe_batch([audio_path], batch_size, language)
        return results[0] if results else {}
