"""빌드 타임에 모델을 다운로드하여 cold start 제거"""

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")

print(f"Downloading model: {model_id}")

AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
AutoProcessor.from_pretrained(model_id)

print(f"Model {model_id} downloaded successfully")
