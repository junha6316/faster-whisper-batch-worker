"""빌드 타임에 모델을 다운로드하여 cold start 제거"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from transformers import pipeline

model_id = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")

print(f"Downloading model: {model_id}")

pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device="cpu",
    model_kwargs={"attn_implementation": "sdpa"},
)

print(f"Model {model_id} downloaded successfully")
