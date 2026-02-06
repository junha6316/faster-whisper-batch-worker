FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=openai/whisper-large-v3-turbo
ENV HF_HOME=/app/models

# flash-attn 빌드에 필요한 ninja
RUN pip install --no-cache-dir ninja packaging

# PyTorch 는 베이스 이미지에 포함되어 있으므로 나머지 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# 모델을 빌드 타임에 다운로드하여 cold start 제거
COPY src/download_model.py /app/download_model.py
RUN python3 download_model.py

# 소스 코드 복사
COPY src/ /app/

CMD ["python3", "-u", "handler.py"]
