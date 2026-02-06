FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=openai/whisper-large-v3-turbo
ENV HF_HOME=/app/models

# ffmpeg (opus 등 오디오 디코딩에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# flash-attn 빌드에 필요한 ninja
RUN pip install --no-cache-dir ninja packaging

# flash-attn은 --no-build-isolation 필요 (별도 설치)
RUN pip install --no-cache-dir --no-build-isolation flash-attn>=2.5.0

# 나머지 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# 모델을 빌드 타임에 다운로드하여 cold start 제거
COPY src/download_model.py /app/download_model.py
RUN python3 download_model.py

# 소스 코드 복사
COPY src/ /app/

CMD ["python3", "-u", "handler.py"]
