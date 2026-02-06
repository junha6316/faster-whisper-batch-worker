FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=turbo
ENV CT2_NUM_WORKERS=2
ENV CT2_COMPUTE_TYPE=float16
ENV TRANSCRIBE_MAX_THREADS=2
ENV RUNPOD_MAX_CONCURRENCY=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ .

# Handler 실행
CMD ["python3", "-u", "handler.py"]
