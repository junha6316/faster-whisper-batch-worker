FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=turbo
ENV CT2_NUM_WORKERS=4
ENV CT2_COMPUTE_TYPE=float16
ENV TRANSCRIBE_MAX_THREADS=4
ENV RUNPOD_MAX_CONCURRENCY=8
# 모델 캐시를 이미지 안에 고정한다 (빌드 시 프리페치 → cold start에서 다운로드 없음)
ENV HF_HOME=/models

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

# 모델 가중치 프리페치 (WHISPER_MODEL 변경 시에만 이 레이어가 다시 빌드된다)
RUN python3 -c "import os; from faster_whisper import download_model; download_model(os.environ['WHISPER_MODEL'])"

# 소스 코드 복사
COPY handler.py batch_transcriber.py ./

# Handler 실행
CMD ["python3", "-u", "handler.py"]
