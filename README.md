# faster-whisper Batch Worker for RunPod Serverless

A RunPod Serverless worker that transcribes **many audio files per job** on a single GPU, using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).

RunPod bills per worker-second, not per audio-second. So the thing that decides your cost is how
much audio one billed GPU-second can chew through. This worker stacks three layers of parallelism
to push that number up.

## Features

- Multiple audio files in one request (`audio_urls` / `audio_base64_list`)
- Three-layer concurrency on a single GPU (see below) — measured ~133× realtime on one RTX 4090
- Audio input by URL or base64
- Segment-level output with timestamps, `avg_logprob`, `no_speech_prob`
- Model weights baked into the image — no download on cold start, no network volume needed
- Silero VAD filter enabled, `temperature=0` (anti-hallucination)

## Layout

```
Dockerfile              # CUDA 12.3 + cuDNN 9 base, model prefetched at build time
requirements.txt        # runpod, faster-whisper
src/handler.py          # RunPod handler: input parsing, concurrency_modifier
src/batch_transcriber.py # WhisperModel + ThreadPoolExecutor
```

## How the concurrency works

Three independent knobs, each at a different layer. All three matter; setting only one leaves
throughput on the table.

| Layer | What it does | Env var |
|---|---|---|
| RunPod concurrency | How many jobs one worker pulls at the same time | `RUNPOD_MAX_CONCURRENCY` |
| ThreadPoolExecutor | How many files inside one job run at the same time | `TRANSCRIBE_MAX_THREADS` |
| CTranslate2 workers | How many inferences run on the GPU at the same time | `CT2_NUM_WORKERS` |

The bottom layer is what makes this work: **CTranslate2 releases the Python GIL during
computation**, so plain Python threads give real GPU parallelism. Model weights are shared across
CT2 workers, so extra workers cost activation memory (~0.5–1.5 GB each) but not another copy of the
model. The handler is `async` and hands transcription to `asyncio.to_thread()`, which frees the
event loop so RunPod can deliver the next job while the current one is still running.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `turbo` | Any faster-whisper model name (`turbo`, `large-v3`, `distil-large-v3`, …). Prefetched at build time. |
| `CT2_NUM_WORKERS` | `4` | CTranslate2 parallel workers. Raise for more throughput, watch VRAM. |
| `CT2_COMPUTE_TYPE` | `float16` | `int8_float16` cuts VRAM at some quality cost. |
| `TRANSCRIBE_MAX_THREADS` | `4` | Files transcribed in parallel within one job. `1` forces sequential. |
| `RUNPOD_MAX_CONCURRENCY` | `8` | Concurrent jobs pulled per worker. |
| `AUDIO_DOWNLOAD_TIMEOUT` | `60` | Per-URL download timeout in seconds. |

Defaults are tuned for a 24 GB GPU (RTX 4090). On a 16 GB card start with `CT2_NUM_WORKERS=2`,
`TRANSCRIBE_MAX_THREADS=2` and raise while watching peak VRAM.

## Deployment

### 1. Build and push

```bash
docker build -t your-dockerhub-username/faster-whisper-batch-worker:latest .
docker push your-dockerhub-username/faster-whisper-batch-worker:latest
```

The build prefetches the model, so the image is ~3–4 GB and needs no network volume.

### 2. Create the endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) → **New Endpoint**
2. Container image: the tag you just pushed
3. GPU: RTX 4090 or better (24 GB recommended for the default settings)
4. Environment variables: override the table above as needed
5. Deploy

## API

### Request

```json
{
  "input": {
    "audio_urls": ["https://example.com/call-1.opus", "https://example.com/call-2.opus"],
    "language": "ko",
    "batch_size": 5
  }
}
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `audio_url` | one of the four | — | Single file by URL |
| `audio_urls` | one of the four | — | Multiple files by URL |
| `audio_base64` | one of the four | — | Single file, base64 |
| `audio_base64_list` | one of the four | — | Multiple files, base64 |
| `language` | no | `ko` | Whisper language code |
| `batch_size` | no | `16` | **Used as `beam_size`**, capped at 5. See caveats. |

Results come back in input order: base64 inputs first, then URL inputs.

Prefer URLs when sending several files — RunPod caps request payloads at 20 MB, and base64 inflates
audio by about a third.

### Response

```json
{
  "results": [
    {
      "transcription": "전체 텍스트",
      "segments": [
        {
          "start": 0.0,
          "end": 2.5,
          "text": "세그먼트 텍스트",
          "avg_logprob": -0.25,
          "no_speech_prob": 0.01
        }
      ],
      "inference_time": 4.41,
      "detected_language": "ko",
      "language_probability": 0.99,
      "segment_count": 5
    }
  ]
}
```

A file that fails gets `{"error": "...", "inference_time": ...}` in its slot; the rest of the batch
still returns. A job-level failure returns `{"error": "..."}`.

### Example

```python
import requests

ENDPOINT = "https://api.runpod.ai/v2/{your-endpoint-id}/runsync"
API_KEY = "your-api-key"

response = requests.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "audio_urls": [
                "https://example.com/call-1.opus",
                "https://example.com/call-2.opus",
            ],
            "language": "ko",
        }
    },
    timeout=600,
)

for result in response.json()["output"]["results"]:
    print(result["detected_language"], result["inference_time"])
    print(result["transcription"])
```

### Local test

```bash
docker build -t faster-whisper-batch-worker .
docker run --rm --gpus all \
  -e RUNPOD_MAX_CONCURRENCY=4 \
  faster-whisper-batch-worker \
  python3 -c "
import asyncio, handler
print(asyncio.run(handler.handler({'input': {'audio_url': 'https://example.com/sample.wav'}})))
"
```

## Measured throughput

One RTX 4090, warm worker, `CT2_NUM_WORKERS=4` / `TRANSCRIBE_MAX_THREADS=4` /
`RUNPOD_MAX_CONCURRENCY=8`, 20 Korean speech files of 2–22 minutes (174.2 minutes total):

| Metric | Value |
|---|---|
| Sum of per-file inference time | 474.5 s (mean per-file RTF 0.052) |
| Wall-clock for all 20 files | **78.6 s** |
| Effective throughput | **~133× realtime** (billed RTF 0.0075) |

The gap between 474.5 s of inference and 78.6 s of wall-clock is the whole point: on RunPod you pay
for the wall-clock, and the three concurrency layers overlap the inference. For comparison, on the
same files and GPU, one-file-per-job sequential processing measured a billed RTF around 0.031 —
roughly 4× more expensive per audio hour.

Cold start is excluded here. Real-world cost depends on how often workers cold-start and how full
your batches are.

## Caveats

- **`batch_size` is not a batch size.** It is passed through as `beam_size` (capped at 5), so the
  default of 16 means `beam_size=5` — the slowest setting. Pass `batch_size: 1` for greedy decoding
  if you want speed and your quality bar allows it. The name is kept for backward compatibility.
- **`BatchedInferencePipeline` is not used.** Parallelism comes from threads plus CT2 workers, not
  from faster-whisper's chunk batching. Combining the two may add throughput; untested here.
- Word-level timestamps are not exposed.
- `vad_filter=True` is always on. If you already stripped silence upstream, this is wasted work.

## License

MIT — see [LICENSE](LICENSE).

Dependencies keep their own licenses: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(MIT), [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT), and the model weights
([mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)
for `turbo`).
