# faster-whisper Batch Worker for RunPod Serverless

A RunPod Serverless worker that transcribes **many audio files per job** on a single GPU, using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).

RunPod bills per worker-second, not per audio-second. So the thing that decides your cost is how
much audio one billed GPU-second can chew through. This worker stacks three layers of parallelism
to push that number up — measured ~133× realtime on one RTX 4090 (see
[Measured throughput](#measured-throughput)).

Prebuilt image: `ghcr.io/junha6316/faster-whisper-batch-worker`

## Features

- Multiple audio files in one request (`audio_urls` / `audio_base64_list`)
- Three-layer concurrency on a single GPU (see [below](#how-the-concurrency-works))
- Audio input by URL or base64
- Segment output with timestamps, `avg_logprob`, `no_speech_prob`; optional word-level timestamps
- Optional `srt` / `vtt` subtitle output, built in the handler with no extra dependency
- Model weights baked into the image — no download on cold start, no network volume needed
- Silero VAD filter on by default (switchable per request), `temperature=0` (anti-hallucination)

## Quick start — deploy the published image

The image is built and pushed by GitHub Actions on every push to `main` and every `v*` tag, so you
do not have to build anything.

```bash
docker pull ghcr.io/junha6316/faster-whisper-batch-worker:latest
```

> The published image appears after the first CI run on `main`, and the GHCR package has to be
> flipped to public once by hand before anonymous `pull` works. Until then, use
> [Build it yourself](#build-it-yourself).

### Create a RunPod endpoint from it

1. [RunPod Serverless](https://www.runpod.io/console/serverless) → **New Endpoint** → **Import from
   Docker Registry**.
2. **Container Image**: `ghcr.io/junha6316/faster-whisper-batch-worker:latest`.
   For a reproducible deployment, pin a specific tag instead of `latest` — either a release tag
   (`:0.1.0`) or an exact commit (`:sha-<40-char-commit-sha>`). `latest` moves on every push to
   `main`.
3. Registry credentials: none needed once the GHCR package is public.
4. GPU: 24 GB or more (RTX 4090 / L4 / A5000 / L40S / A6000). Container disk: 20 GB — the image is
   ~3–4 GB with the weights baked in, and the rest is headroom for temp audio files.
5. Environment variables: override the [table below](#environment-variables) as needed. The image
   already ships production defaults, so you can leave them alone.
6. Deploy, then send a request — see [API](#api).

### RunPod Hub (one-click deploy)

The repo carries `.runpod/hub.json` and `.runpod/tests.json`, which is everything the
[RunPod Hub](https://docs.runpod.io/hub/publishing-guide) needs to render a one-click deploy page
with the six environment variables as form fields.

Listing is **not** automatic and cannot be done from code — a human has to:

1. Add the repository once in the RunPod console (Hub → *Add your repo*).
2. Create a GitHub release. The Hub indexes releases, not commits.
3. Wait for RunPod to build, run `.runpod/tests.json`, and manually review the listing.

Until that is done, use the GHCR path above; it is the same image.

## API

### Request

```json
{
  "input": {
    "audio_urls": ["https://example.com/call-1.opus", "https://example.com/call-2.opus"],
    "language": "ko",
    "beam_size": 5,
    "output_formats": ["text", "srt"]
  }
}
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `audio_url` | one of the four | — | Single file by URL |
| `audio_urls` | one of the four | — | Multiple files by URL |
| `audio_base64` | one of the four | — | Single file, base64 |
| `audio_base64_list` | one of the four | — | Multiple files, base64 |
| `language` | no | auto-detect | Whisper language code (`ko`, `en`, …). Omit it and faster-whisper detects the language per file. |
| `beam_size` | no | `5` | Beam search width, used as given. `1` is greedy decoding — fastest, slightly worse. |
| `batch_size` | no | — | **Deprecated** alias for `beam_size`; logs a warning. `beam_size` wins if both are sent. |
| `vad_filter` | no | `true` | Silero VAD silence removal. Set `false` if you already stripped silence upstream. |
| `word_timestamps` | no | `false` | `true` adds a `words` array to every segment. |
| `output_formats` | no | `["text"]` | Subset of `text`, `srt`, `vtt`. A bare string (`"srt"`) is accepted. Anything else returns an error. |

Results come back in input order: base64 inputs first, then URL inputs.

Prefer URLs when sending several files. RunPod caps request payloads at
[10 MB for `/run` and 20 MB for `/runsync`](https://docs.runpod.io/serverless/workers/handlers/overview),
and base64 inflates audio by about a third.

One change to watch if you used an earlier version of this worker: `language` used to default to
`ko`, and now defaults to auto-detect. Pass `"language": "ko"` explicitly to keep the old behaviour.
Everything else is backward compatible — `batch_size` is still accepted and still clamped at 5, so
existing callers get the same decoding and the same cost as before.

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

Those keys are always present on a successful file. Two things get added on request:

- `word_timestamps: true` → each segment also carries
  `"words": [{"start": 0.0, "end": 0.4, "word": "안녕", "probability": 0.98}, …]`.
- `output_formats` containing `srt` and/or `vtt` → each result also carries an `srt` and/or `vtt`
  string, built from the same segments (`HH:MM:SS,mmm` for SRT, `WEBVTT` header plus
  `HH:MM:SS.mmm` for VTT). `text` needs no flag; it is always in `transcription`.

A file that fails transcription gets `{"error": "...", "inference_time": ...}` in its slot and no
subtitle keys; the rest of the batch still returns. A job-level failure — bad input, a URL that will
not download — returns `{"error": "..."}` with no `results`.

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
            "output_formats": ["text", "srt"],
        }
    },
    timeout=600,
)

for result in response.json()["output"]["results"]:
    print(result["detected_language"], result["inference_time"])
    print(result["transcription"])
    print(result["srt"])
```

### Local test

Needs an NVIDIA GPU and the container toolkit.

```bash
docker run --rm --gpus all \
  -e RUNPOD_MAX_CONCURRENCY=4 \
  ghcr.io/junha6316/faster-whisper-batch-worker:latest \
  python3 handler.py --test_input '{"input": {"audio_url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"}}'
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `turbo` | Any faster-whisper model name (`turbo`, `large-v3`, `distil-large-v3`, …). Only `turbo` is prefetched into the image; any other value is downloaded on the first cold start. |
| `CT2_NUM_WORKERS` | `4` | CTranslate2 parallel workers. Raise for more throughput, watch VRAM. |
| `CT2_COMPUTE_TYPE` | `float16` | Also `int8_float16`, `int8`, `float32`. `int8_float16` cuts VRAM at some quality cost. |
| `TRANSCRIBE_MAX_THREADS` | `4` | Files transcribed in parallel within one job. `1` forces sequential. |
| `RUNPOD_MAX_CONCURRENCY` | `8` | Concurrent jobs pulled per worker. |
| `AUDIO_DOWNLOAD_TIMEOUT` | `60` | Per-URL download timeout in seconds. |

The first five defaults are baked into the image as Dockerfile `ENV`. If you run
`handler.py` outside the image without setting them, the in-code fallbacks are lower:
`CT2_NUM_WORKERS=2`, `TRANSCRIBE_MAX_THREADS=2`, `RUNPOD_MAX_CONCURRENCY=1`.

Defaults are tuned for a 24 GB GPU (RTX 4090). On a 16 GB card start with `CT2_NUM_WORKERS=2`,
`TRANSCRIBE_MAX_THREADS=2` and raise while watching peak VRAM.

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

## Reproduce the benchmark

[`benchmark/`](benchmark/) has a standard-library harness that measures the number above on your own
audio, and compares this worker against
[runpod-workers/worker-faster_whisper](https://github.com/runpod-workers/worker-faster_whisper) under
identical conditions:

```
billed RTF       = total wall-clock seconds / total audio seconds
$ per audio hour = billed RTF x gpu_rate_per_second x 3600
```

```bash
cd benchmark
export RUNPOD_API_KEY=...

# this worker: 8 files per request, 4 requests in flight
python3 run_benchmark.py --audio-dir ./audio --endpoint-id <this-worker-endpoint> \
    --mode batch --files-per-request 8 --concurrency 4 --gpu-rate 0.00031 --out results/batch.csv

# official worker: one file per request, same audio, same GPU, same concurrency
python3 run_benchmark.py --audio-dir ./audio --endpoint-id <official-endpoint> \
    --mode single --payload-style official --concurrency 4 --gpu-rate 0.00031 \
    --extra-input model=turbo --out results/official.csv
```

`--extra-input model=turbo` matters: the official worker defaults to `base`, so without it you
would be comparing turbo against a smaller model.

It needs `ffprobe` (from ffmpeg) for durations, and your own audio — no audio is committed to this
repo. Files are sent inline as base64, so lower `--files-per-request` if the harness warns about
request size. Full flag list, endpoint setup and how to read the output:
[benchmark/README.md](benchmark/README.md).

## Build it yourself

### CI (what publishes the image)

[`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml) builds and pushes to
GHCR on push to `main`, on `v*` tags, and on manual `workflow_dispatch`. It needs **no secrets** —
the built-in `GITHUB_TOKEN` with `packages: write` is enough.

| Trigger | Tags pushed |
|---|---|
| push to `main` (or manual run on `main`) | `latest`, `main`, `sha-<40-char-commit-sha>` |
| push of tag `v1.2.3` | `1.2.3`, `1.2`, `sha-<40-char-commit-sha>` |

Cutting a release is therefore:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

One manual step after the very first successful run: a newly published GHCR package is
[private by default](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).
Open the package page → *Package settings* → *Change visibility* → **Public**, otherwise neither
`docker pull` nor RunPod can fetch it without registry credentials.

### Local build and push

```bash
docker build -t ghcr.io/<your-user>/faster-whisper-batch-worker:latest .
docker push ghcr.io/<your-user>/faster-whisper-batch-worker:latest
```

The build prefetches the model weights, so the image is ~3–4 GB and needs no network volume. Change
`WHISPER_MODEL` in the Dockerfile to bake a different model in.

## Layout

```
Dockerfile                      # CUDA 12.3 + cuDNN 9 base, model prefetched at build time
requirements.txt                # runpod, faster-whisper
handler.py                      # RunPod handler: input parsing, srt/vtt output, concurrency_modifier
batch_transcriber.py            # WhisperModel + ThreadPoolExecutor
.github/workflows/docker-publish.yml  # build and push to GHCR
.runpod/hub.json                # RunPod Hub listing + deploy-time env fields
.runpod/tests.json              # test the Hub runs on each release
benchmark/                      # billed-RTF harness, this worker vs the official worker
```

## Caveats

- **`batch_size` is deprecated.** It is still accepted as an alias for `beam_size` and still
  clamped at 5, exactly as before, so existing callers keep their old behaviour and cost. Only the
  new `beam_size` parameter can go above 5. Move to `beam_size`.
- **`BatchedInferencePipeline` is not used.** Parallelism comes from threads plus CT2 workers, not
  from faster-whisper's chunk batching. Combining the two may add throughput; untested here.
- **A URL that will not download fails the whole job**, not just that file. Per-file isolation only
  covers transcription failures.
- Only `turbo` is prefetched into the image. Setting `WHISPER_MODEL` to anything else means a model
  download on the first cold start of every new worker.
- The image is built for `linux/amd64` only — fine for RunPod GPUs, not runnable on arm64 hosts.
- Base64 input is convenient but bounded by RunPod's request payload limits (10 MB `/run`,
  20 MB `/runsync`). URLs scale further.

## License

MIT — see [LICENSE](LICENSE).

Dependencies keep their own licenses: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(MIT), [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT), and the model weights
([mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)
for `turbo`).
