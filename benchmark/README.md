# Benchmark harness — billed RTF, this worker vs the official worker

RunPod bills serverless workers by wall-clock second. So per-file inference time tells you almost
nothing about cost. The number that decides your bill is:

```
billed RTF       = total wall-clock seconds / total audio seconds
$ per audio hour = billed RTF x gpu_rate_per_second x 3600
```

Lower billed RTF is better. `run_benchmark.py` measures it for one endpoint, and you run it twice —
once against this worker, once against
[runpod-workers/worker-faster_whisper](https://github.com/runpod-workers/worker-faster_whisper) —
with the same audio, the same GPU type, and the same concurrency.

Why the two workers land on different numbers:

| | this worker | official worker |
|---|---|---|
| files per request | many (`audio_base64_list` / `audio_urls`) | one (`audio_base64` / `audio`) |
| handler | async | sync |
| `concurrency_modifier` | yes, one worker pulls several jobs | no, one job at a time per worker |
| jobs for 100 files | 100 / `--files-per-request` | 100 |

Each job carries fixed overhead (queue hop, input materialization, result serialization). Packing
files into fewer jobs and overlapping them on one GPU is what this harness is meant to quantify.

## Requirements

- Python 3.10+ (standard library only — nothing to install)
- `ffprobe` on PATH, for audio durations
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `apt-get install ffmpeg`
- A RunPod API key with access to the endpoints, in `RUNPOD_API_KEY` or `--api-key`

## Prepare the audio

Audio is not committed to this repo, and `benchmark/.gitignore` blocks it. Bring your own:

1. Make a directory, for example `benchmark/audio/`.
2. Put 20-100 clips in it. Formats: wav, mp3, m4a, flac, ogg, opus, webm, mp4, aac.
3. Use clips that look like your real traffic — same language, same length distribution, same codec.
   Length matters more than count: 20 files of 5 minutes tells you more than 200 files of 3 seconds.
4. Keep each request under ~8 MB of base64. The harness sends files inline as base64 and warns when
   a request body gets large. With big files, lower `--files-per-request`.

If you have no real audio yet and only want to check the harness runs, any speech recording works;
the transcript quality is irrelevant to the measurement.

## Deploy the two endpoints

Both endpoints must use the **same GPU type**, otherwise the comparison is meaningless.

1. **This worker** — build and push the image from the repo root, create a serverless endpoint from
   it, and set env vars (see the root README). For the run below, keep
   `RUNPOD_MAX_CONCURRENCY`, `TRANSCRIBE_MAX_THREADS` and `CT2_NUM_WORKERS` at the values you plan
   to use in production.
2. **Official worker** — create a second endpoint from the public image documented in
   runpod-workers/worker-faster_whisper.

Set both endpoints to the same max workers. If you want to compare a single GPU's throughput, set
max workers to 1 on both — otherwise RunPod scales the official endpoint out to many workers and you
are measuring the autoscaler, not the worker.

## Run it

```bash
cd benchmark
export RUNPOD_API_KEY=...

# This worker: 8 files per request, 4 requests in flight
python3 run_benchmark.py \
  --audio-dir ./audio \
  --endpoint-id <this-worker-endpoint-id> \
  --mode batch --files-per-request 8 --concurrency 4 \
  --gpu-rate 0.00031 --language ko \
  --out results/batch.csv

# Official worker: one file per request, same audio, same concurrency
python3 run_benchmark.py \
  --audio-dir ./audio \
  --endpoint-id <official-endpoint-id> \
  --mode single --payload-style official --concurrency 4 \
  --gpu-rate 0.00031 --language ko \
  --extra-input model=turbo \
  --out results/official.csv
```

`--gpu-rate` is dollars per second for the GPU your endpoint runs on. Take the per-hour price from
the RunPod pricing page and divide by 3600 (example: $1.10/h → `0.000306`).

A third run is worth doing: this worker in `--mode single`. Same worker, same GPU, one file per
request. The gap between that and `--mode batch` isolates the batching gain from everything else
(model, compute type, VAD settings).

`--extra-input KEY=VALUE` passes extra fields into the job input, repeatable. Use it when the two
endpoints need different knobs, for example `--extra-input model=turbo` for the official worker.

Warm the endpoint first. The first run after an endpoint is idle pays cold start (container pull,
model load into VRAM) and its billed RTF will look far worse. Run the harness twice against each
endpoint with `--limit 4` before the measured run, and report the measured run only.

## Read the results

Standard output ends with a one-row markdown table:

```
| mode | payload | files/req | concurrency | files OK | requests | audio min | wall-clock s | billed RTF | $/audio-hour |
|---|---|---|---|---|---|---|---|---|---|
| batch | batch-worker | 8 | 4 | 64 | 8 | 182.4 | 91.3 | 0.0083 | $0.0093 |
```

- **billed RTF** — the headline. 0.0083 means one wall-clock second transcribes about 120 seconds of
  audio. Compare this figure between the two runs; the ratio is your cost ratio.
- **$/audio-hour** — the same figure in money. Multiply by your monthly audio hours to project cost.
- **per-request wall-clock** — min / median / max. A wide spread with high concurrency means jobs are
  queueing behind each other rather than running in parallel.
- **worker-reported inference_time** — only this worker returns it. Its sum divided by total audio is
  the inference-only RTF. It is always better than billed RTF; the difference is job overhead and
  idle worker time, which you still pay for.
- **RunPod executionTime** — RunPod's own per-job billed milliseconds, summed over requests. Between
  billed RTF and inference-only RTF. It ignores gaps between jobs, so it is optimistic as a cost
  number but useful for spotting per-job overhead.

The CSV has one row per audio file with `duration_seconds`, the wall-clock of the request it rode in,
`delay_time_ms` (queue wait), `execution_time_ms`, `inference_time`, and any error text. Use it to
find slow outliers or to recompute metrics over a subset.

Failures are counted separately and excluded from the audio total. If the summary prints a failure
warning, fix the cause and rerun — a run with failures is not comparable to a clean one.

## Caveats

- Wall-clock is measured client side, so it includes base64 upload and queue wait. That is
  deliberate: those seconds are part of the end-to-end latency you observe, and queue wait exists
  because a worker is busy. It does mean a slow uplink inflates both runs, so run both from the same
  machine and network.
- RunPod autoscaling, GPU availability, and neighbouring load vary between runs. Repeat each run
  three times and compare medians before drawing a conclusion.
- `--concurrency` is client-side requests in flight. It interacts with the endpoint's max workers and
  with `RUNPOD_MAX_CONCURRENCY` on this worker. Note all three values with every result.
