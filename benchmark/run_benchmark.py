#!/usr/bin/env python3
"""
Billed-RTF benchmark harness for RunPod Whisper serverless endpoints.

RunPod bills serverless workers by wall-clock seconds, so the number that
matters is not per-file inference time but:

    billed RTF        = total wall-clock seconds / total audio seconds
    $ per audio hour  = billed RTF * gpu_rate_per_second * 3600

The harness can drive two shapes of endpoint with the same audio set:

    --mode batch   many files in one request  (this worker)
    --mode single  one file per request       (official worker's only shape)

and two request payload shapes:

    --payload-style batch-worker   audio_base64_list / audio_base64 + language
    --payload-style official       audio_base64 + language (runpod-workers/worker-faster_whisper)

Only runpod's public HTTP API and the standard library are used, so this
script runs anywhere python3 and ffprobe exist.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_BASE = "https://api.runpod.ai/v2"

AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".mp4", ".aac", ".wma",
}

# RunPod rejects oversized request bodies; warn well before the documented limit.
PAYLOAD_WARN_BYTES = 8 * 1024 * 1024

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


# --------------------------------------------------------------------------
# pure calculation (unit-testable, no I/O)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunMetrics:
    total_audio_seconds: float
    wall_clock_seconds: float
    gpu_rate_per_second: float
    billed_rtf: float
    cost_per_audio_hour: float
    cost_of_run: float


def compute_run_metrics(
    total_audio_seconds: float,
    wall_clock_seconds: float,
    gpu_rate_per_second: float,
) -> RunMetrics:
    """
    Billed RTF and the cost it implies.

    total_audio_seconds: sum of durations of the audio that was transcribed OK
    wall_clock_seconds:  wall-clock of the whole run (first request sent →
                         last response received). This is the quantity RunPod
                         charges for when the endpoint is saturated.
    gpu_rate_per_second: endpoint GPU price per second, e.g. 0.00031

    >>> m = compute_run_metrics(600.0, 60.0, 0.00031)
    >>> round(m.billed_rtf, 6)
    0.1
    >>> round(m.cost_per_audio_hour, 6)
    0.1116
    """
    if total_audio_seconds <= 0:
        raise ValueError("total_audio_seconds must be > 0")
    if wall_clock_seconds <= 0:
        raise ValueError("wall_clock_seconds must be > 0")
    if gpu_rate_per_second < 0:
        raise ValueError("gpu_rate_per_second must be >= 0")

    billed_rtf = wall_clock_seconds / total_audio_seconds
    return RunMetrics(
        total_audio_seconds=total_audio_seconds,
        wall_clock_seconds=wall_clock_seconds,
        gpu_rate_per_second=gpu_rate_per_second,
        billed_rtf=billed_rtf,
        cost_per_audio_hour=billed_rtf * gpu_rate_per_second * 3600.0,
        cost_of_run=wall_clock_seconds * gpu_rate_per_second,
    )


def chunk(items: list, size: int) -> list[list]:
    """
    Split a list into consecutive groups of at most `size`.

    >>> chunk([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


# --------------------------------------------------------------------------
# audio discovery / duration
# --------------------------------------------------------------------------

def find_audio_files(audio_dir: Path, limit: int | None) -> list[Path]:
    if not audio_dir.is_dir():
        raise SystemExit(f"error: --audio-dir is not a directory: {audio_dir}")
    files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise SystemExit(
            f"error: no audio files in {audio_dir} "
            f"(looked for {', '.join(sorted(AUDIO_SUFFIXES))})"
        )
    return files[:limit] if limit else files


def probe_duration(path: Path) -> float:
    """Audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit(
            "error: ffprobe not found. Install ffmpeg first "
            "(macOS: brew install ffmpeg, Debian/Ubuntu: apt-get install ffmpeg)."
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"error: ffprobe failed on {path}: {exc.stderr.strip()}")

    text = out.strip()
    try:
        seconds = float(text)
    except ValueError:
        raise SystemExit(f"error: ffprobe returned no duration for {path}: {text!r}")
    if seconds <= 0:
        raise SystemExit(f"error: ffprobe returned non-positive duration for {path}: {seconds}")
    return seconds


# --------------------------------------------------------------------------
# request payloads
# --------------------------------------------------------------------------

def encode_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_extra_input(pairs: list[str]) -> dict:
    """`--extra-input model=turbo --extra-input beam_size=5` → {"model": "turbo", "beam_size": 5}"""
    extra: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"error: --extra-input needs KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            extra[key] = json.loads(raw)
        except json.JSONDecodeError:
            extra[key] = raw
    return extra


def build_payload(
    paths: list[Path],
    payload_style: str,
    language: str,
    extra_input: dict,
) -> dict:
    if payload_style == "official":
        if len(paths) != 1:
            raise SystemExit(
                "error: --payload-style official takes one file per request; "
                "use --mode single"
            )
        job_input = {"audio_base64": encode_base64(paths[0]), "language": language}
    else:
        job_input = {
            "audio_base64_list": [encode_base64(p) for p in paths],
            "language": language,
        }
    job_input.update(extra_input)
    return {"input": job_input}


# --------------------------------------------------------------------------
# RunPod HTTP
# --------------------------------------------------------------------------

def _post_json(url: str, api_key: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_job(
    api_base: str,
    endpoint_id: str,
    api_key: str,
    payload: dict,
    timeout: float,
    poll_interval: float,
) -> dict:
    """
    Submit one job with /runsync and, if it comes back still queued
    (runsync gives up streaming after ~90s), poll /status until terminal.
    """
    deadline = time.monotonic() + timeout
    result = _post_json(
        f"{api_base}/{endpoint_id}/runsync", api_key, payload, timeout=timeout
    )
    job_id = result.get("id")

    while result.get("status") not in TERMINAL_STATUSES:
        if not job_id:
            # No status and no id: treat the body as the final answer.
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} still {result.get('status')} after {timeout}s")
        time.sleep(poll_interval)
        result = _get_json(
            f"{api_base}/{endpoint_id}/status/{job_id}", api_key,
            timeout=max(10.0, poll_interval * 5),
        )

    return result


# --------------------------------------------------------------------------
# per-request execution
# --------------------------------------------------------------------------

@dataclass
class FileRow:
    request_index: int
    file: str
    duration_seconds: float
    files_in_request: int
    request_wall_clock_seconds: float
    status: str
    inference_time: float | None
    delay_time_ms: float | None
    execution_time_ms: float | None
    error: str


def _extract_per_file(output, file_count: int) -> list[dict]:
    """
    Normalize both response shapes into one dict per input file.

    this worker:     {"results": [{...}, {...}]}
    official worker: {"segments": [...], "transcription": "..."}   (single file)
    """
    if isinstance(output, dict) and isinstance(output.get("results"), list):
        per_file = output["results"]
    elif isinstance(output, dict):
        per_file = [output]
    elif isinstance(output, list):
        per_file = output
    else:
        per_file = []

    per_file = [r if isinstance(r, dict) else {} for r in per_file]
    if len(per_file) < file_count:
        per_file += [{}] * (file_count - len(per_file))
    return per_file[:file_count]


def execute_request(
    request_index: int,
    paths: list[Path],
    durations: dict[Path, float],
    args: argparse.Namespace,
    api_key: str,
    extra_input: dict,
) -> list[FileRow]:
    payload = build_payload(paths, args.payload_style, args.language, extra_input)
    body_bytes = len(json.dumps(payload).encode("utf-8"))
    if body_bytes > PAYLOAD_WARN_BYTES:
        print(
            f"warning: request {request_index} body is {body_bytes / 1e6:.1f} MB; "
            "RunPod may reject it. Lower --files-per-request or use shorter clips.",
            file=sys.stderr,
        )

    started = time.perf_counter()
    error = ""
    response: dict = {}
    try:
        response = run_job(
            args.api_base.rstrip("/"),
            args.endpoint_id,
            api_key,
            payload,
            timeout=args.request_timeout,
            poll_interval=args.poll_interval,
        )
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # network error, timeout, bad JSON
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    status = str(response.get("status", "ERROR" if error else "UNKNOWN"))
    output = response.get("output")
    if not error:
        if status not in ("COMPLETED", "UNKNOWN"):
            error = json.dumps(response.get("error") or response)[:300]
        elif isinstance(output, dict) and output.get("error"):
            error = str(output["error"])[:300]

    per_file = _extract_per_file(output, len(paths))
    delay_ms = response.get("delayTime")
    exec_ms = response.get("executionTime")

    rows: list[FileRow] = []
    for path, result in zip(paths, per_file):
        file_error = error or str(result.get("error", ""))[:300]
        rows.append(FileRow(
            request_index=request_index,
            file=path.name,
            duration_seconds=durations[path],
            files_in_request=len(paths),
            request_wall_clock_seconds=elapsed,
            status="OK" if not file_error else "FAILED",
            inference_time=result.get("inference_time"),
            delay_time_ms=delay_ms if isinstance(delay_ms, (int, float)) else None,
            execution_time_ms=exec_ms if isinstance(exec_ms, (int, float)) else None,
            error=file_error,
        ))

    failed_rows = [r for r in rows if r.status == "FAILED"]
    if not failed_rows:
        done = "ok"
    else:
        first_error = failed_rows[0].error[:120]
        done = f"FAILED {len(failed_rows)}/{len(rows)} file(s) ({first_error})"
    print(
        f"  request {request_index:>3}  {len(paths)} file(s)  "
        f"{elapsed:7.2f}s  {done}",
        flush=True,
    )
    return rows


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

CSV_FIELDS = [
    "mode", "payload_style", "files_per_request", "concurrency",
    "request_index", "file", "duration_seconds", "files_in_request",
    "request_wall_clock_seconds", "status", "inference_time",
    "delay_time_ms", "execution_time_ms", "error",
]


def write_csv(path: Path, rows: list[FileRow], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "mode": args.mode,
                "payload_style": args.payload_style,
                "files_per_request": args.files_per_request,
                "concurrency": args.concurrency,
                "request_index": row.request_index,
                "file": row.file,
                "duration_seconds": f"{row.duration_seconds:.3f}",
                "files_in_request": row.files_in_request,
                "request_wall_clock_seconds": f"{row.request_wall_clock_seconds:.3f}",
                "status": row.status,
                "inference_time": "" if row.inference_time is None else f"{row.inference_time:.3f}",
                "delay_time_ms": "" if row.delay_time_ms is None else f"{row.delay_time_ms:.0f}",
                "execution_time_ms": "" if row.execution_time_ms is None else f"{row.execution_time_ms:.0f}",
                "error": row.error,
            })


def print_summary(
    rows: list[FileRow],
    metrics: RunMetrics,
    args: argparse.Namespace,
    request_count: int,
    failed: list[FileRow],
) -> None:
    per_request_wall = sorted({
        (r.request_index, r.request_wall_clock_seconds) for r in rows
    })
    walls = [w for _, w in per_request_wall]

    # Only successful audio is in the RTF denominator, so keep the numerators
    # consistent: skip failed files and any request that had a failure in it.
    ok_rows = [r for r in rows if r.status == "OK"]
    failed_requests = {r.request_index for r in failed}
    inference = [r.inference_time for r in ok_rows
                 if isinstance(r.inference_time, (int, float))]
    billed_exec = sum(
        ms for _, ms in {(r.request_index, r.execution_time_ms) for r in ok_rows
                         if r.request_index not in failed_requests
                         and isinstance(r.execution_time_ms, (int, float))}
    ) / 1000.0

    ok_files = len(ok_rows)
    print()
    print(f"### {args.mode} mode / {args.payload_style} payload / endpoint {args.endpoint_id}")
    print()
    print("| mode | payload | files/req | concurrency | files OK | requests | "
          "audio min | wall-clock s | billed RTF | $/audio-hour |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(
        f"| {args.mode} | {args.payload_style} | {args.files_per_request} | "
        f"{args.concurrency} | {ok_files} | {request_count} | "
        f"{metrics.total_audio_seconds / 60:.1f} | {metrics.wall_clock_seconds:.1f} | "
        f"{metrics.billed_rtf:.4f} | ${metrics.cost_per_audio_hour:.4f} |"
    )
    print()
    print(f"- gpu rate: ${metrics.gpu_rate_per_second:.6f}/s "
          f"(${metrics.gpu_rate_per_second * 3600:.4f}/h)")
    print(f"- this run cost: ${metrics.cost_of_run:.4f} "
          f"(client wall-clock x gpu rate)")
    if walls:
        print(f"- per-request wall-clock s: min {min(walls):.2f} / "
              f"median {statistics.median(walls):.2f} / max {max(walls):.2f}")
    if inference:
        print(f"- worker-reported inference_time s: sum {sum(inference):.1f} / "
              f"median {statistics.median(inference):.2f} "
              f"(inference-only RTF {sum(inference) / metrics.total_audio_seconds:.4f})")
    if billed_exec > 0:
        print(f"- RunPod executionTime sum: {billed_exec:.1f}s "
              f"(RTF {billed_exec / metrics.total_audio_seconds:.4f}) — "
              "per-job billed time, ignores idle gaps between jobs")
    if failed:
        print(f"- WARNING: {len(failed)} file(s) failed; RTF above counts only "
              "successful audio and is not directly comparable to a clean run")
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description=(
            "Measure billed RTF (wall-clock seconds per audio second) and "
            "cost per audio hour for a RunPod Whisper serverless endpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # this worker, 8 files per request, 4 requests in flight\n"
            "  python3 run_benchmark.py --audio-dir ./audio --endpoint-id abc123 \\\n"
            "      --mode batch --files-per-request 8 --concurrency 4 \\\n"
            "      --gpu-rate 0.00031 --out results/batch.csv\n"
            "\n"
            "  # official worker (runpod-workers/worker-faster_whisper): one file per job\n"
            "  python3 run_benchmark.py --audio-dir ./audio --endpoint-id def456 \\\n"
            "      --mode single --payload-style official --concurrency 4 \\\n"
            "      --gpu-rate 0.00031 --out results/official.csv\n"
        ),
    )
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="directory of audio files you prepared yourself (not in this repo)")
    parser.add_argument("--endpoint-id", required=True,
                        help="RunPod serverless endpoint id")
    parser.add_argument("--api-key", default=None,
                        help="RunPod API key (default: RUNPOD_API_KEY env var)")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch",
                        help="batch: several files per request. single: one file per request, "
                             "the shape the official worker forces (default: batch)")
    parser.add_argument("--payload-style", choices=["batch-worker", "official"],
                        default="batch-worker",
                        help="request body shape. batch-worker: audio_base64_list. "
                             "official: audio_base64 (default: batch-worker)")
    parser.add_argument("--files-per-request", type=int, default=8,
                        help="files packed into one request in batch mode (default: 8). "
                             "forced to 1 in single mode")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="requests in flight at once (default: 4)")
    parser.add_argument("--gpu-rate", type=float, default=0.00031,
                        help="endpoint GPU price in dollars per second (default: 0.00031)")
    parser.add_argument("--language", default="ko",
                        help="language code passed to the worker (default: ko)")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark.csv"),
                        help="per-file CSV output path (default: results/benchmark.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N audio files")
    parser.add_argument("--extra-input", action="append", default=[], metavar="KEY=VALUE",
                        help="extra field for the job input, repeatable "
                             "(e.g. --extra-input model=turbo)")
    parser.add_argument("--request-timeout", type=float, default=900.0,
                        help="seconds to wait for one job to finish (default: 900)")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="seconds between /status polls when runsync returns queued "
                             "(default: 2)")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"RunPod API base URL (default: {DEFAULT_API_BASE}); "
                             "override only to point at a stub for testing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise SystemExit("error: pass --api-key or set RUNPOD_API_KEY")

    if args.mode == "single":
        args.files_per_request = 1
    if args.files_per_request < 1:
        raise SystemExit("error: --files-per-request must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("error: --concurrency must be >= 1")
    if args.payload_style == "official" and args.files_per_request != 1:
        raise SystemExit(
            "error: --payload-style official supports one file per request; use --mode single"
        )

    extra_input = parse_extra_input(args.extra_input)
    files = find_audio_files(args.audio_dir, args.limit)

    print(f"probing {len(files)} audio file(s) with ffprobe ...", flush=True)
    durations = {path: probe_duration(path) for path in files}
    total_audio = sum(durations.values())
    groups = chunk(files, args.files_per_request)
    print(
        f"{len(files)} files, {total_audio / 60:.1f} audio minutes, "
        f"{len(groups)} request(s), concurrency {args.concurrency}",
        flush=True,
    )

    run_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        batches = list(pool.map(
            lambda item: execute_request(
                item[0], item[1], durations, args, api_key, extra_input
            ),
            enumerate(groups, start=1),
        ))
    wall_clock = time.perf_counter() - run_started

    rows = [row for batch in batches for row in batch]
    failed = [r for r in rows if r.status == "FAILED"]
    ok_audio = sum(r.duration_seconds for r in rows if r.status == "OK")

    write_csv(args.out, rows, args)
    print(f"\nwrote {args.out} ({len(rows)} row(s))")

    if ok_audio <= 0:
        print("\nevery request failed; no RTF to report. See the error column in the CSV.",
              file=sys.stderr)
        return 1

    metrics = compute_run_metrics(ok_audio, wall_clock, args.gpu_rate)
    print_summary(rows, metrics, args, len(groups), failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
