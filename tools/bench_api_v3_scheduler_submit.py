#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark api_v3 /tts_scheduler_submit concurrency and GPU memory.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:9880")
    parser.add_argument("--endpoint", type=str, default="/tts_scheduler_submit")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=0.1)
    parser.add_argument("--text-lang", type=str, default="zh")
    parser.add_argument("--prompt-lang", type=str, default="zh")
    parser.add_argument("--media-type", type=str, default="wav")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--text-file", type=Path, default=ROOT_DIR / "test_cn.txt")
    parser.add_argument("--wav-dir", type=Path, default=ROOT_DIR / "testwav")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "TEMP/api_v3_bench")
    return parser.parse_args()


def load_requests(args: argparse.Namespace) -> List[Dict[str, Any]]:
    wav_paths_all = sorted(args.wav_dir.glob("*.wav"))
    wav_paths: List[Path] = []
    for wav_path in wav_paths_all:
        with wave.open(str(wav_path), "rb") as handle:
            duration = handle.getnframes() / float(handle.getframerate())
        if 3.0 <= duration <= 10.0:
            wav_paths.append(wav_path)
    if not wav_paths:
        raise FileNotFoundError(f"没有找到 3-10 秒合法 wav: {args.wav_dir}")
    text_lines = [line.strip() for line in args.text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not text_lines:
        raise ValueError(f"没有找到有效文本行: {args.text_file}")

    requests: List[Dict[str, Any]] = []
    for index in range(args.concurrency):
        wav_path = wav_paths[index % len(wav_paths)]
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            raise FileNotFoundError(f"缺少参考文本: {lab_path}")
        requests.append(
            {
                "request_id": f"bench_{args.concurrency:03d}_{index:03d}",
                "text": text_lines[index % len(text_lines)],
                "text_lang": args.text_lang,
                "ref_audio_path": str(wav_path),
                "prompt_lang": args.prompt_lang,
                "prompt_text": lab_path.read_text(encoding="utf-8").strip(),
                "top_k": int(args.top_k),
                "top_p": float(args.top_p),
                "temperature": float(args.temperature),
                "repetition_penalty": float(args.repetition_penalty),
                "sample_steps": int(args.sample_steps),
                "media_type": args.media_type,
                "timeout_sec": float(args.timeout_sec),
            }
        )
    return requests


class GpuMemoryPoller:
    def __init__(self, server_pid: Optional[int], interval_sec: float):
        self.server_pid = server_pid
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None

    def _query_memory_mb(self) -> Optional[int]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        total = 0
        found = False
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                used_mb = int(parts[1])
            except ValueError:
                continue
            if self.server_pid is None or pid == self.server_pid:
                total += used_mb
                found = True
        if self.server_pid is None:
            return total
        return total if found else 0

    def _run(self) -> None:
        while not self._stop.is_set():
            used_mb = self._query_memory_mb()
            self.samples.append({"ts": time.time(), "used_mb": used_mb})
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def summary(self) -> Dict[str, Any]:
        valid = [item for item in self.samples if item["used_mb"] is not None]
        peak = max(valid, key=lambda item: item["used_mb"]) if valid else None
        first = valid[0] if valid else None
        last = valid[-1] if valid else None
        return {
            "server_pid": self.server_pid,
            "sample_count": int(len(self.samples)),
            "start_used_mb": None if first is None else int(first["used_mb"]),
            "peak_used_mb": None if peak is None else int(peak["used_mb"]),
            "peak_delta_mb": None if peak is None or first is None else int(peak["used_mb"] - first["used_mb"]),
            "end_used_mb": None if last is None else int(last["used_mb"]),
            "peak_ts": None if peak is None else float(peak["ts"]),
            "samples": self.samples,
        }


async def submit_one(client: httpx.AsyncClient, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        item = {
            "request_id": payload["request_id"],
            "status_code": int(response.status_code),
            "elapsed_ms": float(elapsed_ms),
            "content_type": response.headers.get("content-type"),
            "audio_bytes": int(len(response.content)),
            "headers": {key: value for key, value in response.headers.items() if key.lower().startswith("x-")},
        }
        if response.status_code != 200:
            try:
                item["error_body"] = response.json()
            except Exception:
                item["error_body"] = response.text
        return item
    except Exception as exc:
        return {
            "request_id": payload["request_id"],
            "status_code": -1,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
            "exception": repr(exc),
        }


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    payloads = load_requests(args)
    url = args.base_url.rstrip("/") + args.endpoint
    poller = GpuMemoryPoller(server_pid=args.server_pid, interval_sec=args.poll_interval_sec)

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=args.timeout_sec + 10.0, write=10.0, pool=10.0)

    started = time.perf_counter()
    poller.start()
    try:
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            results = await asyncio.gather(*[submit_one(client, url, payload) for payload in payloads])
    finally:
        poller.stop()
    wall_ms = (time.perf_counter() - started) * 1000.0

    ok_results = [item for item in results if item["status_code"] == 200]
    failed_results = [item for item in results if item["status_code"] != 200]
    request_total_ms = []
    worker_total_ms = []
    for item in ok_results:
        headers = item.get("headers", {})
        if "x-request-total-ms" in headers:
            request_total_ms.append(float(headers["x-request-total-ms"]))
        if "x-worker-total-ms" in headers:
            worker_total_ms.append(float(headers["x-worker-total-ms"]))

    return {
        "concurrency": int(args.concurrency),
        "server_pid": args.server_pid,
        "request_count": int(len(payloads)),
        "wall_ms": float(wall_ms),
        "success_count": int(len(ok_results)),
        "failure_count": int(len(failed_results)),
        "request_total_ms_avg": float(sum(request_total_ms) / len(request_total_ms)) if request_total_ms else None,
        "request_total_ms_max": float(max(request_total_ms)) if request_total_ms else None,
        "worker_total_ms_avg": float(sum(worker_total_ms) / len(worker_total_ms)) if worker_total_ms else None,
        "worker_total_ms_max": float(max(worker_total_ms)) if worker_total_ms else None,
        "gpu_memory": poller.summary(),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / f"concurrency_{args.concurrency:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(run_benchmark(args))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "concurrency": summary["concurrency"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "wall_ms": summary["wall_ms"],
        "gpu_peak_used_mb": summary["gpu_memory"]["peak_used_mb"],
        "request_total_ms_avg": summary["request_total_ms_avg"],
        "request_total_ms_max": summary["request_total_ms_max"],
        "summary_path": str(summary_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
