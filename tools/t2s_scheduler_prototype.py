#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
gpt_sovits_dir = ROOT_DIR / "GPT_SoVITS"
if str(gpt_sovits_dir) not in sys.path:
    sys.path.append(str(gpt_sovits_dir))

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (  # noqa: E402
    SchedulerRequestSpec,
    T2SFinishedItem,
    T2SRequestState,
    prepare_request_state,
    run_scheduler_continuous,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2S request-local scheduler prototype.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "GPT_SoVITS/configs/tts_infer.yaml")
    parser.add_argument("--request-manifest", type=Path, default=None)
    parser.add_argument("--ref-audio", type=Path, default=ROOT_DIR / "test.wav")
    parser.add_argument("--prompt-text", type=str, default="是啊，主要是因为有调研需求的学者少了。")
    parser.add_argument("--prompt-lang", type=str, default="zh")
    parser.add_argument("--text-file", type=Path, default=ROOT_DIR / "test_en.txt")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-lang", type=str, default="en")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--early-stop-num", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "TEMP/t2s_scheduler/output_run")
    return parser.parse_args()


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pipeline(config_path: Path):
    try:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少运行依赖，请先在 GPT-SoVITS 推理环境中安装 requirements 后再运行该脚本。"
        ) from exc
    tts_config = TTS_Config(str(config_path))
    print(tts_config)
    return TTS(tts_config)


def load_request_specs(args: argparse.Namespace) -> List[SchedulerRequestSpec]:
    if args.request_manifest is not None:
        payload = json.loads(args.request_manifest.read_text(encoding="utf-8"))
        raw_requests = payload["requests"] if isinstance(payload, dict) else payload
        specs: List[SchedulerRequestSpec] = []
        for index, item in enumerate(raw_requests):
            text = item.get("text")
            text_file = item.get("text_file")
            if text is None and text_file is None:
                raise ValueError(f"request[{index}] must provide text or text_file")
            if text is None:
                text = Path(text_file).read_text(encoding="utf-8")
            specs.append(
                SchedulerRequestSpec(
                    request_id=item.get("request_id", f"req_{index:03d}"),
                    ref_audio_path=Path(item["ref_audio_path"]),
                    prompt_text=item["prompt_text"],
                    prompt_lang=item.get("prompt_lang", "zh"),
                    text=text,
                    text_lang=item.get("text_lang", "zh"),
                    top_k=int(item.get("top_k", args.top_k)),
                    top_p=float(item.get("top_p", args.top_p)),
                    temperature=float(item.get("temperature", args.temperature)),
                    repetition_penalty=float(item.get("repetition_penalty", args.repetition_penalty)),
                    early_stop_num=int(item.get("early_stop_num", args.early_stop_num)),
                    ready_step=int(item.get("ready_step", 0)),
                )
            )
        return specs

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    return [
        SchedulerRequestSpec(
            request_id="req_000",
            ref_audio_path=args.ref_audio,
            prompt_text=args.prompt_text,
            prompt_lang=args.prompt_lang,
            text=text,
            text_lang=args.text_lang,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            early_stop_num=args.early_stop_num,
            ready_step=0,
        )
    ]


def summarise_requests(states: List[T2SRequestState]) -> List[Dict[str, Any]]:
    return [
        {
            "request_id": state.request_id,
            "ready_step": int(state.ready_step),
            "ref_audio_path": str(state.ref_audio_path),
            "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
            "all_phone_len": int(state.all_phones.shape[0]),
            "bert_len": int(state.all_bert_features.shape[-1]),
            "norm_text": state.norm_text,
        }
        for state in states
    ]


def summarise_finished(items: List[T2SFinishedItem]) -> List[Dict[str, Any]]:
    return [
        {
            "request_id": item.request_id,
            "semantic_len": int(item.semantic_tokens.shape[0]),
            "finish_idx": int(item.finish_idx),
            "finish_reason": item.finish_reason,
        }
        for item in items
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tts = load_pipeline(args.config)
    model = tts.t2s_model.model
    use_cuda = str(tts.configs.device).startswith("cuda")
    set_seed(args.seed, use_cuda)

    request_specs = load_request_specs(args)
    states = [prepare_request_state(tts, spec) for spec in request_specs]
    finished = run_scheduler_continuous(model, states, max_steps=args.max_steps)

    summary = {
        "request_count": len(states),
        "max_steps": args.max_steps,
        "requests": summarise_requests(states),
        "finished": summarise_finished(finished),
    }
    output_path = args.output_dir / "scheduler_prototype_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1) from None
