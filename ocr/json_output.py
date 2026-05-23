"""
JSON output helpers for OCR CLI.

Goal: mirror the existing console output structure in a machine-readable format.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import torch

from .config import CHARS


def _topk_from_probs(probs: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
    topk_probs, topk_indices = torch.topk(probs, k)
    out: list[dict[str, Any]] = []
    for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices), start=1):
        out.append(
            {
                "rank": rank,
                "char": CHARS[int(idx.item())],
                "probability_percent": round(float(prob.item()) * 100.0, 4),
            }
        )
    return out


def build_image_result_json(
    *,
    image_path: str,
    saved_copy_path: str | None,
    predicted_char: str,
    confidence: float,
    probs: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    return {
        "type": "image",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "input": {"image_path": image_path, "saved_copy_path": saved_copy_path},
        "result": {
            "predicted_char": predicted_char,
            "confidence_percent": round(float(confidence), 4),
            "top5": _topk_from_probs(probs, k=5),
        },
    }


def build_word_result_json(
    *,
    image_path: str,
    saved_copy_path: str | None,
    word: str,
    device: str,
) -> dict[str, Any]:
    return {
        "type": "word",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "input": {"image_path": image_path, "saved_copy_path": saved_copy_path},
        "result": {"word": word},
    }


def build_lines_result_json(
    *,
    image_path: str,
    saved_copy_path: str | None,
    text: str,
    device: str,
) -> dict[str, Any]:
    lines = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(text.splitlines())
        if line != ""
    ]
    return {
        "type": "lines",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "input": {"image_path": image_path, "saved_copy_path": saved_copy_path},
        "result": {
            "lines": lines,
        },
    }


def build_multi_result_json(
    *,
    results: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    return {
        "type": "multi",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "results": results,
    }

def build_page_result_json(
    *,
    image_path: str,
    saved_copy_path: str | None,
    rows: list[dict],
    device: str,
) -> dict[str, Any]:

    words = []

    for i, r in enumerate(rows):
        words.append({
            "index": i + 1,
            "key": r["key"],
            "file": r["file"],
            "text": r["text"],
            "confidence": r["confidence"],
        })

    full_text = " ".join(
        r["text"] for r in rows
        if r["text"]
    )

    confidences = [
        r["confidence"]
        for r in rows
        if r["confidence"] is not None
    ]

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else None
    )

    return {
        "type": "page",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "input": {
            "image_path": image_path,
            "saved_copy_path": saved_copy_path
        },
        "result": {
            "text": full_text,
            "confidence": avg_confidence,
            "words": words,
        },
    }

def dump_json(payload: dict[str, Any], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_json(path: str, payload: dict[str, Any], *, pretty: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_json(payload, pretty=pretty))
        f.write("\n")

