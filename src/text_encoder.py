"""
PGR-3D Text Encoder (T5-base)
-----------------------------
Encodes captions to T5 embeddings for CaptionHead training.

T5-base: google-t5/t5-base, 768-dim encoder, 512-token context.
Mean-pooling over token sequence gives a fixed-size [768] embedding
regardless of caption length — this is the CaptionHead prediction target.

Usage:
    from text_encoder import TextEncoder
    enc = TextEncoder(device="cuda")
    emb = enc.encode_pooled("A red toy fire truck …")   # [768]
    emb = enc.encode("A red toy fire truck …")           # [n_tokens, 768]

CLI (pre-compute for all objects):
    python src/text_encoder.py \
        --caption_dir data/gso30_renders \
        --objects alarm backpack ...
    Writes {obj}/t5_emb.pt per object (mean-pooled [768] tensor).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer

MODEL_ID = "google-t5/t5-base"
MAX_LENGTH = 512

_encoder   = None
_tokenizer = None


def get_t5(device: str = "cpu"):
    global _encoder, _tokenizer
    if _encoder is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _encoder   = T5EncoderModel.from_pretrained(MODEL_ID).to(device).eval()
    return _encoder, _tokenizer


@torch.no_grad()
def encode_text(caption: str, device: str = "cpu") -> torch.Tensor:
    """
    Encode caption to raw token sequence embeddings.
    Returns [n_tokens, 768] float32 on CPU.
    """
    enc, tok = get_t5(device)
    ids = tok(
        caption,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    ).to(device)
    out = enc(**ids)                     # last_hidden_state: [1, n_tokens, 768]
    return out.last_hidden_state.squeeze(0).cpu()   # [n_tokens, 768]


@torch.no_grad()
def encode_text_pooled(caption: str, device: str = "cpu") -> torch.Tensor:
    """
    Encode caption and mean-pool over token dimension.
    Returns [768] float32 on CPU, L2-normalised.
    """
    seq = encode_text(caption, device=device)   # [n_tokens, 768]
    pooled = seq.mean(dim=0)                    # [768]
    return F.normalize(pooled, dim=0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode captions to T5 embeddings")
    parser.add_argument("--caption_dir", required=True, help="Root dir with {obj}/caption.txt")
    parser.add_argument("--output_dir",  default=None,  help="Root dir to write {obj}/t5_emb.pt (default: same as caption_dir)")
    parser.add_argument("--objects",     nargs="+", default=None)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    caption_dir = Path(args.caption_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else caption_dir

    if args.objects:
        objects = args.objects
    else:
        objects = sorted(d.name for d in caption_dir.iterdir() if d.is_dir())

    print(f"Encoding {len(objects)} captions with T5-base …", flush=True)
    get_t5(args.device)   # warm up

    n_done, n_skip, n_fail = 0, 0, 0
    for obj in objects:
        out_path = output_dir / obj / "t5_emb.pt"
        if out_path.exists():
            print(f"  SKIP  {obj}", flush=True)
            n_skip += 1
            continue

        caption_path = caption_dir / obj / "caption.txt"
        if not caption_path.exists():
            print(f"  FAIL  {obj}  (no caption.txt)", flush=True)
            n_fail += 1
            continue

        caption = caption_path.read_text(encoding="utf-8").strip()
        try:
            emb = encode_text_pooled(caption, device=args.device)   # [768]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, out_path)
            print(f"  OK    {obj}  emb={emb.shape}", flush=True)
            n_done += 1
        except Exception as e:
            print(f"  FAIL  {obj}: {e}", flush=True)
            n_fail += 1

    print(f"\nDone: {n_done} encoded, {n_skip} skipped, {n_fail} failed.")


if __name__ == "__main__":
    main()
