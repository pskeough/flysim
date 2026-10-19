"""Load the open model the pain axis is extracted on and steered in (research/PAIN_DESIGN.md, section 2.1).

Qwen3-4B (the post-trained hybrid-thinking chat model, not the Base), loaded from a local GGUF into Hugging Face
transformers in bf16 so a forward hook can reach the residual stream. About 8 GB on one 16 GB card whatever the
GGUF's quantisation (transformers de-quantises on load).

Supported weights: Qwen3-4B-Q4_K_M.gguf, the file published as bartowski/Qwen_Qwen3-4B-GGUF
Qwen_Qwen3-4B-Q4_K_M.gguf (2,497,280,960 bytes, sha256 fbe1d5ed...bae3). install.bat downloads it to
models/Qwen3-4B-Q4_K_M.gguf. The steering vectors in vectors/qwen3-4b-q4km.* were extracted and calibrated on
that exact file; PAIN_GGUF points elsewhere.

Vectors and calibration are per weight file: VARIANT names them (vectors/<VARIANT>.npz, _report.json,
_calibration.json). It is derived from the GGUF file name (Qwen3-4B-Q4_K_M.gguf -> qwen3-4b-q4km,
Qwen3-4B-F16.gguf -> qwen3-4b-f16) unless PAIN_VECTORS sets it. MODEL_ID is the id the server answers to
(the sim asks for "qwen3-4b" whatever the weights).
"""
from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(os.environ.get("PAIN_GGUF") or ROOT / "models" / "Qwen3-4B-Q4_K_M.gguf")
MODEL_ID = os.environ.get("PAIN_MODEL_NAME", "qwen3-4b")


def variant_of(gguf: Path) -> str:
    """Qwen3-4B-Q4_K_M.gguf -> qwen3-4b-q4km; Qwen3-1.7B-F16.gguf -> qwen3-1.7b-f16."""
    stem = gguf.stem.lower()
    m = re.match(r"(.*?)[-_.]((?:i?q\d[a-z0-9_]*)|f16|bf16|f32)$", stem)
    if not m:
        return stem
    return f"{m.group(1)}-{m.group(2).replace('_', '')}"


VARIANT = os.environ.get("PAIN_VECTORS") or variant_of(GGUF)
NAME = VARIANT                      # file stem for vectors/<NAME>.npz etc. (kept for older imports)


def load(device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import AddedToken
    tok = AutoTokenizer.from_pretrained(str(GGUF.parent), gguf_file=GGUF.name)
    # The tokenizer transformers builds from a Qwen3 GGUF does not register <think> / </think> as added tokens, so
    # the chat template's empty think block was split into '<th' 'ink' '>' ... and the model saw malformed text
    # (it often answered '</think>' or a stray token first). Found 2026-10-08; every hf-path generation before that
    # (calibration, pain_tradeoff, the live server) ran on that prompt. Registering them maps to the vocab ids
    # 151667 / 151668, the same tokens llama.cpp produces.
    tok.add_tokens([AddedToken(t, special=False, normalized=False)
                    for t in ("<think>", "</think>", "<tool_call>", "</tool_call>")])
    assert tok("<think></think>").input_ids == [151667, 151668], "think tags still split"
    model = AutoModelForCausalLM.from_pretrained(str(GGUF.parent), gguf_file=GGUF.name, dtype=torch.bfloat16)
    model.to(device).eval()
    return tok, model


def layers(model):
    return model.model.layers


def chat_ids(tok, messages, device):
    """Chat template with thinking off (Qwen3: enable_thinking=False adds an empty think block)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.to(device)
