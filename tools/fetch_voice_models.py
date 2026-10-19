"""Puts the voice server's model files into the Hugging Face cache, so voice/fly_voice_server.py can run offline
(it sets HF_HUB_OFFLINE=1 before loading Kokoro). Run with the voice venv; install.bat --voice does this.

    .venv_voice\\Scripts\\python.exe tools\\fetch_voice_models.py [--dry-run]

Kokoro-82M (hexgrad/Kokoro-82M): config.json, kokoro-v1_0.pth (327 MB), voices/am_puck.pt
faster-whisper small.en (Systran/faster-whisper-small.en): config.json, model.bin (484 MB), tokenizer, vocabulary
Both are pinned to a commit. The cache's refs/main is pointed at that commit, which is what the server's offline
lookups resolve ("main").
"""
from __future__ import annotations

import sys
from pathlib import Path

PINS = {
    "hexgrad/Kokoro-82M": ("f3ff3571791e39611d31c381e3a41a3af07b4987",
                           ["config.json", "kokoro-v1_0.pth", "voices/am_puck.pt"]),
    "Systran/faster-whisper-small.en": ("d1d751a5f8271d482d14ca55d9e2deeebbae577f",
                                        ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]),
}


def main():
    dry = "--dry-run" in sys.argv
    from huggingface_hub import snapshot_download
    for repo, (rev, files) in PINS.items():
        print(f"[{repo}] @ {rev[:12]}: {', '.join(files)}", flush=True)
        if dry:
            continue
        snap = Path(snapshot_download(repo, revision=rev, allow_patterns=files))
        main_ref = snap.parents[1] / "refs" / "main"
        main_ref.parent.mkdir(parents=True, exist_ok=True)
        cur = main_ref.read_text().strip() if main_ref.exists() else ""
        cur_snap = snap.parent / cur if cur else None
        if not cur or not all((cur_snap / f).exists() for f in files):
            main_ref.write_text(rev)      # an existing "main" that already holds these files is left alone
        missing = [f for f in files if not (snap / f).exists()]
        if missing:
            raise SystemExit(f"{repo}: missing after download: {missing}")
        print(f"  ok: {snap}", flush=True)


if __name__ == "__main__":
    main()
