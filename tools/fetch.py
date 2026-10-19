"""Downloads for install.bat: the model weights, llama.cpp, and the connectome data, each pinned by URL, size and
SHA-256. Standard library only, so it runs before any venv exists.

    python tools\\fetch.py detect                 print the llama.cpp build to use: cuda13 | cuda12 | cpu
    python tools\\fetch.py disk GB                exit 1 if the drive holding this folder has less than GB free
    python tools\\fetch.py get ITEM [ITEM ...]    download (resumable), check size and SHA-256, unzip where needed
    python tools\\fetch.py get ITEM --dry-run     print what would be fetched and check each URL answers with the
                                                 expected size; nothing is written
    python tools\\fetch.py list                   the items

A file that is already present with the right size and hash is not fetched again; a partial download (*.part)
resumes. A finished item leaves <file>.ok holding its hash, so later runs skip the re-hash.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_TAG = "b10435"            # pain_server/llama_steer.py's ctypes offsets are for this llama.cpp tag
GH = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
SHIU = "https://github.com/philshiu/Drosophila_brain_model/raw/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOT = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
         "17fc57722002e1a7d38cdd0c89ac382bf92718da/supplemental_files")

# name: (url, bytes, sha256, destination relative to ROOT, unzip into (or None))
ITEMS = {
    # Qwen3-4B Q4_K_M: the exact file the steering vectors in pain_server/vectors were made on
    "model": (
        "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/cb76885dc66d50759b207c5a48c4e78dfa00c638/"
        "Qwen_Qwen3-4B-Q4_K_M.gguf",
        2_497_280_960, "fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3",
        "models/Qwen3-4B-Q4_K_M.gguf", None),
    # llama.cpp b10435, official Windows builds
    "llama-cuda13": (f"{GH}/llama-{LLAMA_TAG}-bin-win-cuda-13.3-x64.zip", 146_711_620,
                     "b6ad54c7fd978e8cec9624d65c402f6fd0cd2a54104601bc273d8ae419f33111",
                     "downloads/llama-cuda-13.3.zip", "llama"),
    "cudart-cuda13": (f"{GH}/cudart-llama-bin-win-cuda-13.3-x64.zip", 390_970_417,
                      "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
                      "downloads/cudart-cuda-13.3.zip", "llama"),
    "llama-cuda12": (f"{GH}/llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", 250_790_810,
                     "2fa0134d20cca7214a09ca74442eaa428d25886a318cb46d23cebfd070210b34",
                     "downloads/llama-cuda-12.4.zip", "llama"),
    "cudart-cuda12": (f"{GH}/cudart-llama-bin-win-cuda-12.4-x64.zip", 391_443_627,
                      "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                      "downloads/cudart-cuda-12.4.zip", "llama"),
    "llama-cpu": (f"{GH}/llama-{LLAMA_TAG}-bin-win-cpu-x64.zip", 18_467_736,
                  "17f7daed898c92bc4c3459e838b98a7910dc92c60062323da840e27b93894a30",
                  "downloads/llama-cpu.zip", "llama"),
    # FlyWire v783 connectome as packaged by Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model)
    "connectivity": (f"{SHIU}/Connectivity_783.parquet", 100_804_642,
                     "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
                     "third_party/Drosophila_brain_model/Connectivity_783.parquet", None),
    "completeness": (f"{SHIU}/Completeness_783.csv", 3_327_347,
                     "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
                     "third_party/Drosophila_brain_model/Completeness_783.csv", None),
    # FlyWire neuron annotations, Schlegel et al. 2024 (github.com/flyconnectome/flywire_annotations)
    "annotations": (f"{ANNOT}/Supplemental_file1_neuron_annotations.tsv", 31_718_505,
                    "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
                    "data/flywire/Supplemental_file1_neuron_annotations.tsv", None),
}
GROUPS = {
    "cuda13": ["llama-cuda13", "cudart-cuda13"],
    "cuda12": ["llama-cuda12", "cudart-cuda12"],
    "cpu": ["llama-cpu"],
    "data": ["connectivity", "completeness", "annotations"],
}


def say(*a):
    print(*a, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def gb(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def verified(dest: Path, size: int, sha: str) -> bool:
    """True when dest is complete and matches; checks the .ok marker first, then hashes."""
    if not dest.exists() or dest.stat().st_size != size:
        return False
    ok = dest.with_name(dest.name + ".ok")
    if ok.exists() and ok.read_text().strip() == sha:
        return True
    say(f"  checking {dest.name} ...")
    if sha256(dest) == sha:
        ok.write_text(sha)
        return True
    return False


def download(url: str, dest: Path, size: int, sha: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"User-Agent": "flysim-installer"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        say(f"  resuming at {gb(have)}")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:          # the server ignored the range: start over
            have = 0
        mode = "ab" if have else "wb"
        t0, last, done = time.time(), 0.0, have
        with open(part, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2 or done == size:
                    last = time.time()
                    rate = (done - have) / max(time.time() - t0, 1e-6)
                    say(f"  {gb(done)} / {gb(size)}  ({100 * done / size:.0f}%, {rate / 1e6:.1f} MB/s)")
    if part.stat().st_size != size:
        raise SystemExit(f"  {dest.name}: got {part.stat().st_size} bytes, expected {size}. Run install.bat again "
                         f"to resume.")
    say(f"  checking {dest.name} ...")
    got = sha256(part)
    if got != sha:
        part.unlink()
        raise SystemExit(f"  {dest.name}: SHA-256 mismatch (got {got}, expected {sha}). The partial file was "
                         f"deleted; run install.bat again.")
    part.replace(dest)
    dest.with_name(dest.name + ".ok").write_text(sha)


def unzip(zp: Path, into: Path):
    mark = into / f".unzipped-{zp.stem}"
    if mark.exists():
        return
    say(f"  unzipping {zp.name} -> {into.relative_to(ROOT)}\\")
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            name = Path(m.filename).name               # flat: the llama.cpp build zips hold one flat folder
            if m.is_dir() or not name:
                continue
            with z.open(m) as src, open(into / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    mark.write_text("ok")


def head(url: str):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "flysim-installer"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, int(r.headers.get("Content-Length") or -1), r.url


def get(names, dry=False):
    todo = []
    for n in names:
        todo += GROUPS.get(n, [n])
    for n in todo:
        if n not in ITEMS:
            raise SystemExit(f"unknown item {n}; have {sorted(ITEMS) + sorted(GROUPS)}")
        url, size, sha, rel, into = ITEMS[n]
        dest = ROOT / rel
        if dry:
            try:
                st, cl, final = head(url)
                note = "OK" if cl == size else f"SIZE DIFFERS (server says {cl})"
                say(f"[{n}] {url}\n    -> {rel}  {gb(size)}  sha256 {sha[:16]}...  HTTP {st}  {note}")
            except Exception as e:
                say(f"[{n}] {url}\n    -> {rel}  {gb(size)}  UNREACHABLE: {e}")
            continue
        say(f"[{n}] {gb(size)} -> {rel}")
        if into and (ROOT / into / f".unzipped-{dest.stem}").exists():
            say(f"  already unzipped into {into}\\")
            continue
        if not verified(dest, size, sha):
            download(url, dest, size, sha)
        else:
            say("  already here")
        if into:
            unzip(dest, ROOT / into)
            dest.unlink()                              # the zip is no longer needed; the .ok marker stays
            say("  done (zip removed)")


def detect():
    """The llama.cpp build for this machine: cuda13 needs driver >= 580 and compute capability >= 7.5 (Turing or
    newer); cuda12 needs driver >= 551.61; otherwise cpu."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "cpu", "no nvidia-smi (no NVIDIA driver)"
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version,compute_cap,memory.total,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        first = out.strip().splitlines()[0]
        drv, cc, mem, name = [x.strip() for x in first.split(",", 3)]
        dv = tuple(int(x) for x in drv.split(".")[:2])
        ccv = tuple(int(x) for x in cc.split("."))
    except Exception as e:
        return "cpu", f"nvidia-smi gave no usable answer ({e})"
    info = f"{name}, {mem} MiB, driver {drv}, compute {cc}"
    if dv >= (580, 0) and ccv >= (7, 5):
        return "cuda13", info
    if dv >= (551, 61):
        return "cuda12", info
    return "cpu", info + " (driver too old for the CUDA builds; update it from nvidia.com for GPU speed)"


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        say(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "detect":
        build, info = detect()
        print(build)
        print(info, file=sys.stderr)
        return 0
    if cmd == "disk":
        need = float(rest[0])
        free = shutil.disk_usage(ROOT).free / 1e9
        if free < need:
            say(f"only {free:.1f} GB free on the drive holding {ROOT}; this step needs about {need:.0f} GB")
            return 1
        say(f"{free:.1f} GB free (needs about {need:.0f} GB)")
        return 0
    if cmd == "list":
        for n, (url, size, sha, rel, into) in ITEMS.items():
            say(f"{n:15s} {gb(size):>9s}  {rel}{'  (unzipped into ' + into + ')' if into else ''}")
        say("groups: " + ", ".join(f"{k} = {' + '.join(v)}" for k, v in GROUPS.items()))
        return 0
    if cmd == "get":
        dry = "--dry-run" in rest
        get([x for x in rest if not x.startswith("--")], dry=dry)
        return 0
    say(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
