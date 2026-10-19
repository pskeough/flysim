# Third-party data, models and software

flysim's own code is under the MIT licence in `LICENSE`. Everything below belongs to its authors and keeps its own terms. "Shipped" means the file is in this repository. "Downloaded" means `install.bat` fetches it from the source named here.

## Connectome and neuron data

| What | Where it comes from | Shipped or downloaded | Terms |
|---|---|---|---|
| `Connectivity_783.parquet`, `Completeness_783.csv` (FlyWire v783 synapse counts and signs, as packaged for the Shiu et al. 2024 model) | github.com/philshiu/Drosophila_brain_model, commit 91bdd1e7 | downloaded to `third_party/Drosophila_brain_model/` | The repository's code is MIT (Philip Shiu and Nico Spiller; copy in `third_party/Drosophila_brain_model/LICENSE`). The repository states no separate licence for the data files. |
| `Supplemental_file1_neuron_annotations.tsv` (cell types, classes, soma positions) | github.com/flyconnectome/flywire_annotations, commit 17fc5772 | downloaded to `data/flywire/` | The repository has no LICENSE file. Its README asks users to cite Schlegel et al. 2024 and Dorkenwald et al. 2024. |
| `neurons.csv.gz`, `classification.csv.gz`, `coordinates.csv.gz` (FlyWire Codex v783 exports: neurotransmitter, neuropil, super-class, positions) | codex.flywire.ai (download needs a Google sign-in) | shipped in `data/flywire/` | No licence is stated on the Codex download pages. The FlyWire credits page says the underlying EM images (Zheng et al. 2018) were released under CC BY-NC 4.0. |
| `data/shiu_named_sets.json`, `data/noci_stand_in.json` | neuron ID lists written for this project from the files above | shipped | MIT, with the FlyWire citations below |

Cite these when you use flysim's brain or anything built on it:

- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124-138. doi:10.1038/s41586-024-07558-y
- Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139-152. doi:10.1038/s41586-024-07686-5
- Shiu, P. K. et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634, 210-219. doi:10.1038/s41586-024-07763-9
- Zheng, Z. et al. (2018). A complete electron microscopy volume of the brain of adult *Drosophila melanogaster*. *Cell* 174, 730-743. doi:10.1016/j.cell.2018.06.019
- Matsliah, A. et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634, 166-180. doi:10.1038/s41586-024-07981-1 (optic-lobe cell types in the annotation file)

## Body

| What | Source | Terms |
|---|---|---|
| flygym 2.1.0 (NeuroMechFly v2): the MuJoCo fly model, the CPG walking controller the body reimplements | pip, github.com/NeLy-EPFL/flygym | Apache-2.0. Cite Wang-Chen, S. et al. (2024), NeuroMechFly v2, *Nature Methods*, doi:10.1038/s41592-024-02497-y |
| FlyBody model and full-size meshes (fetched by flygym from datasets.epfl.ch into `%USERPROFILE%\.cache\flygym_assets`) | github.com/TuragaLab/flybody | The flybody repository is Apache-2.0; no separate licence is stated for the meshes. Cite Vaxenburg, R. et al. (2025), Whole-body physics simulation of fruit fly locomotion, *Nature*, doi:10.1038/s41586-025-09029-4 |
| `data/lite_body.npz` (drive-to-motion calibration for the other flies, measured on the FlyBody model) | made for this project | MIT |
| MuJoCo 3.9.0 | pip, github.com/google-deepmind/mujoco | Apache-2.0 |

## Mind

| What | Source | Terms |
|---|---|---|
| Qwen3-4B, as the Q4_K_M GGUF `Qwen_Qwen3-4B-Q4_K_M.gguf` (sha256 fbe1d5ed...bae3) | downloaded from huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF (quantisation of Qwen/Qwen3-4B) | Apache-2.0 (Qwen/Qwen3-4B model card and LICENSE) |
| Steering vectors and calibrations in `pain_server/vectors/` | extracted on that GGUF for this project | MIT |
| llama.cpp release b10435 (llama.dll, ggml) and the CUDA runtime DLLs from the same release | downloaded from github.com/ggml-org/llama.cpp/releases/tag/b10435 into `llama/` | llama.cpp: MIT. The `cudart-llama-bin-*.zip` DLLs (cudart, cuBLAS) are NVIDIA CUDA redistributables under the NVIDIA CUDA EULA |
| OpenRouter (only with `run.bat --openrouter`) | openrouter.ai | the provider's terms; you supply your own key |

## Simulator libraries

| What | Terms |
|---|---|
| NVIDIA Warp 1.14.0 (`warp-lang`) | Apache-2.0. The PyPI wheels also link NVIDIA libmathdx, which is under the NVIDIA Software License Agreement (Warp README) |
| numba, llvmlite, numpy, pandas, pyarrow, scipy, aiohttp | BSD-2/BSD-3, Apache-2.0 and similar permissive licences, installed by pip |
| three.js r186 (`web/vendor/`) | MIT, Copyright 2010-2026 Three.js Authors (header in each file) |
| Inter and IBM Plex Mono fonts | loaded from Google Fonts at run time; SIL Open Font License 1.1 |

## Voice (optional, `install.bat --voice`)

| What | Source | Terms |
|---|---|---|
| Kokoro-82M (`kokoro-v1_0.pth`, voice `am_puck`) | huggingface.co/hexgrad/Kokoro-82M, commit f3ff3571 | Apache-2.0 |
| kokoro 0.9.4, misaki 0.9.4 | pip | Apache-2.0 |
| faster-whisper 1.2.1 | pip, github.com/SYSTRAN/faster-whisper | MIT |
| Whisper small.en, CTranslate2 conversion | huggingface.co/Systran/faster-whisper-small.en, commit d1d751a5 | MIT |
| spaCy 3.8 and `en_core_web_sm` 3.8.0 | pip, github.com/explosion/spacy-models | MIT |
| PyTorch 2.11.0 (CPU build) | download.pytorch.org | BSD-3-Clause |
| FFmpeg (the binary inside `imageio-ffmpeg`, used for the voice effects) | pip | LGPL/GPL, see the imageio-ffmpeg package |
| Qwen3-TTS VoiceDesign through audio.cpp (not installed; see README) | audio.cpp v0.8.0 by ShugoAI LLC | audio.cpp: Apache-2.0; the Qwen3-TTS weights carry their own licence |
