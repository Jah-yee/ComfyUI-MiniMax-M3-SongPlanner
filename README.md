# ComfyUI MiniMax M3 SongPlanner

**Turn any creative idea into MiniMax Music 3 caption + lyrics — locally, no API keys.**

Uses ComfyUI's native CLIP text-generation stack (the same one powering the core
*Generate Text* node). A local VLM (Qwen3-VL, Gemma-3-Vision) writes both the
structured music description and original lyrics, then both outputs wire directly
to `MiniMaxMusic3TextEncode`.

---

## Quick Start

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/benjiyaya/ComfyUI-MiniMax-M3-SongPlanner.git minimax-m3-planner
```

Restart ComfyUI. The node appears under **M3/SongPlanner**.

### Requirements

- A **recent ComfyUI** with the V3 node API (`comfy_api.latest`) and native
  text-generation CLIP stack (`comfy_extras/nodes_textgen.py`). Update if unsure.
- A **generative VLM** text-encoder checkpoint (`.safetensors` repack) in
  `ComfyUI/models/text_encoders/`:
  - **Qwen3-VL** instruct repacks (recommended, verified generative)
  - **Gemma-3-Vision** instruct repacks (≤12B, also verified)
  - Qwen2.5-VL repacks are vision-tower-only and NOT generative — do not use.

No `pip install` needed. `requirements.txt` is comments only.

---

## The Pipeline

```
User idea ("dark K-pop song about neon rain")
      ↓
  ┌──────────────────────────────────┐
  │  Stage 1: Caption Generation     │  ← VLM reads idea, writes
  │  System: caption_system.txt      │     structured music description
  └──────────────┬───────────────────┘
                 ↓
            caption STRING
                 ↓
  ┌──────────────────────────────────┐
  │  Stage 2: Lyrics Generation      │  ← VLM reads idea + caption,
  │  System: lyrics_system.txt       │     writes original lyrics
  └──────────────┬───────────────────┘
                 ↓
            lyrics STRING
                 ↓
    Wire both to MiniMaxMusic3TextEncode
```

---

## Node Reference

### M3 Song Planner (Local VLM) — `M3SongPlanner`

| Input | Type | Default | Notes |
|---|---|---|---|
| `clip` | CLIP (optional) | — | Connect from CLIPLoader; overrides dropdown |
| `text_encoder` | combo | `<none>` | VLM checkpoint from `models/text_encoders/` |
| `idea` | string (multiline) | `""` | Your creative idea — any length |
| `genre_hint` | string | `""` | Optional genre/mood hint |
| `vocal_config` | combo | `auto` | female / male / duet / instrumental / auto |
| `language` | combo | `English` | English / Chinese / Korean / Japanese / auto |
| `duration_seconds` | float | `120.0` | Target song length (30-300s) |
| `seed` | int | `0` | `control_after_generate: randomize` |
| `temperature` | float | `0.8` | 0 = greedy |
| `top_p` / `top_k` | float / int | `0.95` / `64` | Sampling |
| `max_tokens` | int | `2048` | Per generation (caption and lyrics each) |
| `keep_model_loaded` | bool | `True` | Cache CLIP between runs |

**Outputs:**

| Output | Type | Wire to |
|---|---|---|
| `caption` | STRING | → `MiniMaxMusic3TextEncode` caption input |
| `lyrics` | STRING | → `MiniMaxMusic3TextEncode` lyrics input |
| `debug` | STRING | JSON with timings, clip source, warnings |

---

## System Prompts

Both system prompts are hot-reloaded from disk on every run — edit them without
restarting ComfyUI:

- `prompts/caption_system.txt` — controls caption generation (structured caption format)
- `prompts/lyrics_system.txt` — controls lyrics generation (section tags, language, structure)

### Caption format

The caption system prompt enforces Music 3's three-section structure:

```
### Global Metadata
[genre, tempo, emotional progression, production profile]

### Vocal Details
[lead vocal, timbre, delivery, harmony, effects]

### Arrangement
[section-by-section instrument timeline, 200-400 words]
```

### Lyrics format

The lyrics system prompt enforces section-tagged structure:

```
[Intro]
[Verse]
...
[Chorus]
...
[Bridge]
...
[Outro]
```

Structure scales with duration: short songs get fewer sections, extended songs
get instrumentals and solos.

---

## Example Workflow

```
CLIPLoader (Qwen3-VL-4B) ──CLIP──→ M3 Song Planner ←── idea: "dark K-pop song about neon rain"
                                       ↓
                              ┌── caption STRING ──→ MiniMaxMusic3TextEncode (caption)
                              │
                              └── lyrics STRING  ──→ MiniMaxMusic3TextEncode (lyrics)
                                                        ↓
                                                   KSampler → VAEDecodeAudio → SaveAudio
```

---

## How It Works

The node uses ComfyUI's native CLIP generation pattern:

```python
tokens = clip.tokenize(prompt, skip_template=True, min_length=1)
generated_ids = clip.generate(tokens, do_sample=True, seed=seed,
                              temperature=temp, max_length=max_tokens)
text = clip.decode(generated_ids)
```

This is the same pattern used by:
- ComfyUI's core `Generate Text` node (`comfy_extras/nodes_textgen.py`)
- [ComfyUI-H3-VisionPromptor](https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor)

No API calls, no Ollama, no external processes. Everything runs locally in one
ComfyUI install.

---

## Troubleshooting

- **`The CLIP ... does not support native text generation`** — the checkpoint is
  not a generative VLM (e.g. a plain SD/SDXL CLIP or a Qwen2.5-VL vision-tower-only
  repack). Use a Qwen3-VL or Gemma-3-Vision instruct repack.
- **`clip.tokenize/generate rejected the native arguments`** — ComfyUI is too old.
  Update ComfyUI.
- **VRAM notes** — a 4-7B VLM typically needs ~5-16 GB depending on dtype.
  Set `keep_model_loaded = False` to release VRAM between runs.

---

## Credits

- Engine pattern adapted from
  [ComfyUI-H3-VisionPromptor](https://github.com/benjiyaya/ComfyUI-H3-VisionPromptor)
- Caption format based on
  [MiniMax Music 3 model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- Official caption-rewriter skill:
  [MiniMax-AI/MiniMax-Music3/skills](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills)

License: MIT
