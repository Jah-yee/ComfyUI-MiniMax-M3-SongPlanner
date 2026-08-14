"""
M3 Song Planner Engine — native CLIP text generation via ComfyUI's CLIP stack.

Uses the same tokenize → generate → decode pattern as ComfyUI's native
TextGenerate node and H3-VisionPromptor's engine.py.

Supports model family detection (Qwen / Gemma-3 / Gemma-4) and per-family
chat template construction.

Author: M3 SongPlanner
"""

from __future__ import annotations

# Cache of loaded CLIP objects keyed by absolute checkpoint path.
CLIP_CACHE: dict[str, object] = {}

NONE_OPTION = "<none - use CLIP input>"


# --------------------------------------------------------------------------- #
# CLIP resolution
# --------------------------------------------------------------------------- #

def _clip_is_generatable(clip) -> bool:
    return (
        clip is not None
        and hasattr(clip, "generate")
        and hasattr(clip, "tokenize")
        and hasattr(clip, "decode")
    )


def _assert_generatable(clip, source_desc: str):
    if not _clip_is_generatable(clip):
        raise RuntimeError(
            f"[M3 SongPlanner] The CLIP from {source_desc} does not support native text generation "
            "(missing tokenize/generate/decode). Use a generative VLM text encoder "
            "(e.g. Qwen3-VL / Gemma-3/4-Vision instruct repacks — note Qwen2.5-VL repacks are "
            "vision-tower-only and not generatable) loaded via the native "
            "CLIPLoader node or from models/text_encoders/."
        )


def resolve_clip(clip_input, text_encoder_name: str, keep_model_loaded: bool = True):
    """Return (clip_obj, source_desc). Prefers clip_input; otherwise loads from text_encoders/."""
    if clip_input is not None:
        _assert_generatable(clip_input, "the connected CLIP input")
        return clip_input, "connected CLIP input"

    if text_encoder_name in {NONE_OPTION, "", "none", "None", None}:
        raise RuntimeError(
            "[M3 SongPlanner] No CLIP source: connect a CLIP output (from the native CLIPLoader "
            "node) or pick a model in the 'text_encoder' dropdown."
        )

    try:
        import folder_paths
        import comfy.sd
    except ImportError as e:
        raise RuntimeError(
            "[M3 SongPlanner] Could not import ComfyUI modules. This node must run inside ComfyUI."
        ) from e

    full_path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)
    if keep_model_loaded and full_path in CLIP_CACHE:
        return CLIP_CACHE[full_path], f"text_encoders/{text_encoder_name} (cached)"

    clip = comfy.sd.load_clip(
        ckpt_paths=[full_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    _assert_generatable(clip, f"text_encoders/{text_encoder_name}")
    if keep_model_loaded:
        CLIP_CACHE[full_path] = clip
    return clip, f"text_encoders/{text_encoder_name}"


# --------------------------------------------------------------------------- #
# Model-family detection and chat templating
# --------------------------------------------------------------------------- #

def detect_family(clip) -> str:
    """Sniff the CLIP's transformer/tokenizer class names. Returns qwen|gemma4|gemma|generic."""
    names: list[str] = []
    try:
        transformer = getattr(getattr(clip, "cond_stage_model", None), "transformer", None)
        if transformer is not None:
            names.append(type(transformer).__name__)
    except Exception:
        pass
    try:
        tokenizer = getattr(clip, "tokenizer", None)
        if tokenizer is not None:
            names.append(type(tokenizer).__name__)
            inner = getattr(tokenizer, "tokenizer", None)
            if inner is not None:
                names.append(type(inner).__name__)
    except Exception:
        pass
    haystack = " ".join(names).lower()
    if "qwen" in haystack:
        return "qwen"
    if "gemma4" in haystack:
        return "gemma4"
    if "gemma" in haystack:
        return "gemma"
    return "generic"


def build_chat_text(family: str, system: str, user: str) -> str:
    """Build the raw chat-formatted prompt text for the given model family."""
    if family == "qwen":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if family == "gemma4":
        return f"<|turn>user\n{system}\n\n{user}\n<|turn>model\n"
    if family == "gemma":
        return (
            f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    return f"{system}\n\n{user}"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _strip_echo(decoded: str, family: str) -> str:
    """Strip echoed prompt prefix from decoded text."""
    text = decoded
    for marker in ("<|im_start|>assistant\n", "<start_of_turn>model\n", "<|turn>model\n"):
        if marker in text:
            text = text.split(marker, 1)[1]
    for stop in ("<|im_end|>", "<end_of_turn>", "<|im_start|>", "<start_of_turn>", "<|turn>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate_text(
    clip,
    system: str,
    user: str,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 64,
    max_tokens: int = 2048,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> str:
    """Full chat-format generation using ComfyUI's native CLIP generation stack."""
    _assert_generatable(clip, "the resolved CLIP")
    family = detect_family(clip)
    prompt = build_chat_text(family, system, user)

    try:
        tokens = clip.tokenize(prompt, skip_template=True, min_length=1)
    except TypeError:
        try:
            tokens = clip.tokenize(prompt)
        except TypeError as e:
            raise RuntimeError(
                f"[M3 SongPlanner] clip.tokenize failed: {e}. Update ComfyUI."
            ) from e

    try:
        generated_ids = clip.generate(
            tokens,
            do_sample=temperature > 0.0,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=0.0,
            seed=seed,
        )
    except TypeError as e:
        raise RuntimeError(
            f"[M3 SongPlanner] clip.generate failed: {e}. Update ComfyUI."
        ) from e

    decoded = clip.decode(generated_ids)
    return _strip_echo(decoded, family)
