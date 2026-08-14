"""
M3 Song Planner Node — idea → Music 3 caption + lyrics via native VLM.

Two-stage pipeline using ComfyUI's native CLIP text-generation stack:
  Stage 1: Expand the user's idea into a structured music caption
  Stage 2: Generate original lyrics matching the caption

Outputs two STRING values that wire directly to MiniMaxMusic3TextEncode.

Author: M3 SongPlanner
"""

from __future__ import annotations

import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- ComfyUI V3 node API ----------------------------------------------------

import folder_paths
from comfy_api.latest import io

from . import engine

CATEGORY = "M3/SongPlanner"
NONE_OPTION = engine.NONE_OPTION
MAX_SEED = 0xFFFFFFFF

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


# --- Prompt loading (hot-reloaded from disk each run) ------------------------

def _load_prompt(filename: str, fallback: str) -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return fallback


_CAPTION_FALLBACK = (
    "You are a professional music producer. Write a MiniMax Music 3 structured caption "
    "with three sections: Global Metadata, Vocal Details, and Arrangement. "
    "Be concrete, 200-450 words."
)

_LYRICS_FALLBACK = (
    "You are a professional songwriter. Write original song lyrics with section tags "
    "([Verse], [Chorus], [Bridge], etc.). Keep lines rhythmic and singable."
)


# --- User message builders --------------------------------------------------

def _build_caption_user_msg(
    idea: str,
    genre_hint: str,
    vocal_config: str,
    language: str,
    duration_seconds: float,
) -> str:
    lines = [
        f"Creative idea: {idea}",
    ]
    if genre_hint:
        lines.append(f"Genre/mood hint: {genre_hint}")
    lines.append(f"Vocal configuration: {vocal_config}")
    lines.append(f"Lyric language: {language}")
    lines.append(f"Target duration: {duration_seconds:.0f} seconds")
    lines.append("")
    lines.append("Write the structured music caption now.")
    return "\n".join(lines)


def _build_lyrics_user_msg(
    idea: str,
    caption: str,
    language: str,
    duration_seconds: float,
) -> str:
    # Pick structure based on duration
    if duration_seconds <= 90:
        structure = "[Intro] → [Verse] → [Chorus] → [Outro]"
    elif duration_seconds <= 180:
        structure = "[Intro] → [Verse] → [Pre-Chorus] → [Chorus] → [Verse] → [Chorus] → [Bridge] → [Final Chorus] → [Outro]"
    else:
        structure = "[Intro] → [Verse] → [Pre-Chorus] → [Chorus] → [Verse] → [Chorus] → [Bridge] → [Solo] → [Final Chorus] → [Outro]"

    lines = [
        f"Creative idea: {idea}",
        f"Music description (for context): {caption[:800]}",
        f"Lyric language: {language}",
        f"Target duration: {duration_seconds:.0f} seconds",
        f"Suggested structure: {structure}",
        "",
        "Write the lyrics now.",
    ]
    return "\n".join(lines)


# --- V3 Node Definition -----------------------------------------------------

class M3SongPlanner(io.ComfyNode):
    """Turn a creative idea into Music 3 caption + lyrics using a LOCAL VLM.
    No API keys, no external services. Uses ComfyUI's native text-generation CLIP stack."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="M3SongPlanner",
            display_name="M3 Song Planner (Local VLM)",
            category=CATEGORY,
            search_aliases=["MiniMax", "Music", "M3", "song", "lyrics", "caption", "music3"],
            description=(
                "Turns a creative idea into a production-ready MiniMax Music 3 caption + lyrics "
                "pair using a LOCAL VLM (Qwen3-VL / Gemma-3-Vision). Two-stage pipeline: "
                "caption generation → lyrics generation. No API keys needed."
            ),
            inputs=[
                io.Clip.Input("clip", optional=True,
                              tooltip="Connect CLIP from CLIPLoader. Overrides the dropdown."),
                io.Combo.Input("text_encoder", options=_text_encoder_options(),
                               tooltip="VLM checkpoint from models/text_encoders/ "
                                       "(e.g. Qwen3-VL or Gemma-3-Vision instruct repack)."),
                io.String.Input("idea", multiline=True, default="",
                                placeholder="A dark K-pop song about walking through neon rain at midnight..."),
                io.String.Input("genre_hint", default="",
                                placeholder="e.g. synth-pop, lo-fi hip-hop, cinematic ballad"),
                io.Combo.Input("vocal_config", options=[
                    "female vocals", "male vocals", "duet",
                    "instrumental", "auto (decide from idea)",
                ], default="auto (decide from idea)"),
                io.Combo.Input("language", options=[
                    "English", "Chinese (Mandarin)", "Korean", "Japanese", "auto",
                ], default="English"),
                io.Float.Input("duration_seconds", default=120.0, min=30.0, max=300.0, step=10.0,
                               tooltip="Target song length. Affects structure and token budget."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,
                             control_after_generate="randomize"),
                io.Float.Input("temperature", default=0.8, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=64, min=0, max=500),
                io.Int.Input("max_tokens", default=2048, min=512, max=8192, step=64,
                             tooltip="Max tokens per generation (caption and lyrics each)."),
                io.Boolean.Input("keep_model_loaded", default=True, advanced=True,
                                 tooltip="Cache the text-encoder CLIP between runs."),
            ],
            outputs=[
                io.String.Output("caption",
                                 tooltip="Wire to MiniMaxMusic3TextEncode 'caption' input."),
                io.String.Output("lyrics",
                                 tooltip="Wire to MiniMaxMusic3TextEncode 'lyrics' input."),
                io.String.Output("debug",
                                 tooltip="JSON with timings, clip source, model family, warnings."),
            ],
        )

    @classmethod
    def execute(cls, text_encoder, idea, genre_hint, vocal_config, language,
                duration_seconds, seed, temperature, top_p, top_k, max_tokens,
                keep_model_loaded, clip=None):
        t_start = time.time()

        if not idea.strip():
            raise RuntimeError(
                "[M3 SongPlanner] The 'idea' input is empty. "
                "Type your creative idea (a mood, a concept, a story, anything) "
                "and try again."
            )

        # Resolve VLM
        clip_obj, source_desc = engine.resolve_clip(clip, text_encoder, keep_model_loaded)
        family = engine.detect_family(clip_obj)

        # Resolve auto options
        if vocal_config.startswith("auto"):
            vocal_config = ""  # let the VLM decide
        if language == "auto":
            language = "English"  # safe default

        # Load system prompts (hot-reloaded from disk)
        caption_sys = _load_prompt("caption_system.txt", _CAPTION_FALLBACK)
        lyrics_sys = _load_prompt("lyrics_system.txt", _LYRICS_FALLBACK)

        warnings = []

        # --- Stage 1: Caption generation --------------------------------
        t_cap = time.time()
        caption_user = _build_caption_user_msg(
            idea, genre_hint, vocal_config, language, duration_seconds,
        )
        try:
            caption = engine.generate_text(
                clip_obj,
                system=caption_sys,
                user=caption_user,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        except Exception as e:
            caption = ""
            warnings.append(f"caption generation failed: {e}")
        caption_seconds = time.time() - t_cap

        # --- Stage 2: Lyrics generation ---------------------------------
        t_lyr = time.time()
        lyrics_user = _build_lyrics_user_msg(
            idea, caption, language, duration_seconds,
        )
        try:
            lyrics = engine.generate_text(
                clip_obj,
                system=lyrics_sys,
                user=lyrics_user,
                seed=seed + 1,  # different seed for variety
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        except Exception as e:
            lyrics = ""
            warnings.append(f"lyrics generation failed: {e}")
        lyrics_seconds = time.time() - t_lyr

        # --- Debug output -----------------------------------------------
        debug = json.dumps({
            "clip_source": source_desc,
            "model_family": family,
            "vocal_config": vocal_config or "auto",
            "language": language,
            "duration_seconds": duration_seconds,
            "seed": seed,
            "caption_seconds": round(caption_seconds, 2),
            "lyrics_seconds": round(lyrics_seconds, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "caption_length": len(caption),
            "lyrics_length": len(lyrics),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False)

        return io.NodeOutput(caption, lyrics, debug)


def _text_encoder_options() -> list[str]:
    try:
        return [NONE_OPTION] + folder_paths.get_filename_list("text_encoders")
    except Exception:
        return [NONE_OPTION]
