---
name: vintage
description: Process a video through vintage film/tape simulation. Use when the user wants to apply silent, golden, vhs, or cinematic effects to a video file.
disable-model-invocation: false
allowed-tools: Bash
argument-hint: <input-video> <mode> [--intensity N]
---

# Vintage Video Processor

Apply physics-based vintage video effects to a video file.

## Modes

- **silent** — 1910s orthochromatic film + carbon arc projection
- **golden** — 1950s Technicolor 3-strip IB dye transfer
- **vhs** — 1990s VHS tape (softness, chroma smear, tape saturation)
- **cinematic** — Modern neg→print photochemical pipeline (Vision3 500T → 2383)

## How to run

The processor lives at `/Users/mo/vintage-video/vintage_video.py`.

```bash
cd /Users/mo/vintage-video
python3 vintage_video.py <input> --mode <mode> [--intensity <0.0-2.0>] [-o <output>]
```

If no output path is given, it auto-generates one from the input filename and mode.

## Task

Process: $ARGUMENTS

Parse the arguments to extract:
1. **Input video path** — the file to process (resolve relative paths)
2. **Mode** — one of: silent, golden, vhs, cinematic (default: cinematic)
3. **Intensity** — effect strength 0.0-2.0 (default: 1.0)
4. **Output path** — optional, auto-generated if not given

Then run the processor. The command can take several minutes for long videos — run it and report the result when done.

If the user asks for a style but doesn't use the exact mode name, map it:
- "technicolor", "50s", "wizard of oz" → golden
- "tape", "90s", "retro" → vhs
- "film", "35mm", "kodak", "movie" → cinematic
- "old", "1910s", "black and white", "bw" → silent
