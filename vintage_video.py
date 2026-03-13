#!/usr/bin/env python3
"""
Vintage Video Processor v3.1 — Complete Rewrite
Transforms modern video into authentic period footage.

Modes:
  silent    — 1910s orthochromatic film + carbon arc projection
  golden    — 1950s Technicolor 3-strip (Baselight crosstalk-removal approach)
  vhs       — 1990s VHS tape (softness + chroma smear, NOT scanlines)
  cinematic — Modern neg→print photochemical pipeline (Vision3 500T → 2383)

Usage:
  python vintage_video.py input.mp4 --mode cinematic -o output.mp4
  python vintage_video.py input.mp4 --mode vhs --intensity 0.8
  python vintage_video.py input.mp4 --mode silent
"""

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from pathlib import Path
import subprocess
import argparse
import sys
import os
import time

# ─── Try loading film science (for cinematic mode) ──────────────────────────

HAS_FILM_SCIENCE = False
try:
    import spectral_film_lut as sfl
    from spectral_film_lut.film_spectral import FilmSpectral
    HAS_FILM_SCIENCE = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

class BrownianWalk:
    """Random walk with mean-reversion spring (bounded drift)."""

    def __init__(self, sigma=1.0, spring=0.02, initial=0.0, bounds=None):
        self.sigma = sigma
        self.spring = spring
        self.pos = initial
        self.bounds = bounds

    def step(self):
        self.pos += np.random.normal(0, self.sigma)
        self.pos -= self.pos * self.spring
        if self.bounds is not None:
            self.pos = np.clip(self.pos, self.bounds[0], self.bounds[1])
        return self.pos


def hd_curve(x, gamma=1.0, dmin=0.05, dmax=0.95, toe_width=0.15, shoulder_width=0.15):
    """Hurter-Driffield characteristic curve."""
    x_norm = (x - 0.5) * 2.0
    k = gamma * 4.0
    sig = 1.0 / (1.0 + np.exp(-k * x_norm))
    density = dmin + (dmax - dmin) * sig
    return density


def build_hd_lut(gamma=1.0, dmin=0.05, dmax=0.95, toe_width=0.15, shoulder_width=0.15, size=4096):
    """Precompute H&D curve as lookup table."""
    x = np.linspace(0, 1, size, dtype=np.float32)
    lut = hd_curve(x, gamma, dmin, dmax, toe_width, shoulder_width).astype(np.float32)
    return lut


def apply_lut(image, lut):
    """Apply a float32 LUT to a float32 image (0-1 range)."""
    indices = np.clip(image * (len(lut) - 1), 0, len(lut) - 1).astype(np.int32)
    return lut[indices]


def physical_halation(frame, threshold=0.7, radius_frac=0.05, color=(0.08, 0.04, 0.01)):
    """Physical halation: light scatters through emulsion, bounces off film base.
    Primarily red-weighted (film layers are B-G-R from lens to base)."""
    h, w = frame.shape[:2]
    gray = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]
    bright = np.clip((gray - threshold) / (1.0 - threshold), 0, 1).astype(np.float32)
    radius = max(3, int(min(h, w) * radius_frac))
    if radius % 2 == 0:
        radius += 1
    radius = min(radius, 301)
    halation = cv2.GaussianBlur(bright, (radius, radius), radius / 3.0)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + halation * color[0], 0, 1)
    frame[:, :, 1] = np.clip(frame[:, :, 1] + halation * color[1], 0, 1)
    frame[:, :, 0] = np.clip(frame[:, :, 0] + halation * color[2], 0, 1)
    return frame


class TemporalGrain:
    """AR(1) temporally coherent grain generator.
    grain[t] = alpha * grain[t-1] + sqrt(1-alpha^2) * noise[t]
    Set alpha=0 for fully independent per-frame grain (cinematic mode)."""

    def __init__(self, height, width, channels=3, alpha=0.5, gen_scale=2):
        self.alpha = alpha
        self.beta = np.sqrt(1.0 - alpha ** 2) if alpha > 0 else 1.0
        self.channels = channels
        self.gen_scale = gen_scale
        self.h = height
        self.w = width
        self.gh = max(1, height // gen_scale)
        self.gw = max(1, width // gen_scale)
        self.prev = [np.random.normal(0, 1, (self.gh, self.gw)).astype(np.float32)
                     for _ in range(channels)]

    def generate(self):
        """Generate next frame's grain with temporal coherence."""
        grains = []
        for c in range(self.channels):
            noise = np.random.normal(0, 1, (self.gh, self.gw)).astype(np.float32)
            if self.alpha > 0:
                g = self.alpha * self.prev[c] + self.beta * noise
                self.prev[c] = g
            else:
                g = noise  # fully independent per frame
            # Upscale + slight blur for clumpy grain structure
            if self.gen_scale > 1:
                g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
                g = cv2.GaussianBlur(g, (3, 3), 0.6)
            grains.append(g)
        return grains

    def resize(self, height, width):
        self.h = height
        self.w = width
        self.gh = max(1, height // self.gen_scale)
        self.gw = max(1, width // self.gen_scale)
        self.prev = [np.random.normal(0, 1, (self.gh, self.gw)).astype(np.float32)
                     for _ in range(self.channels)]


def soft_light_blend(base, grain, strength):
    """Soft Light blend mode — grain is most visible in midtones, fades in
    pure black/white. Much more natural than additive blending."""
    # Pegtop's soft light formula: (1-2*grain)*base^2 + 2*grain*base
    # grain is centered at 0, remap to 0.5-centered
    g = np.clip(grain * strength + 0.5, 0, 1)
    result = np.where(
        g <= 0.5,
        base - (1.0 - 2.0 * g) * base * (1.0 - base),
        base + (2.0 * g - 1.0) * (np.sqrt(base) - base)
    )
    return result.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SILENT MODE — 1910s Orthochromatic Film + Carbon Arc Projection
# ═════════════════════════════════════════════════════════════════════════════

class SilentProcessor:
    """1910s film: orthochromatic emulsion, carbon arc flicker, hand-crank,
    nitrate grain (Soft Light blend), gate weave, scratches, iris vignette."""

    def __init__(self, width, height, fps, intensity=1.0):
        self.w = width
        self.h = height
        self.fps = fps
        self.intensity = intensity

        # Brownian gate weave — increased amplitude (weave > scratches for the look)
        self.weave_x = BrownianWalk(sigma=1.5 * intensity, spring=0.012)
        self.weave_y = BrownianWalk(sigma=1.0 * intensity, spring=0.012)

        # H&D curve — nitrate: high contrast, limited latitude
        self.hd_lut = build_hd_lut(gamma=2.0, dmin=0.06, dmax=0.92)

        # Carbon arc flicker — semi-periodic (AC power + arc gap variation)
        self.flicker_t = 0.0
        self.flicker_f1 = 2.3 + np.random.uniform(-0.3, 0.3)
        self.flicker_f2 = 4.7 + np.random.uniform(-0.5, 0.5)
        self.flicker_f3 = 0.8 + np.random.uniform(-0.2, 0.2)
        self.flicker_phi1 = np.random.uniform(0, 2 * np.pi)
        self.flicker_phi2 = np.random.uniform(0, 2 * np.pi)
        self.flicker_phi3 = np.random.uniform(0, 2 * np.pi)

        # Hand-crank speed Brownian walk
        self.crank_speed = BrownianWalk(sigma=0.3, spring=0.05, initial=0.0,
                                         bounds=(-4, 6))

        # Persistent scratches
        self.scratches = []

        # Grain: Soft Light blend, half-res for clumpiness, low temporal coherence
        self.grain = TemporalGrain(height, width, channels=1, alpha=0.2, gen_scale=2)

        # Iris vignette
        self._build_iris_mask()

    def _build_iris_mask(self):
        """Iris vignette with Fourier perturbation (not a perfect circle)."""
        h, w = self.h, self.w
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        dx = (x - cx) / (w * 0.52)
        dy = (y - cy) / (h * 0.52)
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        np.random.seed(42)
        perturbation = np.zeros_like(theta)
        for k in range(2, 7):
            a_k = np.random.uniform(0.01, 0.04)
            phi_k = np.random.uniform(0, 2 * np.pi)
            perturbation += a_k * np.sin(k * theta + phi_k)

        r_adj = r - perturbation
        self.iris_mask = np.clip(1.0 - np.clip((r_adj - 0.85) / 0.15, 0, 1), 0, 1).astype(np.float32)
        self.iris_mask = cv2.GaussianBlur(self.iris_mask, (15, 15), 5)

    def process_frame(self, frame, frame_idx):
        h, w = frame.shape[:2]
        fimg = frame.astype(np.float32) / 255.0

        # 1. ORTHOCHROMATIC CONVERSION (red-blind)
        r, g, b = fimg[:, :, 2], fimg[:, :, 1], fimg[:, :, 0]
        gray = 0.05 * r + 0.45 * g + 0.50 * b

        # 2. H&D CHARACTERISTIC CURVE
        gray = apply_lut(gray, self.hd_lut)

        # 3. CARBON ARC FLICKER
        dt = 1.0 / max(self.fps, 1)
        self.flicker_t += dt
        t = self.flicker_t
        flicker = 1.0 + self.intensity * (
            0.04 * np.sin(2 * np.pi * self.flicker_f1 * t + self.flicker_phi1) +
            0.025 * np.sin(2 * np.pi * self.flicker_f2 * t + self.flicker_phi2) +
            0.015 * np.sin(2 * np.pi * self.flicker_f3 * t + self.flicker_phi3) +
            0.01 * np.random.normal()
        )
        gray = gray * flicker

        # 4. SEPIA TINTING
        result = np.empty((h, w, 3), dtype=np.float32)
        result[:, :, 2] = gray * 1.0
        result[:, :, 1] = gray * 0.85
        result[:, :, 0] = gray * 0.65

        # 5. GRAIN — Soft Light blend (visible in midtones, fades at extremes)
        grain_fields = self.grain.generate()
        grain = grain_fields[0]
        grain_strength = 0.12 * self.intensity
        # Apply per-channel with sepia weighting
        for c, weight in enumerate([0.65, 0.85, 1.0]):
            result[:, :, c] = soft_light_blend(
                np.clip(result[:, :, c], 0, 1),
                grain,
                grain_strength * weight
            )

        # 6. FILM DEGRADATION — scratches + dust
        if np.random.random() < 0.25:
            sx_init = np.random.randint(int(w * 0.1), int(w * 0.9))
            self.scratches.append({
                'walk': BrownianWalk(sigma=0.3, spring=0.01, initial=float(sx_init)),
                'thickness': np.random.randint(1, 3),
                'brightness': np.random.uniform(-0.12, 0.10),
                'life': np.random.randint(8, 80),
            })

        alive = []
        for s in self.scratches:
            sx = int(s['walk'].step())
            s['life'] -= 1
            if s['life'] > 0 and 0 <= sx < w - s['thickness']:
                result[:, sx:sx + s['thickness'], :] += s['brightness']
                alive.append(s)
        self.scratches = alive

        # Dust spots
        n_dust = np.random.randint(0, 6)
        for _ in range(n_dust):
            dx, dy = np.random.randint(0, w), np.random.randint(0, h)
            sz = np.random.randint(2, 5)
            val = np.random.uniform(-0.15, 0.05)
            cv2.circle(result, (dx, dy), sz, (val, val, val), -1)

        # Rare splice marks
        if np.random.random() < 0.003:
            sy = np.random.randint(0, h - 5)
            result[sy:sy + 4, :, :] += 0.3

        # 7. IRIS VIGNETTE
        if self.iris_mask.shape[:2] == (h, w):
            for c in range(3):
                result[:, :, c] *= self.iris_mask

        # 8. BROWNIAN GATE WEAVE
        dx_shift = self.weave_x.step()
        dy_shift = self.weave_y.step()
        M = np.float32([[1, 0, dx_shift], [0, 1, dy_shift]])
        result = np.clip(result, 0, 1)
        result_u8 = (result * 255).astype(np.uint8)
        result_u8 = cv2.warpAffine(result_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return result_u8

    def get_frame_timing(self, frame_idx):
        speed_offset = self.crank_speed.step()
        effective_fps = 16.0 + speed_offset
        return effective_fps


# ═════════════════════════════════════════════════════════════════════════════
# GOLDEN MODE — 1950s Technicolor (Baselight Crosstalk-Removal Approach)
# ═════════════════════════════════════════════════════════════════════════════

class GoldenProcessor:
    """Technicolor look via the proven Baselight crosstalk-removal approach:
    1. sRGB → linear
    2. Crosstalk-removal color matrix (removes green contamination → vivid colors)
    3. Steep S-curve contrast (lifted shadows, smooth highlight rolloff)
    4. Warmth shift
    5. Per-hue saturation boost (reds/magentas hard, blues restrained)
    6. Highlight protection + warm bloom
    7. Per-channel grain (3 separate negatives) + registration artifacts
    """

    def __init__(self, width, height, fps, intensity=1.0):
        self.w = width
        self.h = height
        self.fps = fps
        self.intensity = intensity

        # Process at half resolution for speed (film is inherently soft)
        self.proc_h = max(1, height // 2)
        self.proc_w = max(1, width // 2)

        # Crosstalk-removal color matrix — blended with identity for control.
        # Full Baselight matrix is too aggressive; we use ~35% strength.
        full_matrix = np.array([
            [ 2.658, -1.997,  0.342],
            [-0.134,  0.853,  0.286],
            [-0.062, -1.981,  3.047],
        ], dtype=np.float32)
        identity = np.eye(3, dtype=np.float32)
        blend = 0.50 * intensity  # scale with intensity — more vivid colors
        self.crosstalk_matrix = identity * (1.0 - blend) + full_matrix * blend

        # Registration: 3 independent Brownian walks for dye layer misalignment
        self.reg_walks = [
            (BrownianWalk(sigma=0.4 * intensity, spring=0.03),
             BrownianWalk(sigma=0.4 * intensity, spring=0.03))
            for _ in range(3)
        ]

        # Gate weave
        self.weave_x = BrownianWalk(sigma=0.4 * intensity, spring=0.02)
        self.weave_y = BrownianWalk(sigma=0.3 * intensity, spring=0.02)

        # Independent grain per separation negative (3 strips = 3 channels)
        self.grain = TemporalGrain(self.proc_h, self.proc_w, channels=3, alpha=0.4, gen_scale=3)

        # Pre-build S-curve LUT
        self._build_scurve_lut()

    def _build_scurve_lut(self):
        """Steep S-curve: punchy midtones, smooth highlight rolloff, lifted shadows."""
        x = np.linspace(0, 1, 4096, dtype=np.float32)
        # Lifted shadows (NOT crushed blacks) + smooth highlight rolloff
        shadow_lift = 0.03
        # Parametric S-curve with controllable midtone contrast
        midpoint = 0.42  # slightly below center for Technicolor warmth
        contrast = 1.6   # steep midtone punch
        # Smooth sigmoid
        x_centered = (x - midpoint) * contrast
        curve = 1.0 / (1.0 + np.exp(-x_centered * 4.0))
        # Normalize to full range with shadow lift
        curve = shadow_lift + (1.0 - shadow_lift) * (curve - curve[0]) / (curve[-1] - curve[0])
        self.scurve_lut = curve.astype(np.float32)

    def _per_hue_saturation(self, rgb):
        """Boost saturation selectively by hue — reds/magentas hard, blues restrained."""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        lum = lum[:, :, np.newaxis]

        # Compute hue via simple max-channel method
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c + 1e-7

        # Hue angle approximation (0-6 scale)
        hue = np.zeros_like(r)
        mask_r = (max_c == r)
        mask_g = (max_c == g) & ~mask_r
        mask_b = ~mask_r & ~mask_g
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / chroma[mask_r]) % 6.0
        hue[mask_g] = (b[mask_g] - r[mask_g]) / chroma[mask_g] + 2.0
        hue[mask_b] = (r[mask_b] - g[mask_b]) / chroma[mask_b] + 4.0

        # Saturation multiplier by hue region (subtle — matrix does heavy lifting)
        sat_mult = np.ones_like(r)
        # Reds (hue ~0 or ~6): modest boost
        red_mask = (hue < 0.8) | (hue > 5.2)
        sat_mult[red_mask] = 1.15
        # Oranges/yellows (0.8-2.0): slight
        oy_mask = (hue >= 0.8) & (hue < 2.0)
        sat_mult[oy_mask] = 1.10
        # Greens/cyans (2.0-4.0): slight
        gc_mask = (hue >= 2.0) & (hue < 4.0)
        sat_mult[gc_mask] = 1.10
        # Blues (4.0-5.2): barely touched
        blue_mask = (hue >= 4.0) & (hue < 5.2)
        sat_mult[blue_mask] = 1.03

        # Protect skin tones
        skin_mask = (hue >= 0.5) & (hue < 1.5) & (chroma / (max_c + 1e-7) < 0.5)
        sat_mult[skin_mask] = np.minimum(sat_mult[skin_mask], 1.05)

        sat_mult = sat_mult[:, :, np.newaxis]
        # Blend: base + sat_mult means we scale deviation from luma
        result = lum + (rgb - lum) * sat_mult
        return np.clip(result, 0, 1).astype(np.float32)

    def process_frame(self, frame, frame_idx):
        orig_h, orig_w = frame.shape[:2]
        # Downscale to half resolution for speed
        small = cv2.resize(frame, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
        h, w = self.proc_h, self.proc_w
        fimg = small.astype(np.float32) / 255.0

        # 0. FILM SOFTNESS (1950s lenses were not sharp like modern glass)
        fimg = cv2.GaussianBlur(fimg, (3, 3), 0.8)

        # 1. sRGB TO LINEAR
        linear = np.where(fimg <= 0.04045,
                          fimg / 12.92,
                          ((fimg + 0.055) / 1.055) ** 2.4).astype(np.float32)

        # BGR → RGB
        rgb = linear[:, :, ::-1].copy()

        # 2. CROSSTALK-REMOVAL COLOR MATRIX
        pixels = rgb.reshape(-1, 3)
        transformed = pixels @ self.crosstalk_matrix.T
        # Clamp negatives, normalize per-pixel to preserve luminance
        transformed = np.maximum(transformed, 0)
        # Soft normalize: if any channel > 1, scale that pixel down
        max_vals = np.maximum(transformed.max(axis=1, keepdims=True), 1.0)
        transformed = transformed / max_vals
        rgb = transformed.reshape(h, w, 3).astype(np.float32)

        # 2b. DIFFERENTIAL CHANNEL SHARPNESS (green sharpest, red softest)
        # Real 3-strip: green via direct beam splitter, red through blue bipack
        rgb[:, :, 0] = cv2.GaussianBlur(rgb[:, :, 0], (0, 0), 0.8)  # Red softest
        rgb[:, :, 2] = cv2.GaussianBlur(rgb[:, :, 2], (0, 0), 0.5)  # Blue medium

        # 3. STEEP S-CURVE CONTRAST
        rgb = apply_lut(np.clip(rgb, 0, 1), self.scurve_lut)

        # 4. WARMTH SHIFT (shadow-neutral — real IB prints have deep neutral blacks)
        lum_warmth = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        warmth_mask = np.clip(lum_warmth * 2.5, 0, 1)  # 0 in shadows, 1 in mids/highs
        rgb[:, :, 0] *= 1.0 + 0.06 * warmth_mask  # R up in mids/highs only
        rgb[:, :, 2] *= 1.0 - 0.08 * warmth_mask  # B down in mids/highs only
        rgb = np.clip(rgb, 0, 1)

        # 5. PER-HUE SATURATION BOOST
        rgb = self._per_hue_saturation(rgb)

        # 5b. HELMHOLTZ-KOHLRAUSCH LUMINANCE BOOST (saturated dyes appear self-luminous)
        lum_hk = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, np.newaxis]
        chroma_hk = np.sqrt(np.sum((rgb - lum_hk) ** 2, axis=2))
        hk_boost = chroma_hk * 0.08 * self.intensity
        rgb = np.clip(rgb + hk_boost[:, :, np.newaxis], 0, 1).astype(np.float32)

        # 5c. DYE TRANSFER BLEED (chemical softness at saturated color boundaries)
        lum_g4 = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        chroma_g4 = np.sqrt(np.sum((rgb - lum_g4[:, :, np.newaxis]) ** 2, axis=2))
        chroma_edges = np.abs(cv2.Sobel(chroma_g4, cv2.CV_32F, 1, 0)) + \
                       np.abs(cv2.Sobel(chroma_g4, cv2.CV_32F, 0, 1))
        bleed_mask = np.clip(chroma_edges * 3.0, 0, 1)[:, :, np.newaxis]
        rgb_blurred = cv2.GaussianBlur(rgb, (5, 5), 1.5)
        rgb = (rgb * (1.0 - bleed_mask * 0.4) + rgb_blurred * (bleed_mask * 0.4)).astype(np.float32)

        # 6. HIGHLIGHT PROTECTION — keep highlights luminous white, no color cast
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        highlight_mask = np.clip((lum - 0.7) / 0.3, 0, 1)[:, :, np.newaxis]
        white = lum[:, :, np.newaxis] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rgb = rgb * (1.0 - highlight_mask * 0.5) + white * (highlight_mask * 0.5)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)

        # 7. REGISTRATION ARTIFACTS (sub-pixel dye layer misalignment)
        for c in range(3):
            dx_reg = self.reg_walks[c][0].step()
            dy_reg = self.reg_walks[c][1].step()
            if abs(dx_reg) > 0.05 or abs(dy_reg) > 0.05:
                M = np.float32([[1, 0, dx_reg], [0, 1, dy_reg]])
                rgb[:, :, c] = cv2.warpAffine(rgb[:, :, c], M, (w, h),
                                               borderMode=cv2.BORDER_REFLECT)

        # RGB → BGR for OpenCV operations
        result = rgb[:, :, ::-1].copy()

        # 8. WARM BLOOM ON HIGHLIGHTS
        gray = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
        bloom_mask = np.clip((gray - 0.55) / 0.45, 0, 1).astype(np.float32)
        bloom = result.copy()
        for c in range(3):
            bloom[:, :, c] *= bloom_mask
        br = min(max(h, w) // 25, 151)
        if br % 2 == 0:
            br += 1
        bloom = cv2.GaussianBlur(bloom, (br, br), br / 3.0)
        # Warm bloom: more red, less blue — stronger for visible glow
        result[:, :, 2] = np.clip(result[:, :, 2] + bloom[:, :, 2] * 0.15, 0, 1)
        result[:, :, 1] = np.clip(result[:, :, 1] + bloom[:, :, 1] * 0.10, 0, 1)
        result[:, :, 0] = np.clip(result[:, :, 0] + bloom[:, :, 0] * 0.04, 0, 1)

        # 8b. HALATION (warm glow from light scattering through emulsion)
        result = physical_halation(result, threshold=0.65, radius_frac=0.05,
                                    color=(0.12 * self.intensity,
                                           0.06 * self.intensity,
                                           0.02 * self.intensity))

        # 9. INDEPENDENT GRAIN PER CHANNEL (3-strip = 3 negatives)
        # IB dye transfer suppresses grain; green finest, red coarsest
        grains = self.grain.generate()
        grains[1] = cv2.GaussianBlur(grains[1], (3, 3), 0.8)  # green finest
        grains[0] = cv2.GaussianBlur(grains[0], (3, 3), 0.5)  # blue medium
        # red (grains[2]) unblurred — coarsest
        gray = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
        grain_resp = np.clip(np.sqrt(4.0 * gray * (1.0 - gray)), 0.2, 1.0)
        grain_intensity = 0.05 * self.intensity
        for c in range(3):
            result[:, :, c] = soft_light_blend(
                np.clip(result[:, :, c], 0, 1),
                grains[c],
                grain_intensity * grain_resp
            )

        # 10. LINEAR TO sRGB
        result = np.clip(result, 0, 1)
        result = np.where(result <= 0.0031308,
                          result * 12.92,
                          1.055 * np.power(result, 1.0 / 2.4) - 0.055)

        # 11. VIGNETTE (1950s lens — moderate)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt(((x - w / 2) / (w * 0.5)) ** 2 + ((y - h / 2) / (h * 0.5)) ** 2)
        vig = 1.0 - np.clip((dist - 0.6) / 0.9, 0, 1) ** 2 * 0.35
        for c in range(3):
            result[:, :, c] *= vig

        # 12. GATE WEAVE
        dx_shift = self.weave_x.step()
        dy_shift = self.weave_y.step()
        M = np.float32([[1, 0, dx_shift], [0, 1, dy_shift]])
        result_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        result_u8 = cv2.warpAffine(result_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Upscale back to original resolution
        result_u8 = cv2.resize(result_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return result_u8


# ═════════════════════════════════════════════════════════════════════════════
# VHS MODE — 1990s VHS Tape (Softness + Chroma Smear, NOT Scanlines)
# ═════════════════════════════════════════════════════════════════════════════

class VHSProcessor:
    """Real VHS look: SOFTNESS + chroma smear + warm colors.
    No scanlines (that's CRT), no complex QAM math (causes artifacts).

    Pipeline: direct YIQ bandwidth limiting — simple, fast, correct.
    1. Downscale to 480p
    2. RGB → YIQ
    3. Luma lowpass (~2.5 MHz equivalent) → softness + edge ringing
    4. Chroma extreme lowpass (~500 kHz) → horizontal color smear
    5. Chroma delay, warm shift, slight desat
    6. Subtle noise, timebase jitter, head switching
    7. Upscale back
    """

    def __init__(self, width, height, fps, intensity=1.0):
        self.w = width
        self.h = height
        self.fps = fps
        self.intensity = intensity
        self.vhs_h = 480
        self.vhs_w = 720
        self.frame_count = 0

        # Sample rate for filter design: 720 pixels across ~52.6 μs → ~13.5 MHz
        nyquist_mhz = 13.5 / 2.0  # ~6.75 MHz

        # VHS luma bandwidth: ~2.5 MHz (the main softness)
        vhs_y_cutoff = min(0.85, 2.5 / nyquist_mhz)
        self.vhs_y_b, self.vhs_y_a = sp_signal.butter(3, vhs_y_cutoff, btype='low')

        # VHS chroma bandwidth: ~500 kHz (extreme horizontal smear)
        vhs_c_cutoff = max(0.02, min(0.85, 0.5 / nyquist_mhz))
        self.vhs_c_b, self.vhs_c_a = sp_signal.butter(2, vhs_c_cutoff, btype='low')

        # Edge ringing/sharpening kernel (VCR sharpening circuits)
        # VHS is soft but VCRs add edge enhancement — bright/dark halos on edges
        self.sharpen_kernel = np.array([[-0.3, -0.5, -0.3],
                                         [-0.5,  5.2, -0.5],
                                         [-0.3, -0.5, -0.3]], dtype=np.float32)

        # Head-switch position
        self.head_switch_y = BrownianWalk(sigma=0.2, spring=0.1,
                                           initial=float(self.vhs_h - 6),
                                           bounds=(self.vhs_h - 12, self.vhs_h - 2))

        # Timebase jitter
        self.prev_jitter = np.zeros(self.vhs_h, dtype=np.float32)

        # Hue instability — per-frame color phase wobble ("Never The Same Color")
        self.hue_walk = BrownianWalk(sigma=0.008, spring=0.05, initial=0.0,
                                      bounds=(-0.04, 0.04))

        # Tape saturation LUT — nonlinear soft-knee compression (tanh-based)
        x = np.linspace(0, 1, 4096, dtype=np.float32)
        self.tape_sat_lut = (np.tanh((x - 0.5) * 3.0) * 0.375 + 0.445).astype(np.float32)

        # Store previous frame for interlacing
        self.prev_frame = None

        # Camcorder shake — disabled (other artifacts carry the VHS feel)
        # self.shake_x = BrownianWalk(sigma=1.2 * intensity, spring=0.015)
        # self.shake_y = BrownianWalk(sigma=0.8 * intensity, spring=0.015)

    def _rgb_to_yiq(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        return y.astype(np.float32), i.astype(np.float32), q.astype(np.float32)

    def _yiq_to_rgb(self, y, i, q):
        r = y + 0.956 * i + 0.621 * q
        g = y - 0.272 * i - 0.647 * q
        b = y - 1.106 * i + 1.703 * q
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 1).astype(np.float32)

    def process_frame(self, frame, frame_idx):
        h, w = frame.shape[:2]
        self.frame_count += 1

        # Downscale to NTSC 480p
        small = cv2.resize(frame, (self.vhs_w, self.vhs_h), interpolation=cv2.INTER_AREA)
        fimg = small.astype(np.float32) / 255.0

        # 0. TAPE SATURATION (nonlinear soft-knee compression — the biggest VHS signature)
        # Real tape has milky blacks, soft-knee highlight compression, not linear
        fimg = apply_lut(fimg, self.tape_sat_lut)

        # 0b. SPATIAL SOFTNESS (head wear, tape contact, write-read gap)
        fimg = cv2.GaussianBlur(fimg, (3, 3), 0.6)

        rgb = fimg[:, :, ::-1].copy()

        # 1. RGB → YIQ
        y_chan, i_chan, q_chan = self._rgb_to_yiq(rgb)

        # 2. LUMA BANDWIDTH LIMITING — the core VHS softness
        # IIR filter gives authentic edge ringing (overshoot at transitions)
        y_filtered = sp_signal.lfilter(self.vhs_y_b, self.vhs_y_a,
                                        y_chan, axis=1).astype(np.float32)

        # 3. CHROMA BANDWIDTH LIMITING — extreme horizontal color smear
        i_filtered = sp_signal.lfilter(self.vhs_c_b, self.vhs_c_a,
                                        i_chan, axis=1).astype(np.float32)
        q_filtered = sp_signal.lfilter(self.vhs_c_b, self.vhs_c_a,
                                        q_chan, axis=1).astype(np.float32)

        # 3b. VERTICAL CHROMA BLUR (VHS degrades chroma in both directions)
        i_filtered = gaussian_filter1d(i_filtered, sigma=0.8, axis=0)
        q_filtered = gaussian_filter1d(q_filtered, sigma=0.8, axis=0)

        # 4. CHROMA DELAY (VHS misregistration — 1-2 pixels right)
        shift = 1 + np.random.randint(0, 2)
        i_filtered = np.roll(i_filtered, shift, axis=1)
        q_filtered = np.roll(q_filtered, shift, axis=1)

        # 5. NOISE — horizontally correlated (VHS noise is streaky along scan lines)
        y_noise = np.random.normal(0, 0.025 * self.intensity,
                                    y_filtered.shape).astype(np.float32)
        y_noise = gaussian_filter1d(y_noise, sigma=2.5, axis=1)
        y_filtered += y_noise
        # Context-dependent chroma noise (worse in saturated reds — color-under 629 kHz)
        i_noise = np.random.normal(0, 0.012 * self.intensity,
                                    i_filtered.shape).astype(np.float32)
        q_noise = np.random.normal(0, 0.012 * self.intensity,
                                    q_filtered.shape).astype(np.float32)
        i_noise = gaussian_filter1d(i_noise, sigma=2.5, axis=1)
        q_noise = gaussian_filter1d(q_noise, sigma=2.5, axis=1)
        sat_level = np.sqrt(i_filtered ** 2 + q_filtered ** 2)
        red_weight = 1.0 + np.clip(i_filtered, 0, None) * 1.5
        chroma_scale = (0.5 + sat_level * 2.0) * red_weight
        i_filtered += i_noise * chroma_scale
        q_filtered += q_noise * chroma_scale

        # 5b. LUMA TRAILING — bright objects leave rightward trail (FM recording artifact)
        trail = np.zeros_like(y_filtered)
        trail[:, 1:] = y_filtered[:, :-1]
        y_filtered = y_filtered * 0.96 + trail * 0.04

        # 5c. HUE INSTABILITY — per-frame color phase wobble ("Never The Same Color")
        hue_shift = self.hue_walk.step()
        cos_h, sin_h = np.cos(hue_shift), np.sin(hue_shift)
        i_rot = i_filtered * cos_h - q_filtered * sin_h
        q_rot = i_filtered * sin_h + q_filtered * cos_h
        i_filtered, q_filtered = i_rot, q_rot

        # 6. YIQ → RGB
        result = self._yiq_to_rgb(y_filtered, i_filtered, q_filtered)
        result = result[:, :, ::-1].copy()  # RGB → BGR

        # 6b. EDGE RINGING / SHARPENING (VCR sharpening circuits)
        # VHS is soft but VCRs add edge enhancement — creates bright/dark halos
        luma = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
        sharpened = cv2.filter2D(luma, -1, self.sharpen_kernel)
        edge_boost = (sharpened - luma) * 0.3 * self.intensity
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] + edge_boost, 0, 1)

        # 6c. BRIGHTNESS GAIN + HIGHLIGHT COMPRESSION
        # VHS playback has slight brightness lift; whites bloom/smear (can't hold bright whites)
        result = result * 1.08
        # FM HIGHLIGHT BLOOM (bright areas spread horizontally — FM can't hold peaks)
        bright_mask = np.clip((result - 0.85) / 0.15, 0, 1)
        bloom_h = result * bright_mask
        for c in range(3):
            bloom_h[:, :, c] = gaussian_filter1d(bloom_h[:, :, c], sigma=8, axis=1)
        result = result * (1.0 - bright_mask * 0.5) + bloom_h * 0.5
        result = np.clip(result, 0, 0.92)

        # 7. COLOR DEGRADATION — warm shift, slight desaturation
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.05, 0, 1)  # Red push
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.95, 0, 1)  # Blue loss
        gray = np.mean(result, axis=2, keepdims=True)
        result = result * 0.85 + gray * 0.15  # more desaturation — VHS washes out colors

        # 8. TIMEBASE JITTER (per-scanline horizontal instability)
        jitter = 0.6 * self.prev_jitter + 0.4 * np.random.normal(
            0, 0.4 * self.intensity, self.vhs_h).astype(np.float32)
        self.prev_jitter = jitter
        significant = np.abs(jitter) > 0.3
        for row in np.where(significant)[0]:
            M_row = np.float32([[1, 0, jitter[row]], [0, 1, 0]])
            result[row:row + 1] = cv2.warpAffine(result[row:row + 1], M_row,
                                                   (self.vhs_w, 1),
                                                   borderMode=cv2.BORDER_REFLECT)

        # 9. HEAD-SWITCHING NOISE (bottom ~5 scanlines, most frames)
        hs_y = int(self.head_switch_y.step())
        hs_y = max(0, min(hs_y, self.vhs_h - 4))
        if np.random.random() < 0.7:
            hs_height = np.random.randint(2, 5)
            hs_shift = np.random.uniform(3, 10) * self.intensity
            end_y = min(hs_y + hs_height, self.vhs_h)
            result[hs_y:end_y] = np.roll(result[hs_y:end_y], int(hs_shift), axis=1)
            result[hs_y:end_y] += np.random.normal(
                0, 0.06, result[hs_y:end_y].shape).astype(np.float32)

        # 10. RARE DROPOUTS — bright horizontal streak with exponential decay tail
        if np.random.random() < 0.05 * self.intensity:
            dy = np.random.randint(0, self.vhs_h)
            dx_start = np.random.randint(0, self.vhs_w // 2)
            dx_len = np.random.randint(20, 120)
            dx_end = min(dx_start + dx_len, self.vhs_w)
            streak_len = dx_end - dx_start
            if streak_len > 0:
                decay = np.exp(-np.linspace(0, 4, streak_len)).astype(np.float32)
                brightness = 0.7 + np.random.uniform(0, 0.25)
                for c in range(3):
                    result[dy, dx_start:dx_end, c] = np.clip(
                        result[dy, dx_start:dx_end, c] + brightness * decay, 0, 1)

        result = np.clip(result, 0, 1)

        # 11. INTERLACING (480i — the defining VHS motion characteristic)
        # VHS is natively interlaced: even/odd fields from different time points
        # Blend current frame with previous to create combing on motion
        if self.prev_frame is not None and self.prev_frame.shape == result.shape:
            field_select = self.frame_count % 2  # alternate even/odd fields
            for row in range(field_select, self.vhs_h, 2):
                result[row] = self.prev_frame[row]
        self.prev_frame = result.copy()

        # 12. Convert to uint8
        result = np.clip(result, 0, 1)
        result_u8 = (result * 255).astype(np.uint8)

        # 13. Upscale back to original resolution
        result_u8 = cv2.resize(result_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        return result_u8


# ═════════════════════════════════════════════════════════════════════════════
# CINEMATIC MODE — Modern Neg→Print Photochemical Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class CinematicProcessor:
    """Modern film photochemical pipeline (Vision3 500T → 2383 print).

    Pipeline order (CRITICAL — order matters):
    1. Highlight soft-clip (compress top ~2 stops — biggest visual payoff)
    2. Film softness (mild blur to remove digital crispness)
    3. Neg→print conversion (spectral_film_lut at half resolution for speed)
    4. Halation (threshold bright → large red-weighted blur → additive)
    5. Grain (exposure-dependent, independent per frame, per-channel)
    6. Film breath (per-frame micro-exposure variation)
    7. Bloom (soft highlight glow)
    8. Vignette
    9. Gate weave (Brownian sub-pixel shift — applied LAST)
    """

    def __init__(self, width, height, fps, intensity=1.0):
        self.w = width
        self.h = height
        self.fps = fps
        self.intensity = intensity

        # Process at half resolution for speed
        self.proc_h = max(1, height // 2)
        self.proc_w = max(1, width // 2)

        # Initialize spectral film conversion
        self.film_conv = None
        self.film_neg = None
        if HAS_FILM_SCIENCE:
            try:
                neg = FilmSpectral(sfl.KODAK_5219)
                prt = FilmSpectral(sfl.KODAK_2383)
                self.film_conv = FilmSpectral.generate_conversion(
                    negative_film=neg, print_film=prt,
                    input_colourspace="sRGB", output_colourspace="sRGB",
                    projector_kelvin=6500, exposure_kelvin=4500,
                    exp_comp=0.3, gamut_compression=0.2, sat_adjust=1.0,
                    mode="full",
                )
                self.film_neg = neg
            except Exception as e:
                print(f"  Warning: spectral_film_lut init failed: {e}")

        # Gate weave (visible — even modern 35mm has slight frame instability)
        self.weave_x = BrownianWalk(sigma=0.35 * intensity, spring=0.04)
        self.weave_y = BrownianWalk(sigma=0.25 * intensity, spring=0.04)

        # Grain: independent per frame (alpha=0), per-channel, half-res for clumping
        self.grain = TemporalGrain(self.proc_h, self.proc_w, channels=3, alpha=0.0, gen_scale=2)
        # C4: Coarse grain for shadows (fewer, larger crystals)
        self.grain_coarse = TemporalGrain(self.proc_h, self.proc_w, channels=3, alpha=0.0, gen_scale=4)

        # Film breath oscillators (exposure + contrast + color temperature)
        self.breath_t = 0.0
        self.breath_f1 = 0.08 + np.random.uniform(-0.02, 0.02)
        self.breath_f2 = 0.17 + np.random.uniform(-0.04, 0.04)
        self.breath_phi1 = np.random.uniform(0, 2 * np.pi)
        self.breath_phi2 = np.random.uniform(0, 2 * np.pi)
        # C3: contrast drift oscillator
        self.breath_contrast_f = 0.05 + np.random.uniform(-0.01, 0.01)
        self.breath_contrast_phi = np.random.uniform(0, 2 * np.pi)
        # C3: color temperature drift oscillators (R and B)
        self.breath_temp_f = 0.03 + np.random.uniform(-0.01, 0.01)
        self.breath_temp_phi = np.random.uniform(0, 2 * np.pi)

        # Pre-build highlight soft-clip LUT
        self._build_softclip_lut()

    def _build_softclip_lut(self):
        """Per-channel highlight soft-clip: film emulsion layers clip independently.
        Blue clips first, then green, then red — produces warm-colored highlights."""
        x = np.linspace(0, 1, 4096, dtype=np.float32)
        k = 2.5
        # C1: Per-channel rolloff with different knee points (BGR order)
        knees = {'b': 0.65, 'g': 0.70, 'r': 0.75}
        self.softclip_luts = {}
        for ch, knee in knees.items():
            out = np.where(
                x <= knee,
                x,
                knee + (1.0 - knee) * (1.0 - np.exp(-k * (x - knee) / (1.0 - knee))) /
                (1.0 - np.exp(-k))
            )
            self.softclip_luts[ch] = out.astype(np.float32)
        self.softclip_lut = self.softclip_luts['g']  # default fallback

    def process_frame(self, frame, frame_idx):
        orig_h, orig_w = frame.shape[:2]
        # Downscale to half resolution for speed
        small = cv2.resize(frame, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
        h, w = self.proc_h, self.proc_w
        fimg = small.astype(np.float32) / 255.0

        # 1. PER-CHANNEL HIGHLIGHT SOFT-CLIP (blue clips first → warm highlights)
        fimg[:, :, 0] = apply_lut(fimg[:, :, 0], self.softclip_luts['b'])  # Blue
        fimg[:, :, 1] = apply_lut(fimg[:, :, 1], self.softclip_luts['g'])  # Green
        fimg[:, :, 2] = apply_lut(fimg[:, :, 2], self.softclip_luts['r'])  # Red

        # 2. FILM SOFTNESS (remove digital crispness — film has specific MTF curve)
        fimg = cv2.GaussianBlur(fimg, (3, 3), 1.2)

        # 3. NEGATIVE → PRINT CONVERSION (already at half resolution)
        if self.film_conv is not None:
            converted = self.film_conv(fimg.reshape(-1, 3))
            fimg = np.clip(converted.reshape(h, w, 3), 0, 1).astype(np.float32)
        else:
            # Fallback: cross-coupling matrix (inter-layer chemical interaction) + grade
            # Simulates how neg emulsion layers interact during development
            xc_matrix = np.array([
                [ 1.02, -0.02,  0.04],  # B picks up slight R
                [ 0.01,  1.00, -0.03],  # G slightly loses B
                [-0.03,  0.05,  1.02],  # R picks up slight G
            ], dtype=np.float32)
            pixels = fimg.reshape(-1, 3)
            fimg = np.clip((pixels @ xc_matrix.T).reshape(h, w, 3), 0, 1).astype(np.float32)
            # Manual warm/teal grade with S-curve
            gray = np.mean(fimg, axis=2)
            shadow_mask = np.clip(1.0 - gray * 2, 0, 1)
            highlight_mask = np.clip(gray * 2 - 1, 0, 1)
            fimg[:, :, 0] += shadow_mask * 0.04
            fimg[:, :, 1] += shadow_mask * 0.02
            fimg[:, :, 2] += highlight_mask * 0.05
            fimg[:, :, 1] += highlight_mask * 0.02
            fimg = 0.5 + (fimg - 0.5) * 1.15
            fimg = fimg * 0.92 + 0.03
            fimg = np.clip(fimg, 0, 1)

        # 4. PHYSICAL HALATION (red-weighted — film layers are B-G-R)
        fimg = np.clip(fimg, 0, 1)
        fimg = physical_halation(fimg, threshold=0.65, radius_frac=0.05,
                                  color=(0.15 * self.intensity,
                                         0.06 * self.intensity,
                                         0.02 * self.intensity))

        # 4b. VEILING FLARE (film-era lenses — global contrast reduction from bright areas)
        gray_vf = 0.299 * fimg[:, :, 2] + 0.587 * fimg[:, :, 1] + 0.114 * fimg[:, :, 0]
        highlight_energy = np.mean(np.clip(gray_vf - 0.6, 0, 1))
        flare_strength = highlight_energy * 0.15 * self.intensity
        fimg = fimg * (1.0 - flare_strength) + flare_strength * 0.3  # lift shadows, reduce contrast
        fimg = np.clip(fimg, 0, 1).astype(np.float32)

        # 5. DUAL-SCALE GRAIN — fine grain (midtones/highlights) + coarse grain (shadows)
        grains_fine = self.grain.generate()
        grains_coarse = self.grain_coarse.generate()
        grain_intensity = 0.07 * self.intensity

        if self.film_neg is not None:
            try:
                pixels = fimg.reshape(-1, 3)
                scale = max(w, h) / 36.0
                gf = self.film_neg.grain_transform(
                    pixels, scale=scale / 160.0, std_div=1.0
                ).reshape(h, w, 3).astype(np.float32)
                # Blend fine/coarse grain by luminance
                lum = np.mean(fimg, axis=2)
                shadow_w = np.clip(1.0 - lum * 2.5, 0, 1)  # coarse in shadows
                highlight_w = 1.0 - shadow_w  # fine in mids/highs
                for ci, co in [(2, 0), (1, 1), (0, 2)]:
                    combined = grains_fine[co] * highlight_w + grains_coarse[co] * shadow_w
                    fimg[:, :, ci] += combined * gf[:, :, co]
            except Exception:
                lum = np.mean(fimg, axis=2)
                lum_resp = np.clip(np.sqrt(4 * lum * (1 - lum)), 0.3, 1.0)
                shadow_w = np.clip(1.0 - lum * 2.5, 0, 1)
                highlight_w = 1.0 - shadow_w
                for ci, co, scale in [(2, 0, 1.0), (1, 1, 0.9), (0, 2, 1.1)]:
                    combined = grains_fine[co] * highlight_w + grains_coarse[co] * shadow_w
                    fimg[:, :, ci] += combined * grain_intensity * scale * lum_resp
        else:
            lum = np.mean(fimg, axis=2)
            lum_resp = np.clip(np.sqrt(4 * lum * (1 - lum)), 0.3, 1.0)
            shadow_w = np.clip(1.0 - lum * 2.5, 0, 1)
            highlight_w = 1.0 - shadow_w
            for ci, co, scale in [(2, 0, 1.0), (1, 1, 0.9), (0, 2, 1.1)]:
                combined = grains_fine[co] * highlight_w + grains_coarse[co] * shadow_w
                fimg[:, :, ci] += combined * grain_intensity * scale * lum_resp

        # 6. FILM BREATH (exposure + contrast + color temperature micro-variation)
        dt = 1.0 / max(self.fps, 1)
        self.breath_t += dt
        t = self.breath_t
        # Exposure drift
        breath = 1.0 + self.intensity * 0.025 * (
            np.sin(2 * np.pi * self.breath_f1 * t + self.breath_phi1) * 0.6 +
            np.sin(2 * np.pi * self.breath_f2 * t + self.breath_phi2) * 0.4
        )
        fimg = fimg * breath
        # Contrast drift (low-frequency sinusoidal)
        contrast_drift = 1.0 + self.intensity * 0.015 * np.sin(
            2 * np.pi * self.breath_contrast_f * t + self.breath_contrast_phi)
        fimg = 0.5 + (fimg - 0.5) * contrast_drift
        # Color temperature drift (R/B channels)
        temp_drift = self.intensity * 0.008 * np.sin(
            2 * np.pi * self.breath_temp_f * t + self.breath_temp_phi)
        fimg[:, :, 2] *= 1.0 + temp_drift   # Red
        fimg[:, :, 0] *= 1.0 - temp_drift   # Blue

        # 7. BLOOM (soft highlight glow)
        fimg = np.clip(fimg, 0, 1)
        gray = 0.299 * fimg[:, :, 2] + 0.587 * fimg[:, :, 1] + 0.114 * fimg[:, :, 0]
        bloom_mask = np.clip((gray - 0.5) / 0.5, 0, 1).astype(np.float32)
        bloom = fimg.copy()
        for c in range(3):
            bloom[:, :, c] *= bloom_mask
        br = min(max(h, w) // 30, 151)
        if br % 2 == 0:
            br += 1
        bloom = cv2.GaussianBlur(bloom, (br, br), br / 3.0)
        fimg = np.clip(1.0 - (1.0 - fimg) * (1.0 - bloom * 0.14), 0, 1)

        # 8. VIGNETTE (modern lens — subtle)
        y_coord, x_coord = np.mgrid[0:h, 0:w].astype(np.float32)
        max_dim = max(w / 2, h / 2)
        dist = np.sqrt(((x_coord - w / 2) / max_dim) ** 2 +
                        ((y_coord - h / 2) / max_dim) ** 2)
        vig = 1.0 - 0.30 * self.intensity * np.clip((dist - 0.5) / 1.0, 0, 1) ** 2
        for c in range(3):
            fimg[:, :, c] *= vig

        # 9. GATE WEAVE (Brownian — applied LAST)
        dx_shift = self.weave_x.step()
        dy_shift = self.weave_y.step()
        M = np.float32([[1, 0, dx_shift], [0, 1, dy_shift]])
        result_u8 = (np.clip(fimg, 0, 1) * 255).astype(np.uint8)
        result_u8 = cv2.warpAffine(result_u8, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Upscale back to original resolution
        result_u8 = cv2.resize(result_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return result_u8


# ═════════════════════════════════════════════════════════════════════════════
# VIDEO PIPELINE — ffmpeg pipe I/O (no temp files, no quality loss)
# ═════════════════════════════════════════════════════════════════════════════

class VideoPipeline:
    """Reads/writes video frames via ffmpeg subprocess pipes."""

    def __init__(self, input_path, output_path, mode, fps=None, intensity=1.0, seed=None):
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.mode = mode
        self.target_fps = fps
        self.intensity = intensity

        if seed is not None:
            np.random.seed(seed)

        self._probe()

    def _probe(self):
        """Get video info using ffprobe, handling rotation metadata (iPhone videos)."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", self.input_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            import json
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    raw_w = int(stream["width"])
                    raw_h = int(stream["height"])
                    fps_str = stream.get("r_frame_rate", "24/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        self.fps = float(num) / float(den)
                    else:
                        self.fps = float(fps_str)
                    self.total_frames = int(stream.get("nb_frames", 0))
                    if self.total_frames == 0:
                        duration = float(info.get("format", {}).get("duration", 0))
                        self.total_frames = int(duration * self.fps)

                    rotation = 0
                    for sd in stream.get("side_data_list", []):
                        if "rotation" in sd:
                            rotation = abs(int(sd["rotation"]))
                    tags = stream.get("tags", {})
                    if "rotate" in tags:
                        rotation = abs(int(tags["rotate"]))

                    if rotation in (90, 270):
                        self.width = raw_h
                        self.height = raw_w
                    else:
                        self.width = raw_w
                        self.height = raw_h

                    print(f"  Detected: {raw_w}x{raw_h} (stored), rotation={rotation}°"
                          f" → {self.width}x{self.height} (output)")
                    break
        except Exception:
            cap = cv2.VideoCapture(self.input_path)
            self.width = int(cap.get(3))
            self.height = int(cap.get(4))
            self.fps = cap.get(5)
            self.total_frames = int(cap.get(7))
            cap.release()

    def run(self):
        """Process the video through the selected mode pipeline."""
        out_fps = self.target_fps or self.fps
        if self.mode == "silent" and not self.target_fps:
            out_fps = 18

        print(f"\n  Input:  {self.width}x{self.height} @ {self.fps:.1f}fps, ~{self.total_frames} frames")
        print(f"  Mode:   {self.mode}")
        print(f"  Output: {out_fps:.1f}fps → {self.output_path}")

        processor = self._create_processor()

        frame_skip = max(1, int(self.fps / out_fps)) if self.mode == "silent" else 1

        read_cmd = [
            "ffmpeg", "-i", self.input_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "pipe:1"
        ]
        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)

        out_w, out_h = self.width, self.height

        write_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(out_fps),
            "-i", "pipe:0",
        ]

        if self.mode != "silent":
            write_cmd += ["-i", self.input_path, "-map", "0:v", "-map", "1:a?"]

        if self.mode == "vhs":
            write_cmd += [
                "-af", "highpass=f=100,lowpass=f=7000,aecho=0.6:0.3:40|100:0.3|0.2",
                "-c:v", "libx264", "-crf", "22", "-preset", "medium",
                "-c:a", "aac", "-b:a", "96k",
            ]
        elif self.mode == "silent":
            write_cmd += [
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            ]
        else:
            write_cmd += [
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-c:a", "copy",
            ]

        write_cmd += ["-pix_fmt", "yuv420p", "-v", "quiet", self.output_path]
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)

        frame_size = self.width * self.height * 3
        frame_idx = 0
        processed = 0
        start_time = time.time()

        print(f"  Processing...", flush=True)

        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            if self.mode == "silent" and frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

            result = processor.process_frame(frame.copy(), frame_idx)

            if result.shape[1] != out_w or result.shape[0] != out_h:
                result = cv2.resize(result, (out_w, out_h))

            writer.stdin.write(result.tobytes())

            processed += 1
            frame_idx += 1

            if processed % 30 == 0 or (self.total_frames > 0 and frame_idx >= self.total_frames - 1):
                elapsed = time.time() - start_time
                fps_actual = processed / max(elapsed, 0.01)
                if self.total_frames > 0:
                    pct = frame_idx / self.total_frames * 100
                    remaining = (self.total_frames - frame_idx) / max(fps_actual * frame_skip, 0.01)
                    print(f"    {pct:.0f}% — {fps_actual:.1f} fps — ~{remaining:.0f}s remaining",
                          flush=True)
                else:
                    print(f"    {processed} frames — {fps_actual:.1f} fps", flush=True)

        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

        elapsed = time.time() - start_time
        if os.path.exists(self.output_path):
            size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
            print(f"  Done! {self.output_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
        else:
            print(f"  ERROR: Failed to create output video")

    def _create_processor(self):
        if self.mode == "silent":
            return SilentProcessor(self.width, self.height, self.fps, self.intensity)
        elif self.mode == "golden":
            return GoldenProcessor(self.width, self.height, self.fps, self.intensity)
        elif self.mode == "vhs":
            return VHSProcessor(self.width, self.height, self.fps, self.intensity)
        elif self.mode == "cinematic":
            return CinematicProcessor(self.width, self.height, self.fps, self.intensity)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Vintage Video — physics-based period footage simulation"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-m", "--mode", default="cinematic",
                        choices=["silent", "golden", "vhs", "cinematic"],
                        help="Vintage mode (default: cinematic)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: input_MODE.mp4)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output FPS (default: auto)")
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="Effect intensity 0.0-2.0 (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"{input_path.stem}_{args.mode}.mp4")

    mode_names = {
        "silent": "1910s ORTHOCHROMATIC FILM + CARBON ARC",
        "golden": "1950s TECHNICOLOR (CROSSTALK-REMOVAL)",
        "vhs": "1990s VHS TAPE (SOFTNESS + CHROMA SMEAR)",
        "cinematic": "VISION3 500T → 2383 PRINT PIPELINE",
    }

    print(f"\n{'=' * 55}")
    print(f"  VINTAGE VIDEO v3.1 — {mode_names[args.mode]}")
    print(f"{'=' * 55}")

    pipeline = VideoPipeline(
        input_path, output_path, args.mode,
        fps=args.fps, intensity=args.intensity, seed=args.seed
    )
    pipeline.run()


if __name__ == "__main__":
    main()
