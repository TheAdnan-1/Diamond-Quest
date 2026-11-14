"""Procedural audio engine: synthesized instruments, effects, cave reverb and adaptive music.

Everything is generated with numpy at start-up (in a background thread) and cached to disk:
  * DSP: band-limited wavetable oscillators, FM synthesis, Karplus-Strong plucked strings,
    formant ("choir") filters, STFT time-varying filters, FFT convolution reverb.
  * ~30 sound effects, each with several variations so repeats never sound identical.
  * Per-cave music in two synchronized stems (explore + tension) that crossfade with the
    match intensity, plus a looping ambience bed (wind, drips, birds, lava...).
"""
import math
import os
import random
import threading
import wave

import numpy as np
import pygame

RATE = 44100
AUDIO_VERSION = 10
CACHE = os.path.join(os.path.expanduser("~"), ".diamond_quest", "cache", f"audio_v{AUDIO_VERSION}")


# =========================================================================== DSP toolkit
def T(dur):
    return np.arange(max(1, int(dur * RATE)), dtype=np.float32) / RATE


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


_TABLES = {}


def _tables():
    if _TABLES:
        return _TABLES
    size = 2048
    ph = np.arange(size) / size
    for wave_ in ("saw", "square", "tri"):
        tabs = []
        for octv in range(12):
            f0 = 16.0 * 2 ** octv
            nh = int(min(size // 2 - 1, max(1, (RATE / 2) / f0)))
            k = np.arange(1, nh + 1)
            if wave_ == "saw":
                amps = 1.0 / k
            elif wave_ == "square":
                amps = np.where(k % 2 == 1, 1.0 / k, 0)
            else:
                amps = np.where(k % 2 == 1, ((-1) ** ((k - 1) // 2)) / (k * k), 0)
            tab = np.zeros(size)
            for start in range(0, nh, 128):
                kk = k[start:start + 128]
                tab += (amps[start:start + 128, None] * np.sin(2 * np.pi * kk[:, None] * ph[None, :])).sum(0)
            tab /= np.abs(tab).max()
            tabs.append(tab.astype(np.float32))
        _TABLES[wave_] = tabs
    return _TABLES


def osc(freq, dur, wave_="sine", phase0=0.0):
    """freq: scalar or per-sample array (Hz). Band-limited for saw/square/tri."""
    n = max(1, int(dur * RATE))
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    ph = (np.cumsum(f) / RATE + phase0) % 1.0
    if wave_ == "sine":
        return np.sin(2 * np.pi * ph).astype(np.float32)
    tabs = _tables()[wave_]
    octv = int(np.clip(np.log2(max(16.0, float(np.max(f))) / 16.0), 0, 11))
    tab = tabs[octv]
    idx = ph * len(tab)
    i0 = idx.astype(np.int64) % len(tab)
    fr = (idx - np.floor(idx)).astype(np.float32)
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % len(tab)] * fr


def env(n, a=0.005, d=0.1, s=0.6, r=0.2, hold=None):
    """ADSR envelope of n samples. hold = seconds before release (default: fills)."""
    t = np.arange(n) / RATE
    dur = n / RATE
    rel_start = (dur - r) if hold is None else hold
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = np.clip((t - rel_start) / max(r, 1e-6), 0, 1)
    return (e * (1 - rel)).astype(np.float32)


def perc(n, a=0.002, decay=0.2):
    t = np.arange(n) / RATE
    return (np.clip(t / a, 0, 1) * np.exp(-t / decay)).astype(np.float32)


def noise(dur, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, max(1, int(dur * RATE))).astype(np.float32)


def fft_filter(x, curve):
    """Zero-phase static filter. curve(freqs Hz) -> gain."""
    n = len(x)
    nf = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1 / RATE)
    y = np.fft.irfft(X * curve(f), nf)[:n]
    return y.astype(np.float32)


def lp(fc, order=2, res=0.0):
    def c(f):
        g = 1 / np.sqrt(1 + (f / fc) ** (2 * order))
        if res:
            g = g + res * np.exp(-((f - fc) / (fc * 0.12)) ** 2)
        return g
    return c


def hp(fc, order=2):
    return lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order))


def bp(fc, width=0.5):
    """Gaussian band-pass in log frequency; width in octaves."""
    return lambda f: np.exp(-(np.log2(np.maximum(f, 1) / fc) / width) ** 2)


def stft_filter(x, gain_fn, nfft=2048, hop=512):
    """Time-varying filter: gain_fn(times (F,1), freqs (1,B)) -> gains (F,B). Overlap-add."""
    n = len(x)
    win = np.hanning(nfft).astype(np.float32)
    pad = np.concatenate([np.zeros(nfft, np.float32), x.astype(np.float32), np.zeros(nfft + hop, np.float32)])
    nfr = (len(pad) - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    X = np.fft.rfft(pad[idx] * win, axis=1)
    times = ((np.arange(nfr) * hop + nfft / 2 - nfft) / RATE)[:, None]
    freqs = np.fft.rfftfreq(nfft, 1 / RATE)[None, :]
    Y = (np.fft.irfft(X * gain_fn(times, freqs), nfft, axis=1) * win).astype(np.float32)
    R = nfft // hop
    total = len(pad) + nfft * 2
    out = np.zeros(total, np.float32)
    norm = np.zeros(total, np.float32)
    w2 = win * win
    for k in range(R):
        blk = Y[k::R]
        m = len(blk)
        if not m:
            continue
        s0 = k * hop
        out[s0:s0 + m * nfft] += blk.ravel()
        norm[s0:s0 + m * nfft] += np.tile(w2, m)
    out = out / np.maximum(norm, 1e-3)
    return out[nfft:nfft + n]


def formant(x, vowel="a"):
    F = {"a": ((730, 1.0), (1090, 0.5), (2440, 0.25)), "o": ((450, 1.0), (800, 0.45), (2830, 0.12)),
         "u": ((320, 1.0), (870, 0.35), (2250, 0.1)), "e": ((530, 1.0), (1840, 0.45), (2480, 0.3))}[vowel]

    def c(f):
        g = np.zeros_like(f)
        for fc, amp in F:
            g += amp * np.exp(-((f - fc) / (fc * 0.11)) ** 2)
        return g + 0.02
    return fft_filter(x, c)


def pluck(freq, dur, bright=0.5, decay=0.997, seed=0):
    """Karplus-Strong plucked string (vectorised per period)."""
    total = int(dur * RATE)
    N = max(2, int(RATE / freq))
    rng = np.random.default_rng(seed)
    buf = rng.uniform(-1, 1, N).astype(np.float32)
    if bright < 1:
        k = max(1, int((1 - bright) * 6))
        for _ in range(k):
            buf = 0.5 * (buf + np.roll(buf, 1))
    out = np.zeros(total + N + 1, np.float32)
    out[1:N + 1] = buf
    s = N + 1
    while s < total + 1:
        e = min(s + N, total + 1)
        out[s:e] = decay * 0.5 * (out[s - N:e - N] + out[s - N - 1:e - N - 1])
        s = e
    return out[1:total + 1]


def fm(freq, dur, ratio=3.5, index=4.0, idecay=0.4, adecay=1.0, a=0.002):
    t = T(dur)
    I = index * np.exp(-t / idecay)
    mod = np.sin(2 * np.pi * freq * ratio * t) * I
    y = np.sin(2 * np.pi * freq * t + mod)
    return (y * np.clip(t / a, 0, 1) * np.exp(-t / adecay)).astype(np.float32)


def fold_loop(x, length):
    """Wrap anything past `length` back onto the start (seamless loops)."""
    if x.shape[0] <= length:
        out = np.zeros((length,) + x.shape[1:], np.float32)
        out[:x.shape[0]] = x
        return out
    out = x[:length].copy()
    tail = x[length:]
    while len(tail):
        k = min(len(tail), length)
        out[:k] += tail[:k]
        tail = tail[k:]
    return out


def reverb_ir(dur=2.4, bright=6000, dark=1500, seed=1, early=True):
    """Stereo impulse response: early reflections + darkening exponential noise tail."""
    n = int(dur * RATE)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        tail = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 6.9 / dur)
        tail = stft_filter(tail, lambda tt, f: 1 / np.sqrt(1 + (f / (bright + (dark - bright) *
                                                               np.clip(tt / dur, 0, 1))) ** 4))
        tail *= np.clip(t / 0.02, 0, 1)
        if early:
            for _ in range(12):
                d = int(rng.uniform(0.006, 0.07) * RATE)
                tail[d] += rng.uniform(0.3, 0.8) * (1 if rng.random() < 0.5 else -1)
        out[:, ch] = tail
    out /= np.sqrt((out ** 2).sum(0, keepdims=True)) + 1e-9
    return out


def convolve(x, ir):
    """x: (n,) or (n,2); ir: (m,2). Returns (n+m-1, 2)."""
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    n, m = len(x), len(ir)
    nf = 1 << (n + m - 2).bit_length()
    out = np.zeros((n + m - 1, 2), np.float32)
    for ch in range(2):
        out[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch], nf) * np.fft.rfft(ir[:, ch], nf), nf)[:n + m - 1]
    return out


def stereo(x, pan=0.0, width=0.0):
    """mono -> stereo with constant-power pan (-1..1) and optional Haas widening."""
    l = math.cos((pan + 1) * math.pi / 4)
    r = math.sin((pan + 1) * math.pi / 4)
    out = np.stack([x * l, x * r], 1)
    if width:
        d = int(width * 0.012 * RATE)
        out[d:, 1] = out[d:, 1] * 0.7 + out[:-d, 1] * 0.3 if d else out[:, 1]
    return out.astype(np.float32)


def place(buf, start, sig, gain=1.0):
    """Mix sig (mono or stereo) into buf (stereo) at sample `start`, clipping at the end."""
    s = int(start)
    if s >= len(buf):
        return
    if sig.ndim == 1:
        sig = np.stack([sig, sig], 1)
    e = min(len(buf), s + len(sig))
    buf[s:e] += sig[:e - s] * gain


def mix_reverb(dry, ir, wet=0.3, loop_len=None):
    w = convolve(dry, ir) * wet
    if loop_len:
        w = fold_loop(w, loop_len)
        return dry[:loop_len] + w
    out = np.zeros_like(w)
    out[:len(dry)] = dry
    return out + w


def normalize(x, peak=0.9):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


LEVELS = {"step": -27, "step_ai": -29, "dig": -22, "wait": -26, "swing": -24, "clang": -18, "stun": -20,
          "crumble": -18, "push": -20, "hook_throw": -22, "hook_pull": -20, "arm": -25, "gem_v": -18,
          "gem_r": -16, "unlock": -16, "burn": -18, "bite": -18, "crush": -14, "hurt": -22, "thud": -22,
          "splat": -22, "rattle": -24, "death": -16, "exit": -16, "victory": -14, "defeat": -16,
          "hover": -32, "click": -25, "error": -23,
          "mine_place": -22, "explode": -14, "freeze": -17, "thaw": -20, "grind": -21,
          "fire_loop": -28, "beam_loop": -31}


def loudness(x):
    """Loudest 300 ms RMS in dBFS."""
    m = x.mean(1) if x.ndim == 2 else x
    w = int(0.3 * RATE)
    if len(m) <= w:
        r = np.sqrt((m ** 2).mean())
    else:
        c = np.concatenate([[0], np.cumsum(m.astype(np.float64) ** 2)])
        r = np.sqrt(((c[w:] - c[:-w]) / w).max())
    return 20 * np.log10(max(r, 1e-9))


def level(x, target_db):
    g = 10 ** ((target_db - loudness(x)) / 20)
    y = x * g
    pk = np.abs(y).max()
    if pk > 0.95:
        y = np.tanh(y / 0.95 * 1.2) / np.tanh(1.2) * 0.95
    return y.astype(np.float32)


def soft_clip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


# =========================================================================== instruments
def inst_marimba(f, dur=0.8, vel=1.0):
    y = fm(f, dur, ratio=4.0, index=2.2, idecay=0.02, adecay=0.35) * 0.8
    y += fm(f * 2, dur, ratio=3.0, index=0.8, idecay=0.01, adecay=0.12) * 0.25
    return y * vel


def inst_kalimba(f, dur=1.4, vel=1.0):
    y = fm(f, dur, ratio=5.4, index=1.6, idecay=0.03, adecay=0.6) * 0.7
    y += pluck(f, dur, 0.85, 0.996, int(f)) * 0.35
    return y * vel


def inst_bell(f, dur=2.5, vel=1.0, ratio=3.5):
    y = fm(f, dur, ratio=ratio, index=3.0, idecay=0.5, adecay=0.9) * 0.6
    y += fm(f * 2.01, dur, ratio=1.41, index=1.2, idecay=0.3, adecay=0.5) * 0.25
    return y * vel


def inst_celesta(f, dur=1.6, vel=1.0):
    y = fm(f, dur, ratio=1.0, index=1.2, idecay=0.05, adecay=0.55) * 0.6
    y += fm(f * 4, dur, ratio=1.0, index=0.5, idecay=0.02, adecay=0.15) * 0.2
    return y * vel


def inst_pluck(f, dur=1.2, vel=1.0, bright=0.6):
    return pluck(f, dur, bright, 0.996, int(f * 3)) * vel


def inst_oud(f, dur=1.0, vel=1.0):
    y = pluck(f, dur, 0.75, 0.993, int(f * 7))
    y += pluck(f * 1.003, dur, 0.6, 0.992, int(f * 11)) * 0.6
    return fft_filter(y, lambda fr: 1 + 0.6 * np.exp(-((fr - 2200) / 600) ** 2)) * vel * 0.7


def inst_pad(freqs, dur, wave_="saw", cutoff=1200, sweep=0.0, vel=1.0, detune=0.008, attack=0.8, release=1.2):
    n = int(dur * RATE)
    y = np.zeros(n, np.float32)
    for k, f in enumerate(freqs):
        for d in (-detune, 0.0, detune):
            y += osc(f * (1 + d), dur, wave_, phase0=random.random())
    y /= max(1, len(freqs) * 3)
    if sweep:
        y = stft_filter(y, lambda t, fr: 1 / np.sqrt(1 + (fr / (cutoff * (1 + sweep * np.sin(t * 0.7) ** 2))) ** 4))
    else:
        y = fft_filter(y, lp(cutoff, 2))
    return y * env(n, attack, 1.0, 0.85, release) * vel


def inst_choir(freqs, dur, vowel="a", vel=1.0):
    n = int(dur * RATE)
    y = np.zeros(n, np.float32)
    t = T(dur)
    for f in freqs:
        for d in (-0.006, 0.0, 0.007):
            vib = 1 + 0.004 * np.sin(2 * np.pi * (5.2 + d * 50) * t)
            y += osc(f * (1 + d) * vib, dur, "saw", phase0=random.random())
    y = formant(y / max(1, len(freqs) * 3), vowel)
    return normalize(y, 1.0) * env(n, 0.9, 1.0, 0.9, 1.4) * vel


def inst_brass(f, dur, vel=1.0):
    n = int(dur * RATE)
    y = osc(f, dur, "saw") * 0.6 + osc(f * 1.004, dur, "saw") * 0.4
    y = stft_filter(y, lambda t, fr: 1 / np.sqrt(1 + (fr / (600 + 3200 * np.clip(t / 0.12, 0, 1) *
                                                          np.exp(-np.clip(t - 0.12, 0, 9) * 1.5))) ** 4))
    return y * env(n, 0.03, 0.3, 0.7, 0.25) * vel


def inst_bass(f, dur, vel=1.0, wave_="tri", drive=0.0):
    n = int(dur * RATE)
    y = osc(f, dur, wave_) * 0.8 + osc(f * 0.5, dur, "sine") * 0.4
    if drive:
        y = soft_clip(fft_filter(y, lp(900, 2)), 1 + drive * 3)
    return y * env(n, 0.005, 0.15, 0.6, 0.08) * vel


def drum_kick(vel=1.0, dur=0.45):
    t = T(dur)
    f = 45 + 90 * np.exp(-t / 0.035)
    y = osc(f, dur, "sine") * np.exp(-t / 0.22)
    click = np.zeros_like(y)
    click[:int(0.004 * RATE)] = np.random.default_rng(1).uniform(-1, 1, int(0.004 * RATE)) * 0.4
    return (y + fft_filter(click, lp(4000))) * vel


def drum_taiko(vel=1.0, pitch=70, seed=0):
    dur = 1.0
    t = T(dur)
    f = pitch * (1 + 0.6 * np.exp(-t / 0.05))
    body = osc(f, dur, "sine") * np.exp(-t / 0.35)
    body += osc(f * 1.52, dur, "sine") * np.exp(-t / 0.15) * 0.3
    hit = fft_filter(noise(dur, seed) * np.exp(-t / 0.04), bp(500, 1.2)) * 0.6
    return (body + hit) * vel


def drum_tom(pitch=120, vel=1.0, seed=0):
    dur = 0.6
    t = T(dur)
    f = pitch * (1 + 0.4 * np.exp(-t / 0.04))
    y = osc(f, dur, "sine") * np.exp(-t / 0.2)
    y += fft_filter(noise(dur, seed) * np.exp(-t / 0.03), bp(900, 1)) * 0.3
    return y * vel


def drum_snare(vel=1.0, seed=0):
    dur = 0.35
    t = T(dur)
    y = osc(190, dur, "sine") * np.exp(-t / 0.06) * 0.5
    y += fft_filter(noise(dur, seed), hp(1500)) * np.exp(-t / 0.09)
    return y * vel


def drum_shaker(vel=1.0, seed=0, dur=0.09):
    t = T(dur)
    y = fft_filter(noise(dur, seed), bp(7000, 0.8)) * np.clip(t / 0.015, 0, 1) * np.exp(-t / 0.03)
    return y * vel


def drum_frame(vel=1.0, seed=0, pitch=95):
    dur = 0.5
    t = T(dur)
    y = osc(pitch * (1 + 0.3 * np.exp(-t / 0.03)), dur, "sine") * np.exp(-t / 0.18)
    y += fft_filter(noise(dur, seed) * np.exp(-t / 0.02), bp(2500, 1.0)) * 0.35
    return y * vel


def drum_bongo(pitch=320, vel=1.0, seed=0):
    dur = 0.25
    t = T(dur)
    y = osc(pitch * (1 + 0.2 * np.exp(-t / 0.01)), dur, "sine") * np.exp(-t / 0.07)
    y += fft_filter(noise(dur, seed) * np.exp(-t / 0.006), bp(3000, 1)) * 0.3
    return y * vel


def metal_hit(f=420, dur=1.2, vel=1.0, seed=0):
    t = T(dur)
    y = np.zeros_like(t)
    rng = np.random.default_rng(seed)
    for r, a in ((1.0, 1.0), (2.76, 0.6), (5.40, 0.4), (8.93, 0.25), (13.3, 0.15)):
        y += np.sin(2 * np.pi * f * r * (1 + rng.uniform(-0.01, 0.01)) * t) * a * np.exp(-t / (dur * 0.35 / r ** 0.4))
    y += fft_filter(noise(dur, seed) * np.exp(-t / 0.01), hp(2000)) * 0.6
    return y * vel


# =========================================================================== sound effects
def _grains(dur, count, seed, band=(800, 4000), decay=0.02, spread=1.0):
    rng = np.random.default_rng(seed)
    n = int(dur * RATE)
    out = np.zeros(n, np.float32)
    for _ in range(count):
        s = int(rng.uniform(0, spread) * n * rng.uniform(0, 1))
        gl = int(rng.uniform(0.01, 0.05) * RATE)
        g = rng.uniform(-1, 1, gl).astype(np.float32) * np.exp(-np.arange(gl) / (decay * RATE))
        e = min(n, s + gl)
        out[s:e] += g[:e - s] * rng.uniform(0.3, 1.0)
    fc = rng.uniform(*band)
    return fft_filter(out, bp(fc, 1.2))


def build_effects(ir_small, ir_big):
    """name -> list of stereo float arrays (variations)."""
    S = {}

    def rv(x, ir=ir_small, wet=0.18):
        return mix_reverb(x, ir, wet)

    def var(fn, k=3):
        return [fn(i) for i in range(k)]

    # ---- movement
    def step(i):
        t = T(0.16)
        y = fft_filter(noise(0.16, 10 + i) * np.exp(-t / 0.025), bp(700 + i * 150, 1.3)) * 0.8
        y += osc(90 + i * 8, 0.16, "sine") * np.exp(-t / 0.04) * 0.6
        y += _grains(0.16, 6, 20 + i, (1500, 4000), 0.006, 0.6) * 0.25
        return rv(stereo(y, 0), wet=0.12)
    S["step"] = var(step, 4)

    def step_ai(i):
        t = T(0.22)
        y = metal_hit(260 + i * 25, 0.22, 0.35, 30 + i)
        y += fft_filter(noise(0.22, 40 + i) * np.exp(-t / 0.05), bp(1800, 0.6)) * 0.3
        whirr = osc(600 + 300 * np.exp(-t / 0.05), 0.22, "square") * np.exp(-t / 0.06) * 0.08
        return rv(stereo(y + fft_filter(whirr, lp(2500)), 0), wet=0.12)
    S["step_ai"] = var(step_ai, 3)

    def dig(i):
        t = T(0.35)
        y = _grains(0.35, 22, 50 + i, (500, 1600), 0.015, 0.7) * 1.2
        y += osc(70, 0.35, "sine") * np.exp(-t / 0.06) * 0.5
        y += fft_filter(noise(0.35, 60 + i) * np.exp(-t / 0.08), lp(900)) * 0.6
        return rv(stereo(y, 0, 0.5), wet=0.15)
    S["dig"] = var(dig, 4)

    S["wait"] = [rv(stereo(inst_celesta(midi(84), 0.6, 0.5) + inst_celesta(midi(91), 0.6, 0.25), 0), ir_big, 0.3)]

    # ---- tools
    def swing(i):
        dur = 0.32
        y = noise(dur, 70 + i)
        y = stft_filter(y, lambda t, f: np.exp(-(np.log2(np.maximum(f, 1) / (600 + 3000 * np.clip(t / dur, 0, 1)))
                                                  / 0.6) ** 2))
        y *= np.sin(np.pi * np.clip(T(dur) / dur, 0, 1)) ** 2
        return stereo(y * 0.8, -0.3 + i * 0.3)
    S["swing"] = var(swing, 3)

    def clang(i):
        y = metal_hit(330 + i * 40, 1.6, 1.0, 80 + i)
        t = T(1.6)
        y += osc(110, 1.6, "sine") * np.exp(-t / 0.08) * 0.5
        return rv(stereo(y, 0, 0.6), ir_big, 0.3)
    S["clang"] = var(clang, 3)

    def stun(i):
        out = np.zeros(int(1.4 * RATE), np.float32)
        base = 79 + i
        for k, m in enumerate((0, 4, 7, 12, 16)):
            place2 = inst_bell(midi(base + m), 0.9, 0.5, ratio=3.0)
            s = int(k * 0.05 * RATE)
            out[s:s + len(place2)] += place2[:len(out) - s]
        t = T(1.4)
        out += fft_filter(noise(1.4, 90 + i), bp(9000, 0.5)) * np.exp(-t / 0.3) * 0.15
        return rv(stereo(out, 0, 0.7), ir_big, 0.35)
    S["stun"] = var(stun, 2)

    def crumble(i):
        dur = 1.2
        t = T(dur)
        y = _grains(dur, 70, 100 + i, (400, 2500), 0.03, 1.0) * 1.3
        y += fft_filter(noise(dur, 110 + i), lp(300)) * np.exp(-t / 0.4) * 1.2
        y += osc(55 * (1 + 0.5 * np.exp(-t / 0.1)), dur, "sine") * np.exp(-t / 0.25) * 0.8
        return rv(stereo(y, 0, 0.8), ir_big, 0.3)
    S["crumble"] = var(crumble, 2)

    def push(i):
        dur = 0.55
        t = T(dur)
        y = noise(dur, 120 + i)
        y = stft_filter(y, lambda tt, f: np.exp(-(np.log2(np.maximum(f, 1) / (180 + 120 * np.sin(tt * 40) ** 2)) / 0.5) ** 2))
        y = y * 3 * np.sin(np.pi * np.clip(t / dur, 0, 1))
        y += _grains(dur, 30, 130 + i, (900, 2400), 0.008, 1.0) * 0.4
        return rv(stereo(y, 0, 0.4), wet=0.2)
    S["push"] = var(push, 3)

    def hook_throw(i):
        dur = 0.35
        y = noise(dur, 140 + i)
        y = stft_filter(y, lambda t, f: np.exp(-(np.log2(np.maximum(f, 1) / (2500 - 1500 * np.clip(t / dur, 0, 1))) / 0.4) ** 2))
        y *= np.sin(np.pi * np.clip(T(dur) / dur, 0, 1))
        for k in range(4):
            c = metal_hit(1800 + k * 300, 0.06, 0.15, 150 + k + i)
            s = int((0.05 + k * 0.06) * RATE)
            y[s:s + len(c)] += c[:len(y) - s]
        return stereo(y, 0, 0.5)
    S["hook_throw"] = var(hook_throw, 2)

    def hook_pull(i):
        dur = 0.5
        y = np.zeros(int(dur * RATE), np.float32)
        for k in range(8):
            c = metal_hit(1500 + (k % 3) * 400 + i * 50, 0.08, 0.35, 160 + k + i)
            s = int(k * 0.05 * RATE)
            y[s:s + len(c)] += c[:len(y) - s]
        y += fft_filter(noise(dur, 170 + i), bp(400, 0.8)) * np.sin(np.pi * np.clip(T(dur) / dur, 0, 1)) * 0.6
        return rv(stereo(y, 0, 0.5), wet=0.2)
    S["hook_pull"] = var(hook_pull, 2)

    S["arm"] = [stereo(inst_celesta(midi(88), 0.25, 0.4) + inst_celesta(midi(95), 0.25, 0.3), 0)]

    # ---- diamonds
    def gem_v(i):
        out = np.zeros(int(2.2 * RATE), np.float32)
        root = 84 + i * 2
        for k, m in enumerate((0, 7, 12)):
            b = inst_bell(midi(root + m), 1.8, 0.55 - k * 0.1, ratio=3.5 if k else 2.0)
            s = int(k * 0.06 * RATE)
            out[s:s + len(b)] += b[:len(out) - s]
        t = T(2.2)
        out += fft_filter(noise(2.2, 180 + i), bp(10000, 0.4)) * np.exp(-t / 0.25) * 0.12
        return rv(stereo(out, 0, 0.8), ir_big, 0.38)
    S["gem_v"] = var(gem_v, 3)

    def gem_r(i):
        out = np.zeros(int(3.0 * RATE), np.float32)
        root = 76 + i
        for k, m in enumerate((0, 4, 7, 12, 16, 19)):
            b = inst_bell(midi(root + m), 1.6, 0.4, ratio=3.5)
            s = int(k * 0.055 * RATE)
            out[s:s + len(b)] += b[:len(out) - s]
        ch = inst_choir([midi(root - 12), midi(root - 5), midi(root)], 1.6, "a", 0.25)
        out[int(0.2 * RATE):int(0.2 * RATE) + len(ch)] += ch[:len(out) - int(0.2 * RATE)]
        t = T(3.0)
        out += fft_filter(noise(3.0, 190 + i), bp(9000, 0.5)) * np.exp(-t / 0.5) * 0.12
        return rv(stereo(out, 0, 1.0), ir_big, 0.42)
    S["gem_r"] = var(gem_r, 2)

    def unlock(i):
        out = np.zeros(int(2.8 * RATE), np.float32)
        for k, m in enumerate((0, 4, 7, 12)):
            b = inst_brass(midi(60 + m), 0.5 if k < 3 else 1.6, 0.4)
            s = int(k * 0.13 * RATE)
            out[s:s + len(b)] += b[:len(out) - s]
            c = inst_bell(midi(84 + m), 1.2, 0.25)
            out[s:s + len(c)] += c[:len(out) - s]
        return rv(stereo(out, 0, 0.8), ir_big, 0.35)
    S["unlock"] = var(unlock, 1)

    # ---- hurt
    def burn(i):
        dur = 1.0
        t = T(dur)
        y = noise(dur, 200 + i)
        y = stft_filter(y, lambda tt, f: np.exp(-(np.log2(np.maximum(f, 1) / (300 + 1800 * np.exp(-tt * 3))) / 0.9) ** 2))
        y *= np.clip(t / 0.05, 0, 1) * np.exp(-t / 0.4) * 1.5
        crack = _grains(dur, 40, 210 + i, (2500, 6000), 0.002, 1.0) * 0.5
        sizzle = fft_filter(noise(dur, 220 + i), bp(5000, 0.6)) * np.exp(-t / 0.5) * 0.15
        body = fft_filter(y + crack + sizzle, lp(6500, 2))
        return rv(stereo(body, 0, 0.6), wet=0.2)
    S["burn"] = var(burn, 3)

    def bite(i):
        dur = 0.6
        t = T(dur)
        hiss = fft_filter(noise(dur, 230 + i), bp(5200, 0.6)) * np.clip(t / 0.08, 0, 1) * \
            np.clip((0.22 - t) / 0.04, 0, 1) * 0.5
        snap = fft_filter(noise(dur, 240 + i), bp(1400, 0.8)) * np.exp(-np.clip(t - 0.18, 0, 9) / 0.02) * (t > 0.18)
        squelch = osc(240 * (1 - 0.4 * np.clip((t - 0.18) / 0.2, 0, 1)), dur, "sine") * \
            np.exp(-np.clip(t - 0.18, 0, 9) / 0.08) * (t > 0.18) * 0.5
        return rv(stereo(hiss + snap * 1.4 + squelch, 0, 0.5), wet=0.2)
    S["bite"] = var(bite, 3)

    def crush(i):
        dur = 1.6
        t = T(dur)
        sub = osc(60 * (1 + np.exp(-t / 0.05)), dur, "sine") * np.exp(-t / 0.35) * 1.2
        body = fft_filter(noise(dur, 250 + i), lp(500)) * np.exp(-t / 0.3) * 1.3
        deb = _grains(dur, 60, 260 + i, (600, 3000), 0.03, 0.9) * 0.6
        return rv(stereo(soft_clip(sub + body, 1.6) + deb, 0, 0.8), ir_big, 0.3)
    S["crush"] = var(crush, 2)

    def hurt(i):
        dur = 0.35
        t = T(dur)
        f = (180 + i * 30) * (1 + 0.15 * np.exp(-t / 0.05)) * (1 - 0.15 * t / dur)
        y = osc(f, dur, "saw")
        y = formant(y, "u" if i % 2 else "o") * env(len(t), 0.01, 0.1, 0.7, 0.15)
        return rv(stereo(normalize(y, 0.5), 0), wet=0.12)
    S["hurt"] = var(hurt, 3)

    def thud(i):
        dur = 0.5
        t = T(dur)
        y = osc(70 * (1 + 0.5 * np.exp(-t / 0.03)), dur, "sine") * np.exp(-t / 0.12)
        y += fft_filter(noise(dur, 270 + i), lp(700)) * np.exp(-t / 0.06) * 0.8
        y += _grains(dur, 10, 280 + i, (800, 2000), 0.02, 0.6) * 0.3
        return rv(stereo(y, 0, 0.5), wet=0.2)
    S["thud"] = var(thud, 3)

    def splat(i):
        dur = 0.5
        t = T(dur)
        y = fft_filter(noise(dur, 290 + i), bp(600, 1.2)) * np.exp(-t / 0.08)
        y += osc(300 * np.exp(-t / 0.1) + 80, dur, "sine") * np.exp(-t / 0.1) * 0.5
        return rv(stereo(y, 0), wet=0.2)
    S["splat"] = var(splat, 2)

    def rattle(i):
        dur = 0.7
        y = np.zeros(int(dur * RATE), np.float32)
        for k in range(16):
            g = fft_filter(noise(0.025, 300 + k + i * 20), bp(5500, 0.6)) * np.exp(-T(0.025) / 0.006)
            s = int(k * 0.035 * RATE)
            y[s:s + len(g)] += g[:len(y) - s] * (0.6 + 0.4 * (k % 2))
        return rv(stereo(y, 0, 0.5), wet=0.15)
    S["rattle"] = var(rattle, 2)

    def death(i):
        out = np.zeros(int(3.5 * RATE), np.float32)
        for k, m in enumerate((0, -3, -7, -12)):
            c = inst_choir([midi(64 + m), midi(71 + m)], 1.0, "o", 0.5)
            s = int(k * 0.35 * RATE)
            out[s:s + len(c)] += c[:len(out) - s]
        d = drum_taiko(1.2, 55, 310)
        out[:len(d)] += d
        return rv(stereo(out, 0, 0.8), ir_big, 0.4)
    S["death"] = var(death, 1)

    def exit_(i):
        dur = 2.5
        t = T(dur)
        y = noise(dur, 320)
        y = stft_filter(y, lambda tt, f: np.exp(-(np.log2(np.maximum(f, 1) / (200 + 5000 * np.clip(tt / dur, 0, 1) ** 2)) / 0.5) ** 2))
        y *= np.sin(np.pi * np.clip(t / dur, 0, 1)) * 1.2
        for k, m in enumerate((0, 7, 12, 19, 24)):
            b = inst_bell(midi(72 + m), 1.5, 0.3)
            s = int((0.4 + k * 0.12) * RATE)
            y[s:s + len(b)] += b[:len(y) - s]
        return rv(stereo(y, 0, 1.0), ir_big, 0.4)
    S["exit"] = var(exit_, 1)

    # ---- jingles
    def victory(i):
        out = np.zeros(int(4.0 * RATE), np.float32)
        seq = [(0, 60, 0.25), (0.18, 64, 0.25), (0.36, 67, 0.25), (0.54, 72, 1.8)]
        for st, m, d in seq:
            for off in (0, 4, 7) if d > 1 else (0,):
                b = inst_brass(midi(m + off - (12 if off else 0)), d, 0.4)
                s = int(st * RATE)
                out[s:s + len(b)] += b[:len(out) - s]
        for st in (0.54, 0.62):
            tk = drum_taiko(0.8, 80, 330)
            s = int(st * RATE)
            out[s:s + len(tk)] += tk[:len(out) - s]
        for k in range(6):
            b = inst_bell(midi(84 + (0, 4, 7, 12, 16, 19)[k]), 1.6, 0.25)
            s = int((0.6 + k * 0.08) * RATE)
            out[s:s + len(b)] += b[:len(out) - s]
        return rv(stereo(out, 0, 1.0), ir_big, 0.35)
    S["victory"] = var(victory, 1)

    def defeat(i):
        out = np.zeros(int(4.0 * RATE), np.float32)
        for k, (m, d) in enumerate(((69, 0.5), (65, 0.5), (62, 0.5), (61, 2.0))):
            c = inst_choir([midi(m), midi(m - 12)], d + 0.6, "o", 0.5)
            s = int(k * 0.42 * RATE)
            out[s:s + len(c)] += c[:len(out) - s]
        tk = drum_taiko(0.8, 50, 340)
        s = int(1.26 * RATE)
        out[s:s + len(tk)] += tk[:len(out) - s]
        return rv(stereo(out, 0, 1.0), ir_big, 0.4)
    S["defeat"] = var(defeat, 1)

    # ---- mines
    def mine_place(i):
        out = np.zeros(int(0.9 * RATE), np.float32)
        t = T(0.08)
        clk = fft_filter(noise(0.08, 400 + i), bp(2500, 0.7)) * np.exp(-t / 0.01)
        out[:len(clk)] += clk
        for k in range(2):
            bp_ = osc(1760, 0.07, "square") * env(int(0.07 * RATE), 0.002, 0.03, 0.7, 0.02) * 0.35
            st = int((0.18 + k * 0.16) * RATE)
            out[st:st + len(bp_)] += fft_filter(bp_, lp(4000))
        return rv(stereo(out, 0), wet=0.15)
    S["mine_place"] = var(mine_place, 2)

    def explode(i):
        dur = 2.4
        t = T(dur)
        boom = osc(48 * (1 + 1.5 * np.exp(-t / 0.04)), dur, "sine") * np.exp(-t / 0.5) * 1.4
        blast = fft_filter(noise(dur, 410 + i), lp(1200)) * np.exp(-t / 0.35) * 1.4
        crack = fft_filter(noise(dur, 420 + i), hp(2500)) * np.exp(-t / 0.05) * 0.9
        deb = _grains(dur, 80, 430 + i, (700, 3500), 0.03, 0.9) * 0.5
        y = soft_clip(boom + blast, 2.0) + crack + deb
        return rv(stereo(y, 0, 1.0), ir_big, 0.35)
    S["explode"] = var(explode, 2)

    # ---- lens statues
    def freeze(i):
        """A bright lens zap, then ice crackling shut around you."""
        dur = 1.8
        t = T(dur)
        zap = osc(2400 * (1 + 0.6 * np.exp(-t / 0.08)), dur, "sine") * np.exp(-t / 0.18) * 0.5
        shimmer = np.zeros_like(t)
        for k, m in enumerate((0, 7, 12, 16, 19, 24)):
            b = inst_bell(midi(84 + m + (i % 2)), 1.2, 0.18)
            st = int((0.04 + k * 0.035) * RATE)
            shimmer[st:st + len(b)] += b[:len(shimmer) - st]
        crack = _grains(dur, 70, 470 + i, (2500, 9000), 0.012, 0.9) * np.exp(-np.maximum(t - 0.15, 0) / 0.35) * 0.9
        hiss = fft_filter(noise(dur, 480 + i), hp(5000)) * np.exp(-t / 0.4) * 0.35
        return rv(stereo(zap + shimmer + crack + hiss, 0, 1.0), ir_big, 0.3)
    S["freeze"] = var(freeze, 2)

    def thaw(i):
        dur = 1.2
        t = T(dur)
        shards = _grains(dur, 55, 490 + i, (1800, 7000), 0.02, 0.9) * np.exp(-t / 0.3)
        tinkle = np.zeros_like(t)
        for k, m in enumerate((24, 19, 12)):
            b = inst_celesta(midi(84 + m), 0.5, 0.3)
            st = int((0.05 + k * 0.07) * RATE)
            tinkle[st:st + len(b)] += b[:len(tinkle) - st]
        return rv(stereo(shards + tinkle, 0, 1.0), wet=0.25)
    S["thaw"] = var(thaw, 2)

    def grind(i):
        """Heavy stone statue scraping round on its base."""
        dur = 0.9
        t = T(dur)
        y = fft_filter(noise(dur, 500 + i), bp(260, 0.9)) * (0.6 + 0.4 * np.sin(t * 60)) * \
            np.sin(np.pi * np.clip(t / dur, 0, 1)) * 1.5
        y += _grains(dur, 30, 510 + i, (300, 1500), 0.03, 0.8) * 0.4
        return rv(stereo(y, 0), wet=0.2)
    S["grind"] = var(grind, 2)

    # ---- UI
    S["hover"] = [stereo(inst_celesta(midi(96), 0.12, 0.2), 0)]
    S["click"] = [rv(stereo(inst_marimba(midi(84), 0.3, 0.6) + inst_marimba(midi(91), 0.3, 0.3), 0), wet=0.15)]
    err = np.zeros(int(0.3 * RATE), np.float32)
    for k, f in enumerate((196, 165)):
        b = osc(f, 0.14, "square") * env(int(0.14 * RATE), 0.005, 0.05, 0.6, 0.04)
        st = int(k * 0.11 * RATE)
        err[st:st + len(b)] += b
    S["error"] = [stereo(fft_filter(err, lp(1600)) * 0.4, 0)]
    return S


# =========================================================================== music
STYLES = {
    "jungle": {"root": 50, "bpm": 96, "scale": [0, 2, 3, 5, 7, 9, 10],
               "prog": [[0, 3, 7], [-2, 2, 5], [5, 9, 12], [0, 3, 7], [-4, 0, 3], [-2, 2, 5], [-5, -2, 2], [0, 3, 7]],
               "lead": "kalimba", "arp": "marimba", "pad": ("saw", 1100), "perc": "wood", "choir": "o"},
    "frost": {"root": 52, "bpm": 84, "scale": [0, 2, 3, 5, 7, 8, 10],
              "prog": [[0, 3, 7, 14], [-4, 0, 3, 7], [3, 7, 10, 14], [-2, 2, 5, 9], [5, 8, 12, 15], [0, 3, 7, 14],
                       [-4, 0, 3, 7], [-1, 3, 6, 11]],
              "lead": "celesta", "arp": "bell", "pad": ("tri", 2600), "perc": "glass", "choir": "u"},
    "ember": {"root": 45, "bpm": 100, "scale": [0, 1, 4, 5, 7, 8, 10],
              "prog": [[0, 4, 7], [1, 5, 8], [0, 4, 7], [-2, 1, 5], [5, 8, 12], [1, 5, 8], [0, 4, 7], [0, 4, 7]],
              "lead": "oud", "arp": "pluck", "pad": ("saw", 650), "perc": "frame", "choir": "a"},
}


def _melody(style, rng, bars, beat):
    """Generate a motif-based melody: list of (start_beat, midi, length_beats)."""
    sc = style["scale"]
    root = style["root"] + 24
    notes = []
    rhythms = [[0, 1, 1.5, 2, 3], [0, 0.5, 1, 2, 2.5, 3], [0, 1.5, 2, 3.5], [0, 2, 2.5, 3], [0, 0.75, 1.5, 3]]
    motif = rng.choice(rhythms)
    deg = 4
    for bar in range(bars):
        chord = style["prog"][bar % len(style["prog"])]
        rh = motif if bar % 4 != 3 else rng.choice(rhythms)
        if bar % 8 == 7:
            rh = [0, 2]
        for k, b in enumerate(rh):
            strong = b in (0, 2)
            if strong:
                target = rng.choice(chord)
                cands = [d for d in range(-7, 15) if (sc[d % 7] + 12 * (d // 7)) % 12 == target % 12]
                deg = min(cands, key=lambda d: abs(d - deg)) if cands else deg
            else:
                deg += rng.choice([-1, 1, 1, 2, -2])
            deg = int(np.clip(deg, -3, 11))
            m = root + sc[deg % 7] + 12 * (deg // 7)
            nxt = rh[k + 1] if k + 1 < len(rh) else 4
            notes.append((bar * 4 + b, m, nxt - b))
    return notes


def render_music(biome, seed=7):
    """Two synchronised stems for a biome: (explore, tension), each a seamless stereo loop."""
    st = STYLES[biome]
    rng = random.Random(seed + len(biome))
    random.seed(seed)
    bpm = st["bpm"]
    beat = 60 / bpm
    bars = 16
    length = int(round(bars * 4 * beat * RATE))
    A = np.zeros((length + RATE * 3, 2), np.float32)
    Bt = np.zeros((length + RATE * 3, 2), np.float32)
    root = st["root"]
    S = lambda b: int(b * beat * RATE)  # noqa: E731
    # ---- pads (explore)
    wv, cut = st["pad"]
    for bar in range(0, bars, 2):
        ch = st["prog"][bar % len(st["prog"])]
        freqs = [midi(root + 12 + n) for n in ch]
        pad = inst_pad(freqs, beat * 8 + 1.0, wv, cut, sweep=0.6, vel=0.5, attack=1.2, release=1.5)
        place(A, S(bar * 4), stereo(pad, -0.15, 1.0))
        if bar % 4 == 0:
            ch2 = inst_choir([midi(root + 12 + ch[0]), midi(root + 24 + ch[-1] % 12)], beat * 8, st["choir"], 0.22)
            place(A, S(bar * 4), stereo(ch2, 0.2, 1.0))
    # ---- arpeggios (explore)
    for bar in range(bars):
        ch = st["prog"][bar % len(st["prog"])]
        pat = [0, 1, 2, 1, 2, 0, 1, 2] if bar % 2 == 0 else [0, 2, 1, 2, 0, 2, 1, 2]
        for k, idx in enumerate(pat):
            m = root + 24 + ch[idx % len(ch)] + (12 if k == 7 and bar % 4 == 1 else 0)
            f = midi(m)
            if st["arp"] == "marimba":
                note = inst_marimba(f, 0.6, 0.32)
            elif st["arp"] == "bell":
                note = inst_bell(f, 1.6, 0.18, ratio=3.5) if k % 2 == 0 else inst_celesta(f, 0.8, 0.14)
            else:
                note = inst_pluck(f, 0.7, 0.28, 0.5)
            place(A, S(bar * 4 + k * 0.5), stereo(note, -0.5 + 0.14 * k, 0.4))
    # ---- lead melody (explore, second half and variations)
    for (b, m, ln) in _melody(st, rng, bars, beat):
        if b < 16 and rng.random() < 0.5:
            continue
        f = midi(m)
        if st["lead"] == "kalimba":
            note = inst_kalimba(f, max(0.6, ln * beat * 1.4), 0.42)
        elif st["lead"] == "celesta":
            note = inst_celesta(f, max(0.8, ln * beat * 1.6), 0.4)
        else:
            note = inst_oud(f, max(0.6, ln * beat * 1.2), 0.5)
        place(A, S(b), stereo(note, 0.25, 0.6))
    # ---- light percussion (explore)
    for bar in range(bars):
        for k in range(8):
            b = bar * 4 + k * 0.5
            if st["perc"] == "wood":
                if k in (0, 5):
                    place(A, S(b), stereo(drum_frame(0.45, bar * 8 + k, 90), -0.1))
                if k in (3, 7):
                    place(A, S(b), stereo(drum_bongo(330 if k == 3 else 260, 0.3, bar + k), 0.4))
                place(A, S(b + 0.25), stereo(drum_shaker(0.18 + 0.08 * (k % 2), bar * 9 + k), 0.6))
            elif st["perc"] == "glass":
                if k % 2 == 1:
                    place(A, S(b), stereo(fft_filter(noise(0.04, bar * 8 + k) * np.exp(-T(0.04) / 0.008),
                                                     bp(8000, 0.4)) * 0.3, 0.5))
                if k == 0 and bar % 2 == 0:
                    place(A, S(b), stereo(drum_kick(0.35), 0))
            else:
                if k in (0, 3, 6):
                    place(A, S(b), stereo(drum_frame(0.5, bar * 8 + k, 85 if k == 0 else 110), -0.1))
                if k in (2, 5, 7):
                    place(A, S(b), stereo(drum_bongo(420, 0.22, bar + k), 0.35))
    # ---- sub bass (explore)
    for bar in range(bars):
        ch = st["prog"][bar % len(st["prog"])]
        place(A, S(bar * 4), stereo(inst_bass(midi(root - 12 + ch[0]), beat * 3.8, 0.35, "sine"), 0))
    # ---- tension stem: driving drums + ostinato bass + stabs
    for bar in range(bars):
        ch = st["prog"][bar % len(st["prog"])]
        for k in range(8):
            b = bar * 4 + k * 0.5
            if k in (0, 3, 4):
                place(Bt, S(b), stereo(drum_taiko(0.8 if k == 0 else 0.55, 62 if k == 0 else 75, bar * 8 + k), 0))
            if k in (2, 6):
                place(Bt, S(b), stereo(drum_snare(0.35, bar * 8 + k), 0.1))
            bm = root - 12 + ch[0] + (12 if k in (3, 7) else 0)
            place(Bt, S(b), stereo(inst_bass(midi(bm), beat * 0.45, 0.45, "saw", drive=0.6 if biome == "ember" else 0.2),
                                   0))
            for j in range(2):
                place(Bt, S(b + j * 0.25), stereo(drum_shaker(0.22 if j else 0.14, bar * 16 + k * 2 + j), -0.5))
        if bar % 4 == 3:
            for k in range(4):
                place(Bt, S(bar * 4 + 2 + k * 0.5), stereo(drum_tom(160 - k * 22, 0.5, bar * 4 + k), -0.3 + 0.2 * k))
        stab = inst_brass(midi(root + 12 + ch[0]), beat * 0.4, 0.22) + inst_brass(midi(root + 12 + ch[-1]), beat * 0.4, 0.18)
        place(Bt, S(bar * 4 + 3.5), stereo(stab, 0.2, 0.5))
    ir = reverb_ir(3.2 if biome == "frost" else 2.4, 7000, 1200, seed=11)
    A = mix_reverb(A, ir, 0.38, length)
    Bt = mix_reverb(Bt, ir, 0.18, length)
    A = normalize(soft_clip(A, 1.2), 0.78)
    Bt = normalize(soft_clip(Bt, 1.3), 0.8)
    return A.astype(np.float32), Bt.astype(np.float32)


def render_ambience(biome, dur=24.0, seed=3):
    n = int(dur * RATE)
    out = np.zeros((n + RATE * 3, 2), np.float32)
    rng = np.random.default_rng(seed)
    t = T(dur)
    # wind / rumble bed with slow resonant movement
    for ch in range(2):
        bed = noise(dur, seed + ch)
        if biome == "frost":
            bed = stft_filter(bed, lambda tt, f: np.exp(-(np.log2(np.maximum(f, 1) / (500 + 400 * np.sin(tt * 0.5 + ch) ** 2)) / 0.35) ** 2))
            out[:n, ch] += bed * 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * t / dur * 2 + ch))
        elif biome == "ember":
            bed = fft_filter(bed, lp(160, 3))
            out[:n, ch] += bed * 2.5
        else:
            bed = fft_filter(bed, bp(350, 1.4))
            out[:n, ch] += bed * 0.35 * (0.7 + 0.3 * np.sin(2 * np.pi * t / dur * 3 + ch))
    # events
    for _ in range(int(dur * (1.4 if biome != "frost" else 0.8))):
        s = int(rng.uniform(0, dur) * RATE)
        pan = rng.uniform(-0.8, 0.8)
        if biome == "jungle":
            k = rng.random()
            if k < 0.35:                                     # water drip
                f = rng.uniform(1100, 1900)
                tt = T(0.25)
                ev = np.sin(2 * np.pi * (f + 900 * np.exp(-tt / 0.02)) * tt) * np.exp(-tt / 0.05) * 0.25
            elif k < 0.7:                                    # bird chirp
                tt = T(0.35)
                f = rng.uniform(2500, 4200)
                ev = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.25 * np.sin(2 * np.pi * 18 * tt))) / RATE) \
                    * np.sin(np.pi * np.clip(tt / 0.35, 0, 1)) ** 2 * 0.08
            else:                                            # insect buzz
                tt = T(1.2)
                ev = fft_filter(noise(1.2, int(s)), bp(5200, 0.15)) * (0.5 + 0.5 * np.sin(2 * np.pi * 24 * tt)) * \
                    np.sin(np.pi * tt / 1.2) * 0.12
        elif biome == "frost":
            if rng.random() < 0.6:                            # ice chime
                ev = inst_bell(midi(int(rng.choice([88, 91, 93, 95, 98]))), 2.0, 0.12, ratio=3.9)
            else:                                            # ice creak
                tt = T(0.6)
                ev = fft_filter(noise(0.6, int(s)), bp(rng.uniform(300, 700), 0.2)) * np.sin(np.pi * tt / 0.6) * 0.5
        else:
            if rng.random() < 0.6:                            # lava bubble
                tt = T(0.3)
                f = rng.uniform(80, 160)
                ev = np.sin(2 * np.pi * (f + 200 * tt / 0.3) * tt) * np.exp(-tt / 0.08) * 0.5
            else:                                            # crackle burst
                ev = _grains(0.6, 25, int(s), (2500, 6000), 0.002, 1.0) * 0.25
        place(out, s, stereo(ev.astype(np.float32), pan))
    ir = reverb_ir(3.0, 5000, 900, seed=21)
    out = mix_reverb(out[:n + RATE * 3], ir, 0.45, n)
    return normalize(out, 0.55).astype(np.float32)


# =========================================================================== worker jobs
def _set_rate(rate):
    global RATE
    if rate != RATE:
        RATE = rate
        _TABLES.clear()


def trim(x, db=-62):
    """Cut the silent tail (keeps a short fade so nothing clicks)."""
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    thr = 10 ** (db / 20)
    idx = np.nonzero(m > thr)[0]
    end = int(idx[-1]) + int(0.05 * RATE) if len(idx) else len(x)
    end = min(len(x), end)
    y = x[:end].copy()
    f = min(len(y), int(0.05 * RATE))
    if f > 1:
        y[-f:] *= np.linspace(1, 0, f)[:, None] if y.ndim == 2 else np.linspace(1, 0, f)
    return y


def master(x, lo=30, hi=11000):
    """Gentle mastering EQ for loops: remove rumble/DC and soften the extreme top."""
    curve = lambda f: hp(lo, 2)(f) * (1 / np.sqrt(1 + (f / hi) ** 2))  # noqa: E731
    return np.stack([fft_filter(x[:, ch], curve) for ch in range(2)], 1).astype(np.float32)


def job_effects(rate):
    _set_rate(rate)
    from .sound_design import build_bank
    bank = build_bank(rate)
    result={}
    for name,variants in bank.items():
        mastered=[]
        for x in variants:
            y=level(x,LEVELS.get(name,-20))
            if not name.endswith('_loop'):y=trim(y)
            peak=float(np.abs(y).max())
            if peak>.94:y*=.94/peak
            mastered.append(y)
        result[name]=mastered
    return result


def job_biome(b, rate):
    _set_rate(rate)
    a, bt = render_music(b)
    a = normalize(master(a), 0.78)
    bt = normalize(master(bt, hi=9000), 0.8)
    from .sound_design import ambience
    return a,bt,master(ambience(b,rate),40,12000)


# =========================================================================== player
def _to_sound(x, channels):
    x = np.clip(x, -1, 1)
    if channels == 1:
        x = x.mean(1, keepdims=True)
    data = (x * 32767).astype(np.int16)
    return pygame.mixer.Sound(buffer=np.ascontiguousarray(data).tobytes())


def _save_wav(path, x):
    data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.ascontiguousarray(data).tobytes())


def _load_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != RATE or w.getnchannels() != 2:
            return None
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return data.reshape(-1, 2).astype(np.float32) / 32767


class Audio:
    MUSIC_A, MUSIC_B, AMBI = 0, 1, 2

    def __init__(self):
        global RATE
        self.ok = False
        self.sounds = {}
        self.music = {}
        self.ambi = {}
        self.music_on = True
        self.sfx_on = True
        self.music_vol = 0.7
        self.sfx_vol = 0.85
        self.level = 1.0
        self.intensity = 0.0
        self.cur_biome = None
        self.want_biome = None
        self._lock = threading.Lock()
        self.channels = 2
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            try:
                pygame.mixer.init(44100, -16, 2, 1024)
            except pygame.error:
                pygame.mixer.init()
            freq, _size, chans = pygame.mixer.get_init()
            RATE = freq
            self.channels = chans
            pygame.mixer.set_num_channels(32)
            pygame.mixer.set_reserved(5)
            self.ch = [pygame.mixer.Channel(i) for i in range(3)]
            self.scene_ch=[pygame.mixer.Channel(3),pygame.mixer.Channel(4)]
            self.scene_levels=[0.0,0.0]
            self._duck_until=0.0
            self._last_play={}
            self.ok = True
        except pygame.error:
            return
        threading.Thread(target=self._build_all, daemon=True).start()

    # ------------------------------------------------------------ building
    def _cache(self, name):
        try:
            os.makedirs(CACHE, exist_ok=True)
        except OSError:
            return None
        if name.startswith("fx_") or name.endswith("_amb"):
            name="edition3_"+name
        return os.path.join(CACHE, f"{name}_{RATE}.wav")

    def _cached(self, name, fn):
        path = self._cache(name)
        if path and os.path.exists(path):
            x = _load_wav(path)
            if x is not None:
                return x
        x = fn()
        if path:
            try:
                _save_wav(path, x)
            except Exception:
                pass
        return x

    def _build_all(self):
        """Load from cache or synthesise in worker processes (keeps the game loop smooth)."""
        try:
            from .assets import _pool
            pool = _pool()
        except Exception:
            pool = None
        try:
            bank = self._load_fx_cache()
            futs = {}
            if bank is None:
                futs["fx"] = pool.submit(job_effects, RATE) if pool else None
            order = [self.want_biome] + ["jungle", "frost", "ember"]
            for b in dict.fromkeys(x for x in order if x):
                if self._load_biome_cache(b):
                    continue
                futs[b] = pool.submit(job_biome, b, RATE) if pool else None
            if "fx" in futs:
                bank = futs["fx"].result() if futs["fx"] else job_effects(RATE)
                self._save_fx_cache(bank)
            with self._lock:
                self.sounds = {k: [_to_sound(x, self.channels) for x in v] for k, v in bank.items()}
            for b, f in futs.items():
                if b == "fx":
                    continue
                a, bt, amb = f.result() if f else job_biome(b, RATE)
                for k, x in (("a", a), ("b", bt), ("amb", amb)):
                    p = self._cache(f"music_{b}_{k}")
                    if p:
                        _save_wav(p, x)
                self._install_biome(b, a, bt, amb)
        except Exception:
            pass

    def _load_fx_cache(self):
        from pathlib import Path
        import json
        folder=Path(__file__).resolve().parent/'sound_bank'
        manifest=folder/'index.json'
        if manifest.exists():
            try:
                spec=json.loads(manifest.read_text())
                if spec['rate']==RATE and spec['version']==3:
                    bank={k:[_load_wav(str(folder/name)) for name in names] for k,names in spec['sounds'].items()}
                    if all(all(x is not None for x in sounds) for sounds in bank.values()):return bank
            except (OSError,ValueError,KeyError):pass
        idx = self._cache("fx_index")
        if not idx or not os.path.exists(idx.replace(".wav", ".txt")):
            return None
        bank = {}
        with open(idx.replace(".wav", ".txt")) as f:
            for nm in (ln.strip() for ln in f):
                if not nm:
                    continue
                base, i = nm.rsplit("#", 1)
                x = _load_wav(self._cache(f"fx_{base}_{i}"))
                if x is None:
                    return None
                bank.setdefault(base, []).append(x)
        return bank

    def _save_fx_cache(self, bank):
        idx = self._cache("fx_index")
        if not idx:
            return
        names = []
        for base, lst in bank.items():
            for i, x in enumerate(lst):
                _save_wav(self._cache(f"fx_{base}_{i}"), x)
                names.append(f"{base}#{i}")
        with open(idx.replace(".wav", ".txt"), "w") as f:
            f.write("\n".join(names))

    def _load_biome_cache(self, b):
        # The authored sample-instrument score ships with the game. It is
        # always preferred over procedural fallback or any older user cache.
        from pathlib import Path
        folder=Path(__file__).resolve().parent/'music_bank'
        if (folder/f'{b}_a.ogg').exists() and (folder/f'{b}_b.ogg').exists():
            try:
                mus=tuple(pygame.mixer.Sound(str(folder/f'{b}_{k}.ogg')) for k in ('a','b'))
                amb=pygame.mixer.Sound(str(Path(__file__).resolve().parent/'sound_bank'/f'ambience_{b}.wav'))
                with self._lock:self.music[b]=mus;self.ambi[b]=amb
                if self.want_biome==b and self.cur_biome!=b:self._start_biome(b)
                return True
            except (pygame.error,OSError):pass
        parts = []
        for k in ("a", "b", "amb"):
            p = self._cache(f"music_{b}_{k}")
            if k=='amb':
                from pathlib import Path
                bundled=Path(__file__).resolve().parent/'sound_bank'/f'ambience_{b}.wav'
                if bundled.exists():p=str(bundled)
            x = _load_wav(p) if p and os.path.exists(p) else None
            if x is None:
                return False
            parts.append(x)
        self._install_biome(b, *parts)
        return True

    def _install_biome(self, b, a, bt, amb):
        with self._lock:
            self.music[b] = (_to_sound(a, self.channels), _to_sound(bt, self.channels))
            self.ambi[b] = _to_sound(amb, self.channels)
        if self.want_biome == b and self.cur_biome != b:
            self._start_biome(b)

    def _build_biome(self, b):
        if b in self.music or self._load_biome_cache(b):
            return
        a, bt, amb = job_biome(b, RATE)
        self._install_biome(b, a, bt, amb)

    # ------------------------------------------------------------ control
    def set_volumes(self, music, sfx):
        self.music_vol, self.sfx_vol = max(0,min(1,music)),max(0,min(1,sfx))
        if self.ok and self.sfx_vol==0:
            for index in range(3,pygame.mixer.get_num_channels()):pygame.mixer.Channel(index).stop()
        self._apply()

    def set_music_level(self, level):
        self.level = 1.0 if level == "menu" else 0.75
        self._apply()

    def set_intensity(self, k):
        self.intensity = max(0.0, min(1.0, k))
        self._apply()

    def set_ambience(self, biome):
        """Switch the music + ambience to a cave theme (crossfades)."""
        self.want_biome = biome
        if not self.ok or biome == self.cur_biome:
            return
        if biome in self.music:
            self._start_biome(biome)
        else:
            threading.Thread(target=self._build_biome, args=(biome,), daemon=True).start()

    def _start_biome(self, b):
        with self._lock:
            mus = self.music.get(b)
            amb = self.ambi.get(b)
        if not mus:
            return
        for c in self.ch:
            c.fadeout(1200)
        self.cur_biome = b
        self.ch[0].play(mus[0], loops=-1, fade_ms=1500)
        self.ch[1].play(mus[1], loops=-1, fade_ms=1500)
        if amb:
            self.ch[2].play(amb, loops=-1, fade_ms=2500)
        self._apply()

    def _apply(self):
        if not self.ok:
            return
        mv = self.music_vol * self.level if self.music_on else 0.0
        remaining=max(0.0,getattr(self,'_duck_until',0)-pygame.time.get_ticks()/1000)
        mv*=1-.30*min(1.0,remaining/.25)
        k = self.intensity
        try:
            self.ch[0].set_volume(mv * (1.0 - 0.35 * k))
            self.ch[1].set_volume(mv * (k ** 0.8) * 0.95)
            self.ch[2].set_volume(self.sfx_vol * 0.55 if self.sfx_on else 0.0)
        except Exception:
            pass

    def toggle_music(self):
        self.music_on = not self.music_on
        self._apply()
        return self.music_on

    def toggle_sfx(self):
        self.sfx_on = not self.sfx_on
        if self.ok and not self.sfx_on:
            for index in range(3,pygame.mixer.get_num_channels()):
                pygame.mixer.Channel(index).stop()
        self._apply()
        return self.sfx_on

    def play(self, name, pan=0.5, vol=1.0):
        if not (self.ok and self.sfx_on):
            return
        lst = self.sounds.get(name)
        if not lst:
            return
        now=pygame.time.get_ticks()/1000
        gap=.055 if name in ('hover','rattle') else .018
        if now-self._last_play.get(name,-99)<gap:return
        self._last_play[name]=now
        if name in ('explode','crush','freeze'):
            self._duck_until=now+.5
            self._apply()
        snd = random.choice(lst)
        ch = snd.play()
        if ch:
            pan = min(1.0, max(0.0, pan))
            v = self.sfx_vol * vol
            ch.set_volume(v * math.cos(pan * math.pi / 2) * 1.41, v * math.sin(pan * math.pi / 2) * 1.41)

    def set_scene(self,fire=0.0,beam=0.0):
        """Quiet local fire/beam texture layers, below action transients."""
        if not self.ok:return
        for i,(name,value) in enumerate((('fire_loop',fire),('beam_loop',beam))):
            channel=self.scene_ch[i]
            value=max(0.0,min(1.0,value))
            if not self.sfx_on or self.sfx_vol==0:value=0
            sounds=self.sounds.get(name)
            if value>.025 and sounds:
                if not channel.get_busy():channel.play(sounds[0],loops=-1,fade_ms=250)
                channel.set_volume(self.sfx_vol*value*(.65 if i==0 else .5))
            elif channel.get_busy():channel.fadeout(250)
        self._apply()

    def ready(self):
        return bool(self.sounds)
