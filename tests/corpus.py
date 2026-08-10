"""Deterministic synthetic audio for the quality corpus.

Fixtures are generated rather than committed. Two reasons: a restoration test
suite needs several minutes of multi-class audio, which is tens of megabytes of
binary in a repository that would never diff usefully; and, more importantly,
generated material comes with *exact* ground truth. When the damage is applied
by a function whose parameters are known -- a 375ms drift, a 15dB expander, a
3.4dB dip between 5 and 8kHz -- a measurement harness can be checked against the
number that was actually imposed, instead of against a previous run of itself.

The `damage_*` functions deliberately reproduce the defects reported in
`docs/audio-restoration-plan.md`, so that Phase 0 can be verified without the
reference guitar pair.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SR = 48000


# --- Instruments ---------------------------------------------------------


@lru_cache(maxsize=64)
def _pluck(frequency: float, duration_s: float, sr: int, damping: float = 0.996,
           seed: int = 0) -> np.ndarray:
    """Karplus-Strong string.

    Chosen over a decaying sine because it has the two properties the transient
    and decay metrics exist to measure: a genuinely sharp attack, and a natural
    exponential tail with harmonics that die at different rates.

    Cached per frequency: the long IIR denominator makes each note cost real
    time, and the fixtures only ever use a handful of pitches. A real instrument
    also sounds the same each time the same string is plucked, so reusing the
    waveform is not a shortcut that costs realism.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed if seed else int(frequency * 1000) % 100003)
    delay = max(2, int(round(sr / frequency)))
    length = int(duration_s * sr)
    excitation = np.zeros(length)
    burst = min(delay, length)
    excitation[:burst] = rng.uniform(-1.0, 1.0, burst)

    a = np.zeros(delay + 2)
    a[0] = 1.0
    a[delay] = -0.5 * damping
    a[delay + 1] = -0.5 * damping
    note = lfilter([1.0], a, excitation)
    note.setflags(write=False)  # cached: callers must not mutate it in place
    return note


def _note_grid(seconds: float, bpm: float, sr: int) -> list[int]:
    step = int(round(60.0 / bpm * sr))
    return list(range(0, int(seconds * sr) - step, step))


def _place(canvas: np.ndarray, signal: np.ndarray, start: int, gain: float = 1.0) -> None:
    end = min(len(canvas), start + len(signal))
    if end > start:
        canvas[start:end] += signal[: end - start] * gain


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1).astype(np.float32)


def _normalise(audio: np.ndarray, peak: float = 0.5) -> np.ndarray:
    current = float(np.abs(audio).max())
    if current <= 0:
        return audio.astype(np.float32)
    return (audio * (peak / current)).astype(np.float32)


# Every real recording has a floor -- preamp noise, room tone, the converter.
# Synthetic material has digital silence between events instead, which is not
# just unrealistic but actively misleading here: a denoise gate asked about a
# signal with a -240dBFS floor correctly answers "nothing to remove", and the
# whole decision path then goes untested. -70dBFS is a quiet but ordinary floor.
DEFAULT_ROOM_DB = -70.0


def _with_room_tone(audio: np.ndarray, room_db: float, seed: int) -> np.ndarray:
    """Add a broadband bed at a known level, after normalisation.

    Added after normalising so the level is an honest dBFS figure rather than
    something the normaliser scales unpredictably.
    """
    if not np.isfinite(room_db):
        return audio
    rng = np.random.default_rng(seed)
    bed = rng.normal(0, 1, audio.shape) * 10 ** (room_db / 20)
    return (audio + bed).astype(np.float32)


def solo_guitar(seconds: float = 12.0, sr: int = SR, seed: int = 7,
                room_db: float = DEFAULT_ROOM_DB) -> np.ndarray:
    """Solo acoustic guitar: sharp picks, long decays, real stereo width.

    This is the stand-in for the reference `guitar.wav`: no percussion, a beat
    grid a tracker can only partially find, and most of its energy below 3kHz.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    left, right = np.zeros(n), np.zeros(n)
    scale = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 392.0, 493.88]

    # Phrases separated by rests. The irregular note spacing is what leaves a
    # beat tracker with partial grid coverage; the rests are what give the track
    # a wide enough level range for a gain-vs-level curve to have quiet bins to
    # report. A continuous stream of notes never drops below about -35dBFS, and
    # a downward expander with a -75dB knee would then be invisible.
    position = 0
    while position < n - sr:
        for _ in range(int(rng.integers(2, 5))):
            frequency = scale[rng.integers(0, len(scale))]
            note = _pluck(frequency, 2.5, sr)
            pan = rng.uniform(0.35, 0.65)
            gain = rng.uniform(0.5, 1.0)
            _place(left, note, position, gain * (1 - pan))
            _place(right, note, position, gain * pan)
            position += int(rng.uniform(0.28, 0.55) * sr)
        position += int(rng.uniform(1.2, 2.0) * sr)

    return _with_room_tone(_normalise(_stereo(left, right)), room_db, seed + 500)


def solo_piano(seconds: float = 12.0, sr: int = SR, seed: int = 11,
               room_db: float = DEFAULT_ROOM_DB) -> np.ndarray:
    """Struck strings with inharmonic partials and pedal-length decays."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    left, right = np.zeros(n), np.zeros(n)
    time = np.arange(int(3.0 * sr)) / sr

    for start in _note_grid(seconds, 96.0, sr):
        root = float(rng.choice([130.81, 164.81, 196.0, 261.63, 329.63, 392.0]))
        note = np.zeros_like(time)
        for partial in range(1, 9):
            # Real piano partials stretch sharp; the constant is a typical B.
            frequency = root * partial * np.sqrt(1 + 0.0004 * partial**2)
            if frequency > sr / 2:
                break
            note += np.exp(-time * (1.6 + 0.55 * partial)) * np.sin(
                2 * np.pi * frequency * time
            ) / partial
        pan = rng.uniform(0.4, 0.6)
        gain = rng.uniform(0.6, 1.0)
        _place(left, note, start, gain * (1 - pan))
        _place(right, note, start, gain * pan)

    return _with_room_tone(_normalise(_stereo(left, right)), room_db, seed + 500)


def vocal(seconds: float = 12.0, sr: int = SR, seed: int = 13,
          room_db: float = DEFAULT_ROOM_DB) -> np.ndarray:
    """Sung vowel line: vibrato, formants, breath between phrases."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    time = np.arange(n) / sr

    pitch = np.zeros(n)
    envelope = np.zeros(n)
    position = 0
    while position < n:
        length = int(rng.uniform(0.4, 1.1) * sr)
        end = min(n, position + length)
        note = float(rng.choice([196.0, 220.0, 246.94, 261.63, 293.66]))
        span = end - position
        pitch[position:end] = note
        envelope[position:end] = np.minimum(
            1.0, np.minimum(np.arange(span) / (0.06 * sr), (span - np.arange(span)) / (0.12 * sr))
        )
        position = end + int(rng.uniform(0.05, 0.3) * sr)

    vibrato = 1 + 0.012 * np.sin(2 * np.pi * 5.4 * time)
    phase = 2 * np.pi * np.cumsum(pitch * vibrato) / sr
    # Sawtooth-ish glottal source: harmonics that formants can shape.
    source = sum(np.sin(phase * k) / k for k in range(1, 25))
    voiced = source * envelope

    # Two resonators standing in for F1/F2.
    shaped = voiced
    for centre, q in ((700.0, 8.0), (1220.0, 10.0)):
        w0 = 2 * np.pi * centre / sr
        alpha = np.sin(w0) / (2 * q)
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
        shaped = shaped + 2.0 * lfilter(b, a, voiced)

    breath = rng.normal(0, 1, n) * 6e-4 * (envelope > 0)
    mono = shaped + breath
    # A voice sits centre with a little ambience, not hard mono: the two early
    # reflections differ per side, which is what gives the side channel content.
    early = np.concatenate([np.zeros(int(0.013 * sr)), mono[: n - int(0.013 * sr)]])
    late = np.concatenate([np.zeros(int(0.021 * sr)), mono[: n - int(0.021 * sr)]])
    voice = _normalise(_stereo(mono + 0.14 * early, mono + 0.14 * late))
    return _with_room_tone(voice, room_db, seed + 500)


def drums(seconds: float = 12.0, sr: int = SR, seed: int = 17, bpm: float = 120.0,
          room_db: float = DEFAULT_ROOM_DB) -> np.ndarray:
    """Strict-grid kit. This is the only class a tempo gate should ever accept."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    left, right = np.zeros(n), np.zeros(n)

    kick_t = np.arange(int(0.35 * sr)) / sr
    kick = np.sin(2 * np.pi * (55 + 90 * np.exp(-kick_t * 40)) * kick_t) * np.exp(-kick_t * 12)

    snare_t = np.arange(int(0.25 * sr)) / sr
    noise = rng.normal(0, 1, len(snare_t))
    w0 = 2 * np.pi * 1800 / sr
    alpha = np.sin(w0) / (2 * 1.2)
    snare = lfilter([alpha, 0, -alpha], [1 + alpha, -2 * np.cos(w0), 1 - alpha], noise)
    snare = snare * np.exp(-snare_t * 22) + np.sin(2 * np.pi * 190 * snare_t) * np.exp(
        -snare_t * 26
    ) * 0.5

    hat_t = np.arange(int(0.06 * sr)) / sr
    hat = rng.normal(0, 1, len(hat_t)) * np.exp(-hat_t * 90)
    w0 = 2 * np.pi * 9000 / sr
    alpha = np.sin(w0) / (2 * 0.8)
    hat = lfilter([1, -2 * np.cos(w0), 1], [1 + alpha, -2 * np.cos(w0), 1 - alpha], hat)

    beat = 60.0 / bpm
    for index in range(int(seconds / (beat / 2))):
        start = int(index * beat / 2 * sr)
        if index % 4 == 0:
            _place(left, kick, start, 0.9)
            _place(right, kick, start, 0.9)
        if index % 4 == 2:
            _place(left, snare, start, 0.75)
            _place(right, snare, start, 0.72)
        _place(left, hat, start, 0.3)
        _place(right, hat, start, 0.34)

    return _with_room_tone(_normalise(_stereo(left, right)), room_db, seed + 500)


def full_mix(seconds: float = 12.0, sr: int = SR, seed: int = 19,
             room_db: float = DEFAULT_ROOM_DB) -> np.ndarray:
    """Drums plus bass plus guitar: percussive, steady, full-band."""
    kit = drums(seconds, sr, seed=seed, room_db=-np.inf)
    guitar = solo_guitar(seconds, sr, seed=seed + 1, room_db=-np.inf)

    n = int(seconds * sr)
    bass = np.zeros(n)
    for index, start in enumerate(_note_grid(seconds, 120.0, sr)):
        frequency = [55.0, 65.41, 73.42, 49.0][index % 4]
        note = _pluck(frequency, 0.6, sr, damping=0.999)
        _place(bass, note, start, 0.8)
    bass_stereo = _stereo(bass, bass)

    mixed = _normalise(0.55 * kit + 0.30 * bass_stereo + 0.35 * guitar)
    return _with_room_tone(mixed, room_db, seed + 500)


def clean_full_band(seconds: float = 12.0, sr: int = SR, seed: int = 23,
                    room_db: float = -90.0) -> np.ndarray:
    """The null-test subject: already good, with a floor low enough that there
    is genuinely nothing worth removing."""
    return full_mix(seconds, sr, seed=seed, room_db=room_db)


# --- Degradations (the input side of "damaged material") -----------------


def add_hiss(audio: np.ndarray, level_db: float = -48.0, seed: int = 29,
             sr: int = SR, high_pass_hz: float | None = None) -> np.ndarray:
    """Hiss at a known level relative to the program.

    `high_pass_hz` restricts the noise to the band above it. Tape hiss and
    converter noise really do sit mostly up there, and it matters for testing:
    noise inside 100Hz-10kHz cannot be removed without moving bands that the
    verification gate protects, so the two cases behave differently by design.
    """
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    noise = rng.normal(0, 1, audio.shape)
    if high_pass_hz:
        sos = butter(8, high_pass_hz / (sr / 2), btype="high", output="sos")
        noise = sosfilt(sos, noise, axis=0)
        noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-12)
    return (audio + noise * rms * 10 ** (level_db / 20)).astype(np.float32)


def band_limit(audio: np.ndarray, sr: int = SR, cutoff_hz: float = 8000.0) -> np.ndarray:
    """Steep low-pass: the spectral cliff a bandwidth gate is supposed to find."""
    from scipy.signal import cheby1, sosfiltfilt

    sos = cheby1(10, 0.5, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def codec_degrade(audio: np.ndarray, sr: int = SR, cutoff_hz: float = 11000.0,
                  seed: int = 31) -> np.ndarray:
    """Low bitrate: a cliff, plus spectral holes and quantisation in what is left."""
    import librosa

    rng = np.random.default_rng(seed)
    limited = band_limit(audio, sr, cutoff_hz)
    channels = [limited] if limited.ndim == 1 else [limited[:, i] for i in range(limited.shape[1])]

    out = []
    for channel in channels:
        spec = librosa.stft(channel.astype(np.float32), n_fft=1024, hop_length=256)
        magnitude = np.abs(spec)
        # Drop the quietest bins per frame, the way a coder discards masked ones.
        floor = np.percentile(magnitude, 45, axis=0, keepdims=True)
        keep = magnitude >= floor
        spec = spec * (keep + 0.05 * ~keep)
        jitter = rng.normal(0, 1, spec.shape) + 1j * rng.normal(0, 1, spec.shape)
        spec += jitter * magnitude * 0.02
        out.append(librosa.istft(spec, n_fft=1024, hop_length=256, length=len(channel)))

    result = out[0] if len(out) == 1 else np.stack(out, axis=1)
    return result.astype(np.float32)


# --- Damage that reproduces the reported V2/V3 defects --------------------


def damage_warp(audio: np.ndarray, sr: int = SR, start_s: float = 2.0, end_s: float = 8.0,
                drift_ms: float = -375.0) -> np.ndarray:
    """Accumulate `drift_ms` of timing error between `start_s` and `end_s`.

    Reproduces defect #1: the region inside the beat grid is resampled while the
    head is copied and the tail merely inherits the accumulated shift, so the
    output is shorter and everything after the warp sits early.
    """
    n = audio.shape[0]
    start, end = int(start_s * sr), int(end_s * sr)
    start, end = max(0, start), min(n, end)
    if end <= start:
        return audio.copy()

    drift = int(round(drift_ms / 1000.0 * sr))
    span = end - start
    target = span + drift
    if target < 8:
        return audio.copy()

    source_index = np.linspace(0, span - 1, target)
    if audio.ndim == 1:
        middle = np.interp(source_index, np.arange(span), audio[start:end])
        return np.concatenate([audio[:start], middle, audio[end:]]).astype(np.float32)
    columns = [
        np.interp(source_index, np.arange(span), audio[start:end, c])
        for c in range(audio.shape[1])
    ]
    middle = np.stack(columns, axis=1)
    return np.concatenate([audio[:start], middle, audio[end:]], axis=0).astype(np.float32)


def frame_level_db(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-frame level in dBFS, on the same footing as `damage_expander`'s knee.

    Tests use this to place a knee inside the material's own level range. The
    plan quotes an absolute -75dBFS knee, which is where the reference guitar's
    noise floor happens to sit; synthetic fixtures have their own floor, and
    hard-coding -75 would just mean the expander never engages.
    """
    import librosa

    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    rms = librosa.feature.rms(y=mono.astype(np.float32), frame_length=2048, hop_length=512)[0]
    return 20 * np.log10(rms + 1e-12)


def damage_expander(audio: np.ndarray, sr: int = SR, knee_db: float = -75.0,
                    max_attenuation_db: float = 15.75) -> np.ndarray:
    """Downward expansion below `knee_db`, ramping to `max_attenuation_db`.

    Reproduces defect #2: the measured denoise gain curve, which reaches about
    -15.75dB on the quietest frames and roughly -1.5dB on loud material.
    """
    import librosa

    channels = [audio] if audio.ndim == 1 else [audio[:, i] for i in range(audio.shape[1])]
    out = []
    for channel in channels:
        spec = librosa.stft(channel.astype(np.float32), n_fft=2048, hop_length=512)
        # Frame level comes from the time domain, not from summed STFT bins:
        # only the former is in honest dBFS, which is what an absolute knee
        # threshold has to be compared against.
        rms = librosa.feature.rms(y=channel.astype(np.float32), frame_length=2048,
                                  hop_length=512, center=True)[0]
        frame_db = 20 * np.log10(rms[: spec.shape[1]] + 1e-12)
        # Linear in dB from the knee down to 20dB below it.
        below = np.clip((knee_db - frame_db) / 20.0, 0.0, 1.0)
        gain_db = -1.5 - (max_attenuation_db - 1.5) * below
        spec = spec * (10 ** (gain_db / 20))[None, :]
        out.append(librosa.istft(spec, n_fft=2048, hop_length=512, length=len(channel)))
    result = out[0] if len(out) == 1 else np.stack(out, axis=1)
    return result.astype(np.float32)


def damage_presence(audio: np.ndarray, sr: int = SR, low_hz: float = 5000.0,
                    high_hz: float = 10000.0, gain_db: float = -3.0) -> np.ndarray:
    """Flat cut across a band. Reproduces defect #3, the 5-10kHz presence loss."""
    import librosa

    scale = 10 ** (gain_db / 20)
    channels = [audio] if audio.ndim == 1 else [audio[:, i] for i in range(audio.shape[1])]
    out = []
    for channel in channels:
        spec = librosa.stft(channel.astype(np.float32), n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band = (freqs >= low_hz) & (freqs < high_hz)
        spec[band] *= scale
        out.append(librosa.istft(spec, n_fft=2048, hop_length=512, length=len(channel)))
    result = out[0] if len(out) == 1 else np.stack(out, axis=1)
    return result.astype(np.float32)


def damage_synthetic_hf(audio: np.ndarray, sr: int = SR, level_db: float = -70.0,
                        period_s: float = 9.0, seed: int = 37) -> np.ndarray:
    """Add incoherent high band whose level pulses with the chunk stride.

    Reproduces defect #4: content above 10kHz with near-zero coherence to the
    source, ~70dB down, swinging on a ~9 second cycle.
    """
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    n = audio.shape[0]
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    noise = rng.normal(0, 1, audio.shape)
    sos = butter(8, 10000 / (sr / 2), btype="high", output="sos")
    hf = sosfilt(sos, noise, axis=0)

    time = np.arange(n) / sr
    modulation = 10 ** ((3.0 * np.sin(2 * np.pi * time / period_s)) / 20)
    if audio.ndim == 2:
        modulation = modulation[:, None]
    return (audio + hf * rms * 10 ** (level_db / 20) * modulation).astype(np.float32)


def damage_collapse_stereo(audio: np.ndarray) -> np.ndarray:
    """Fold to mono, keeping two channels. Reproduces the side-information loss."""
    if audio.ndim == 1:
        return audio.copy()
    mid = audio.mean(axis=1)
    return np.stack([mid, mid], axis=1).astype(np.float32)


CLEAN_CLASSES = {
    "clean_full_band": clean_full_band,
    "solo_guitar": solo_guitar,
    "solo_piano": solo_piano,
    "vocal": vocal,
    "full_mix": full_mix,
    "drums": drums,
}
