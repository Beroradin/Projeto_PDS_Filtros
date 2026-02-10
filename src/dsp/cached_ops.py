"""
Cached wrappers for heavy DSP operations.

Uso: importe estas funções no lugar das chamadas diretas ao SignalProcessor.
O @st.cache_data garante que, para os mesmos argumentos, o resultado é
retornado do cache em vez de ser recalculado.

IMPORTANTE: os argumentos precisam ser hashable. Arrays numpy são
automaticamente suportados pelo Streamlit, mas objetos como file uploaders
não são — por isso load_audio recebe bytes, não o file object.
"""

import numpy as np
import numpy.typing as npt
import streamlit as st
from typing import Tuple, Optional, Union

from src.dsp.processor import SignalProcessor, LMSFilter


# ============================================================
# Geração de sinais (cache por num_samples + seed)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8)
def cached_generate_white_noise(
    num_samples: int, seed: int = 42
) -> npt.NDArray[np.float64]:
    return SignalProcessor.generate_white_noise(num_samples, seed)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_generate_impulsive_noise(
    num_samples: int, seed: int = 42, prob: float = 0.01
) -> npt.NDArray[np.float64]:
    return SignalProcessor.generate_impulsive_noise(num_samples, seed, prob)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_generate_chirp_noise(
    num_samples: int,
    fs: float,
    f_start: float = 20.0,
    f_end: float = 20000.0,
    method: str = 'linear'
) -> npt.NDArray[np.float64]:
    return SignalProcessor.generate_chirp_noise(num_samples, fs, f_start, f_end, method)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_generate_high_freq_noise(
    num_samples: int,
    fs: float,
    freq: float = 16000.0,
    seed: int = 42
) -> npt.NDArray[np.float64]:
    return SignalProcessor.generate_high_freq_noise(num_samples, fs, freq, seed)


# ============================================================
# Mixagem
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8)
def cached_mix_signals(
    clean_signal: npt.NDArray[np.float64],
    noise: npt.NDArray[np.float64],
    target_snr_db: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return SignalProcessor.mix_signals(clean_signal, noise, target_snr_db)


# ============================================================
# Análise espectral (as mais pesadas em CPU)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=16)
def cached_compute_fft(
    signal: npt.NDArray[np.float64],
    fs: float,
    window_type: str = 'hann'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return SignalProcessor.compute_fft(signal, fs, window_type)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_compute_stft(
    signal: npt.NDArray[np.float64],
    fs: float,
    window: str = 'hann',
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    return SignalProcessor.compute_stft(signal, fs, window, nperseg, noverlap)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_compute_cwt(
    signal: npt.NDArray[np.float64],
    fs: float,
    wavelet: str = 'cmor1.5-1.0',
    num_scales: int = 127  # em vez de passar o array, passa o tamanho
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    scales = np.arange(1, num_scales + 1)
    return SignalProcessor.compute_cwt(signal, fs, wavelet, scales)


# ============================================================
# Design de filtros (leve, mas cachear evita redesign repetido)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=16)
def cached_design_fir(
    numtaps: int,
    cutoff_hz: float,
    fs: float,
    window: str = 'hann',
    pass_zero: bool = True,
    method: str = 'window'
) -> npt.NDArray[np.float64]:
    return SignalProcessor.design_fir(numtaps, cutoff_hz, fs, window, pass_zero, method)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_design_iir(
    order: int,
    cutoff_hz: float,
    fs: float,
    btype: str = 'lowpass',
    ftype: str = 'butter',
    rp: float = 1.0,
    rs: float = 40.0
) -> npt.NDArray[np.float64]:
    return SignalProcessor.design_iir(order, cutoff_hz, fs, btype, ftype, rp, rs)


# ============================================================
# Aplicação de filtro (a operação mais pesada em runtime)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8)
def cached_apply_filter(
    signal: npt.NDArray[np.float64],
    _coeffs_key: str,  # chave única para identificar o filtro no cache
    coeffs: Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray],
    zero_phase: bool = False
) -> npt.NDArray[np.float64]:
    """
    Wrapper cacheado para apply_filter.

    _coeffs_key serve como identificador hashable para os coeficientes.
    Gere com: f"{filter_type}_{order}_{cutoff}_{fs}"
    """
    return SignalProcessor.apply_filter(signal, coeffs, zero_phase)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analyze_filter(
    _coeffs_key: str,
    coeffs: Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray],
    fs: float
) -> dict:
    return SignalProcessor.analyze_filter(coeffs, fs)


# ============================================================
# Métricas e autocorrelação
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8)
def cached_compute_snr(
    clean: npt.NDArray[np.float64],
    noisy: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    return SignalProcessor.compute_snr(clean, noisy)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_compute_autocorrelation(
    signal: npt.NDArray[np.float64],
    max_lag: int = 1000
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return SignalProcessor.compute_autocorrelation(signal, max_lag)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_compute_histogram(
    signal: npt.NDArray[np.float64],
    bins: int = 50,
    density: bool = True
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return SignalProcessor.compute_histogram(signal, bins, density)


# ============================================================
# Áudio — cache por conteúdo do arquivo (bytes)
# ============================================================

@st.cache_data(show_spinner="Carregando áudio...")
def cached_load_audio(
    file_bytes: bytes
) -> Tuple[int, npt.NDArray[np.float64]]:
    """
    Cacheia o load de áudio pelo conteúdo (bytes) do arquivo.

    Uso:
        uploaded = st.file_uploader(...)
        if uploaded:
            sr, data = cached_load_audio(uploaded.getvalue())
    """
    import io
    return SignalProcessor.load_audio(io.BytesIO(file_bytes))


# ============================================================
# LMS adaptativo
# ============================================================

@st.cache_data(show_spinner=False, max_entries=4)
def cached_lms_filter(
    x: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    num_taps: int,
    mu: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lms = LMSFilter(num_taps, mu)
    return lms.run(x, d)