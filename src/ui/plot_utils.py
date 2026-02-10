"""
Utilitários de visualização com downsampling inteligente.

O Plotly serializa TODOS os pontos como JSON → memória explode com sinais longos.
A tela tem ~1920px de largura, então mais de ~5000 pontos é desperdício visual.

Estratégias implementadas:
  1. decimation  — pega 1 a cada N pontos (rápido, bom para sinais suaves)
  2. minmax      — para cada "bin" de pontos, guarda min e max
                   (preserva picos e envoltória, ideal para áudio)
  3. lttb        — Largest-Triangle-Three-Buckets (melhor qualidade visual,
                   um pouco mais lento, mas ainda O(n))

Uso no layout:
    from src.ui.plot_utils import downsample, downsample_xy, create_lightweight_figure

    # Sinal no domínio do tempo
    t_ds, y_ds = downsample_xy(t, signal, max_points=5000, method='minmax')
    fig.add_trace(go.Scatter(x=t_ds, y=y_ds, mode='lines'))

    # Espectro (frequência x magnitude)
    f_ds, mag_ds = downsample_xy(freqs, mag_db, max_points=3000)
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Literal


# ============================================================
# Funções de downsampling
# ============================================================

def downsample(
    signal: npt.NDArray[np.float64],
    max_points: int = 5000,
    method: Literal['decimation', 'minmax', 'lttb'] = 'minmax'
) -> npt.NDArray[np.float64]:
    """
    Reduz o número de pontos de um sinal 1D para visualização.

    Args:
        signal: sinal original
        max_points: máximo de pontos no resultado
        method: estratégia de downsampling

    Returns:
        sinal reduzido (apenas para plotagem, NÃO para cálculos!)
    """
    n = len(signal)
    if n <= max_points:
        return signal

    if method == 'decimation':
        step = n // max_points
        return signal[::step]

    elif method == 'minmax':
        return _downsample_minmax(signal, max_points)

    elif method == 'lttb':
        return _downsample_lttb_y(signal, max_points)

    return signal[:: (n // max_points)]


def downsample_xy(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    max_points: int = 5000,
    method: Literal['decimation', 'minmax', 'lttb'] = 'minmax'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Reduz (x, y) juntos mantendo a correspondência entre eixos.

    Ideal para espectros (freq, magnitude) ou sinais com eixo de tempo explícito.
    """
    n = len(y)
    if n <= max_points:
        return x[:n], y[:n]

    if method == 'decimation':
        step = n // max_points
        return x[:n:step], y[::step]

    elif method == 'minmax':
        return _downsample_minmax_xy(x, y, max_points)

    elif method == 'lttb':
        return _downsample_lttb_xy(x, y, max_points)

    step = n // max_points
    return x[:n:step], y[::step]


# ============================================================
# Min-Max (preserva envoltória — ótimo para áudio)
# ============================================================

def _downsample_minmax(
    signal: npt.NDArray[np.float64],
    max_points: int
) -> npt.NDArray[np.float64]:
    """
    Divide o sinal em buckets e guarda min e max de cada um.
    Resultado tem ~2 * (max_points // 2) pontos alternando min/max.
    """
    n = len(signal)
    n_buckets = max_points // 2  # cada bucket gera 2 pontos
    bucket_size = n / n_buckets

    result = np.empty(n_buckets * 2, dtype=signal.dtype)

    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)
        chunk = signal[start:end]
        if len(chunk) == 0:
            result[2 * i] = result[2 * i + 1] = signal[-1]
        else:
            result[2 * i] = np.min(chunk)
            result[2 * i + 1] = np.max(chunk)

    return result


def _downsample_minmax_xy(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    max_points: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Min-max preservando eixo x (usa posição real do min e do max)."""
    n = len(y)
    n_buckets = max_points // 2
    bucket_size = n / n_buckets

    x_out = np.empty(n_buckets * 2, dtype=x.dtype)
    y_out = np.empty(n_buckets * 2, dtype=y.dtype)

    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)
        chunk = y[start:end]

        if len(chunk) == 0:
            x_out[2 * i] = x_out[2 * i + 1] = x[-1]
            y_out[2 * i] = y_out[2 * i + 1] = y[-1]
        else:
            idx_min = start + np.argmin(chunk)
            idx_max = start + np.argmax(chunk)

            # Ordena cronologicamente
            if idx_min <= idx_max:
                x_out[2 * i], y_out[2 * i] = x[idx_min], y[idx_min]
                x_out[2 * i + 1], y_out[2 * i + 1] = x[idx_max], y[idx_max]
            else:
                x_out[2 * i], y_out[2 * i] = x[idx_max], y[idx_max]
                x_out[2 * i + 1], y_out[2 * i + 1] = x[idx_min], y[idx_min]

    return x_out, y_out


# ============================================================
# LTTB — Largest Triangle Three Buckets
# (melhor qualidade visual, O(n))
# ============================================================

def _downsample_lttb_y(
    signal: npt.NDArray[np.float64],
    max_points: int
) -> npt.NDArray[np.float64]:
    """LTTB com eixo x implícito (índices)."""
    x = np.arange(len(signal), dtype=np.float64)
    x_out, y_out = _downsample_lttb_xy(x, signal, max_points)
    return y_out


def _downsample_lttb_xy(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    max_points: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Largest-Triangle-Three-Buckets downsampling.
    Ref: Sveinn Steinarsson, 2013
    """
    n = len(y)
    if n <= max_points:
        return x.copy(), y.copy()

    # Sempre mantém primeiro e último ponto
    x_out = np.empty(max_points, dtype=x.dtype)
    y_out = np.empty(max_points, dtype=y.dtype)

    x_out[0] = x[0]
    y_out[0] = y[0]
    x_out[max_points - 1] = x[n - 1]
    y_out[max_points - 1] = y[n - 1]

    bucket_size = (n - 2) / (max_points - 2)

    a_x = x[0]
    a_y = y[0]

    for i in range(1, max_points - 1):
        # Bucket atual
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1
        bucket_end = min(bucket_end, n)

        # Próximo bucket — média para o "ponto C"
        next_start = int(i * bucket_size) + 1
        next_end = int((i + 1) * bucket_size) + 1
        next_end = min(next_end, n)

        avg_x = np.mean(x[next_start:next_end]) if next_start < next_end else x[-1]
        avg_y = np.mean(y[next_start:next_end]) if next_start < next_end else y[-1]

        # Escolhe o ponto do bucket atual que forma o maior triângulo
        best_idx = bucket_start
        max_area = -1.0

        for j in range(bucket_start, bucket_end):
            # Área do triângulo (A, ponto_j, C)
            area = abs(
                (x[j] - a_x) * (avg_y - a_y)
                - (avg_x - a_x) * (y[j] - a_y)
            )
            if area > max_area:
                max_area = area
                best_idx = j

        x_out[i] = x[best_idx]
        y_out[i] = y[best_idx]

        a_x = x[best_idx]
        a_y = y[best_idx]

    return x_out, y_out


# ============================================================
# Helper para criar figuras Plotly leves
# ============================================================

def create_lightweight_figure(
    layout_kwargs: Optional[dict] = None
) -> "plotly.graph_objects.Figure":
    """
    Cria uma Figure do Plotly com configurações otimizadas para
    reduzir uso de memória no servidor Streamlit.
    """
    import plotly.graph_objects as go

    default_layout = dict(
        template="plotly_dark",
        margin=dict(l=50, r=20, t=40, b=40),
        # Desabilita animações que consomem memória
        transition=dict(duration=0),
        # WebGL para renderização mais rápida com muitos pontos
        # (aplicar no trace com go.Scattergl em vez de go.Scatter)
    )

    if layout_kwargs:
        default_layout.update(layout_kwargs)

    fig = go.Figure(layout=default_layout)
    return fig


def plot_signal_lightweight(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    name: str = "",
    max_points: int = 5000,
    method: str = 'minmax',
    use_webgl: bool = True,
    **trace_kwargs
) -> "plotly.graph_objects.Scatter":
    """
    Retorna um trace Plotly já com downsampling aplicado.

    Use go.Scattergl (WebGL) por padrão para performance no browser.

    Uso:
        fig = create_lightweight_figure({"title": "Sinal"})
        fig.add_trace(plot_signal_lightweight(t, signal, "Clean"))
        st.plotly_chart(fig, use_container_width=True)
    """
    import plotly.graph_objects as go

    x_ds, y_ds = downsample_xy(x, y, max_points=max_points, method=method)

    ScatterClass = go.Scattergl if use_webgl else go.Scatter

    return ScatterClass(
        x=x_ds,
        y=y_ds,
        name=name,
        mode='lines',
        **trace_kwargs
    )