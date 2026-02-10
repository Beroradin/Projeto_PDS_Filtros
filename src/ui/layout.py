import streamlit as st
import numpy as np
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
from PIL import Image

# ============================================================
# IMPORTS OTIMIZADOS: cached_ops + plot_utils
# ============================================================
from src.dsp.processor import SignalProcessor, LMSFilter
from src.dsp.cached_ops import (
    cached_generate_white_noise,
    cached_generate_impulsive_noise,
    cached_generate_chirp_noise,
    cached_generate_high_freq_noise,
    cached_mix_signals,
    cached_compute_fft,
    cached_compute_stft,
    cached_compute_cwt,
    cached_design_fir,
    cached_design_iir,
    cached_apply_filter,
    cached_analyze_filter,
    cached_compute_snr,
    cached_compute_autocorrelation,
    cached_compute_histogram,
    cached_load_audio,
    cached_lms_filter,
)
from src.ui.plot_utils import (
    downsample_xy,
    downsample,
    create_lightweight_figure,
    plot_signal_lightweight,
)

# ============================================================
# Constantes de performance
# ============================================================
MAX_PLOT_POINTS = 5000        # Pontos máx por trace no Plotly
MAX_SPEC_TIME_BINS = 600      # Bins temporais máx para espectrograma
MAX_CWT_PTS = 4000            # Amostras máx para CWT
MAX_DURATION_SECONDS = 30     # Duração máx de áudio (trunca se maior)


def init_session_state():
    """Initializes Streamlit session state with proper defaults."""
    defaults = {
        'signal': np.zeros(1000),
        'sampling_rate': 1000.0,
        'signal_source': 'none',
        'last_snr': 10.0,
        'last_noise_type': 'Branco',
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_audio(signal, fs, label=""):
    """
    Safe wrapper for st.audio
    Normalizes signal to [-1, 1] range to avoid clipping distortion.
    """
    if signal is None or len(signal) == 0:
        return
    
    if not np.all(np.isfinite(signal)):
        st.warning(f"⚠️ {label} Áudio contém valores inválidos (NaN/Inf). Impossível reproduzir.")
        return
    
    max_val = np.max(np.abs(signal))
    if max_val > 1.0:
        sig_norm = signal / (max_val + 1e-6)
    else:
        sig_norm = signal
    
    sig_norm = sig_norm.astype(np.float32)
    st.audio(sig_norm, sample_rate=int(fs))


def create_time_domain_plot(signal, fs, title, color, show_envelope=False):
    """Creates a time domain plot with minmax downsampling and optional envelope."""
    N = len(signal)
    t_full = np.linspace(0, N / fs, N)
    
    # Downsampling minmax preserva picos
    t_ds, s_ds = downsample_xy(t_full, signal, max_points=MAX_PLOT_POINTS, method='minmax')
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t_ds, y=s_ds,
        mode='lines',
        name='Sinal',
        line=dict(color=color, width=1)
    ))
    
    if show_envelope and len(s_ds) > 100:
        window_size = max(1, len(s_ds) // 100)
        envelope_pos = []
        envelope_neg = []
        t_env = []
        
        for i in range(0, len(s_ds), window_size):
            chunk = s_ds[i:i+window_size]
            if len(chunk) > 0:
                envelope_pos.append(np.max(chunk))
                envelope_neg.append(np.min(chunk))
                t_env.append(t_ds[i])
        
        fig.add_trace(go.Scattergl(
            x=t_env, y=envelope_pos,
            mode='lines',
            name='Envelope',
            line=dict(color=color, width=2, dash='dash'),
            opacity=0.6
        ))
        fig.add_trace(go.Scattergl(
            x=t_env, y=envelope_neg,
            mode='lines',
            name='Envelope',
            line=dict(color=color, width=2, dash='dash'),
            opacity=0.6,
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified'
    )
    
    return fig


def fig_to_bytes(fig, format='png'):
    """Convert plotly figure to bytes for download"""
    img_bytes = fig.to_image(format=format, width=1200, height=800, scale=2)
    return img_bytes


def _make_coeffs_key(arch, **kwargs):
    """Gera chave hashable única para identificar um filtro no cache."""
    parts = [arch]
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")
    return "|".join(parts)


def render_layout():
    """Renders the main application layout."""
    st.set_page_config(
        page_title="DSP Analytics Suite",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better aesthetics
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00CC96 0%, #AB63FA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #00CC96;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">📡 Suíte de Análise de Sinais Digitais (DSP)</div>', unsafe_allow_html=True)
    st.markdown("##### Processamento Digital de Sinais com Filtros FIR, IIR e Adaptativos (LMS)")
    st.markdown("---")
    
    init_session_state()

    # --- SIDEBAR: Signal Source ---
    with st.sidebar:
        st.header("🎚️ 1. Fonte do Sinal")
        
        source_type = st.radio(
            "Escolha a fonte de dados:",
            ["Gerar Sintético", "Upload Arquivo", "Exemplo Padrão"],
            index=0,
            help="Selecione como deseja carregar o sinal"
        )
        
        signal_changed = False
        if st.session_state.get('last_source_type') != source_type:
            signal_changed = True
            st.session_state.last_source_type = source_type

        if source_type == "Upload Arquivo":
            uploaded_file = st.file_uploader(
                "Carregar arquivo de áudio",
                type=["wav", "flac", "mp3"],
                help="Formatos suportados: WAV, FLAC, MP3"
            )
            if uploaded_file is not None:
                try:
                    # CACHE: usa bytes para hashabilidade
                    fs, data = cached_load_audio(uploaded_file.getvalue())
                    
                    # Trunca se muito longo (economia de RAM)
                    max_samples = int(MAX_DURATION_SECONDS * fs)
                    if len(data) > max_samples:
                        data = data[:max_samples]
                        st.warning(f"⚠️ Áudio truncado para {MAX_DURATION_SECONDS}s para economizar memória.")
                    
                    st.session_state.signal = data
                    st.session_state.sampling_rate = fs
                    st.session_state.signal_source = f"upload_{uploaded_file.name}"
                    
                    st.session_state.pop('mixed_signal', None)
                    st.session_state.pop('filtered_signal', None)
                    st.session_state.pop('noise_ref', None)
                    
                    st.success(f"✅ Carregado: {len(data):,} amostras @ {fs:,.0f} Hz")
                    st.info(f"📊 Duração: {len(data)/fs:.2f} segundos")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
                
        elif source_type == "Exemplo Padrão":
            st.info("📁 Tentará carregar 'audio_10s.wav' do diretório local")
            
            if st.button("🔄 Carregar Arquivo de Exemplo", use_container_width=True):
                try:
                    possible_paths = [
                        "audio_10s.wav",
                        "./audio_10s.wav",
                        "../audio_10s.wav",
                        "/home/claude/audio_10s.wav"
                    ]
                    
                    loaded = False
                    for path in possible_paths:
                        if os.path.exists(path):
                            # Lê como bytes para usar o cache
                            with open(path, 'rb') as f:
                                file_bytes = f.read()
                            fs, data = cached_load_audio(file_bytes)
                            
                            st.session_state.signal = data
                            st.session_state.sampling_rate = fs
                            st.session_state.signal_source = "example_audio"
                            
                            st.session_state.pop('mixed_signal', None)
                            st.session_state.pop('filtered_signal', None)
                            st.session_state.pop('noise_ref', None)
                            
                            st.success(f"✅ Carregado: {len(data):,} amostras @ {fs:,.0f} Hz")
                            st.info(f"📊 Duração: {len(data)/fs:.2f} segundos")
                            loaded = True
                            break
                    
                    if not loaded:
                        st.error("❌ Arquivo 'audio_10s.wav' não encontrado em nenhum caminho conhecido")
                        st.info("💡 Coloque o arquivo no mesmo diretório do script")
                        
                except Exception as e:
                    st.error(f"❌ Erro: {e}")

        else:  # Synthetic Generation
            st.markdown("##### Parâmetros de Geração")
            
            gen_type = st.selectbox(
                "Tipo de Sinal",
                ["Ruído Branco", "Ruído Impulsivo", "Chirp", "Alta Frequência (16kHz)"],
                help="Escolha o tipo de sinal sintético"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                N = st.number_input(
                    "Amostras (N)",
                    min_value=100,
                    max_value=200000,
                    value=10000,
                    step=1000,
                    help="Número de amostras a gerar"
                )
            with col2:
                fs_gen = st.number_input(
                    "Taxa (Hz)",
                    min_value=1.0,
                    max_value=48000.0,
                    value=8000.0,
                    step=1000.0,
                    help="Taxa de amostragem em Hz"
                )
            
            f_start, f_end = 0.0, 500.0
            high_freq = 20000.0
            
            if gen_type == "Chirp":
                st.markdown("**Parâmetros do Chirp**")
                col1, col2 = st.columns(2)
                with col1:
                    f_start = st.number_input(
                        "F Início (Hz)",
                        value=0.0,
                        help="Frequência inicial"
                    )
                with col2:
                    f_end = st.number_input(
                        "F Fim (Hz)",
                        value=fs_gen/2 - 100,
                        help="Frequência final"
                    )
            elif gen_type == "Alta Frequência (16kHz)":
                st.markdown("**Parâmetros do Ruído de Alta Frequência**")
                high_freq = st.number_input(
                    "Frequência (Hz)",
                    min_value=1000.0,
                    max_value=fs_gen/2 - 100,
                    value=min(16000.0, fs_gen/2 - 100),
                    help="Frequência do ruído de alta frequência"
                )
            
            seed = st.number_input(
                "Seed (Reprodutibilidade)",
                value=42,
                help="Seed para geração aleatória reproduzível"
            )
            
            if st.button("⚡ Gerar Sinal", use_container_width=True):
                with st.spinner("Gerando sinal..."):
                    st.session_state.sampling_rate = fs_gen
                    
                    # CACHE: funções cacheadas
                    if gen_type == "Ruído Branco":
                        sig = cached_generate_white_noise(int(N), int(seed))
                    elif gen_type == "Ruído Impulsivo":
                        sig = cached_generate_impulsive_noise(int(N), int(seed))
                    elif gen_type == "Chirp":
                        sig = cached_generate_chirp_noise(int(N), fs_gen, f_start, f_end)
                    elif gen_type == "Alta Frequência (16kHz)":
                        sig = cached_generate_high_freq_noise(int(N), fs_gen, high_freq, int(seed))
                    
                    st.session_state.signal = sig
                    st.session_state.signal_source = f"synthetic_{gen_type}_{seed}"
                    
                    st.session_state.pop('mixed_signal', None)
                    st.session_state.pop('filtered_signal', None)
                    st.session_state.pop('noise_ref', None)
                    
                    st.success(f"✅ Sinal gerado: {len(sig):,} amostras")

        st.divider()
        
        # --- NOISE MIXING SECTION ---
        st.header("🔊 2. Adição de Ruído")
        
        col1, col2 = st.columns(2)
        with col1:
            target_snr = st.slider(
                "SNR Alvo (dB)",
                min_value=-10.0,
                max_value=40.0,
                value=10.0,
                step=1.0,
                help="Relação Sinal-Ruído desejada"
            )
        
        with col2:
            noise_type = st.selectbox(
                "Tipo de Ruído",
                ["Branco", "Impulsivo", "Chirp", "Alta Freq (16kHz)"],
                help="Escolha o tipo de ruído a adicionar"
            )
        
        noise_changed = (
            st.session_state.last_snr != target_snr or
            st.session_state.last_noise_type != noise_type
        )
        
        if st.button("🎯 Aplicar Ruído", use_container_width=True):
            with st.spinner("Misturando sinal com ruído..."):
                try:
                    N = len(st.session_state.signal)
                    fs = st.session_state.sampling_rate
                    
                    # CACHE: geração de ruído cacheada
                    if noise_type == "Branco":
                        n = cached_generate_white_noise(N)
                    elif noise_type == "Impulsivo":
                        n = cached_generate_impulsive_noise(N)
                    elif noise_type == "Chirp":
                        n = cached_generate_chirp_noise(N, fs, 0, fs/2 - 100)
                    else:  # Alta Freq
                        high_f = min(16000.0, fs/2 - 100)
                        n = cached_generate_high_freq_noise(N, fs, high_f)
                    
                    # CACHE: mixagem cacheada
                    mixed, scaled_noise = cached_mix_signals(
                        st.session_state.signal, n, target_snr
                    )
                    
                    st.session_state.mixed_signal = mixed
                    st.session_state.noise_ref = scaled_noise
                    st.session_state.last_snr = target_snr
                    st.session_state.last_noise_type = noise_type
                    
                    st.session_state.pop('filtered_signal', None)
                    
                    st.success(f"✅ Ruído aplicado! SNR = {target_snr} dB")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao aplicar ruído: {e}")
        
        # Show current state
        st.divider()
        st.markdown("##### 📊 Estado Atual")
        
        state_info = f"""
        - **Sinal Original**: {len(st.session_state.signal):,} amostras
        - **Taxa de Amostragem**: {st.session_state.sampling_rate:,.0f} Hz
        - **Duração**: {len(st.session_state.signal)/st.session_state.sampling_rate:.2f}s
        """
        
        if 'mixed_signal' in st.session_state:
            state_info += f"\n- **Ruído**: {noise_type} (SNR: {target_snr} dB)"
        
        if 'filtered_signal' in st.session_state:
            state_info += f"\n- **Filtro**: Aplicado ✅"
        
        st.info(state_info)

    # --- MAIN CONTENT TABS ---
    tab_teoria, tab_eda, tab_spec, tab_stats, tab_filter, tab_lms = st.tabs([
        "📚 Teoria DSP",
        "📊 EDA & Visualização Temporal",
        "🌊 Análise Espectral (FFT/STFT/CWT)",
        "📈 Estatísticas & Distribuições",
        "🛠️ Filtros Digitais (FIR/IIR)",
        "🧠 Filtros Adaptativos (LMS)"
    ])

    # =================================================================
    # TAB 0: TEORIA DSP
    # =================================================================
    with tab_teoria:
        st.header("📚 Fundamentos de Processamento Digital de Sinais")
        
        st.markdown("""
        Esta aba apresenta os conceitos teóricos fundamentais utilizados nesta ferramenta de análise DSP.
        """)
        
        # Filtros FIR
        with st.expander("🔷 **Filtros FIR (Finite Impulse Response)**", expanded=True):
            st.markdown("""
            ### O que são Filtros FIR?
            
            Filtros FIR são filtros digitais cuja resposta ao impulso tem duração finita. Suas principais características são:
            
            **Vantagens:**
            - ✅ **Sempre estáveis** (não possuem polos fora da origem)
            - ✅ **Fase linear** (não distorcem a forma do sinal)
            - ✅ **Implementação simples** (apenas operações de multiplicação e soma)
            
            **Desvantagens:**
            - ❌ Requerem maior ordem para mesma especificação de filtro IIR
            - ❌ Maior custo computacional
            
            ### Equação de Diferenças
            
            A saída y[n] de um filtro FIR é dada por:
            
            ```
            y[n] = b₀x[n] + b₁x[n-1] + b₂x[n-2] + ... + bₘx[n-M]
            ```
            
            Onde:
            - `x[n]` é o sinal de entrada
            - `bₖ` são os coeficientes do filtro (também chamados de "taps")
            - `M` é a ordem do filtro
            
            ### Métodos de Projeto
            
            **1. Janelamento (Window Method)**
            - Aplica-se uma janela (Hann, Hamming, Blackman, etc.) aos coeficientes do filtro ideal
            - Simples e rápido
            - Trade-off entre largura da banda de transição e ondulação
            
            **2. Parks-McClellan (Algoritmo de Remez)**
            - Otimização equiripple (ondulação uniforme)
            - Minimiza o erro máximo na banda passante e rejeitada
            - Filtros mais eficientes (menor ordem para mesma especificação)
            
            ### Como Aplicar um Filtro FIR
            
            ```python
            # Projeto do filtro
            b = SignalProcessor.design_fir(
                numtaps=51,           # Número de coeficientes
                cutoff_hz=1000,       # Frequência de corte
                fs=8000,              # Taxa de amostragem
                window='hann',        # Tipo de janela
                pass_zero=True,       # True=lowpass, False=highpass
                method='window'       # 'window' ou 'parks-mcclellan'
            )
            
            # Aplicação do filtro
            filtered_signal = SignalProcessor.apply_filter(
                signal=noisy_signal,
                coeffs=(b, np.array([1.0])),
                zero_phase=False      # False=causal, True=não-causal
            )
            ```
            """)
        
        # Filtros IIR
        with st.expander("🔶 **Filtros IIR (Infinite Impulse Response)**"):
            st.markdown("""
            ### O que são Filtros IIR?
            
            Filtros IIR possuem resposta ao impulso de duração infinita devido à realimentação (feedback). 
            
            **Vantagens:**
            - ✅ **Mais eficientes** (menor ordem para mesma especificação)
            - ✅ **Menor custo computacional**
            - ✅ Assemelham-se a filtros analógicos clássicos
            
            **Desvantagens:**
            - ❌ **Podem ser instáveis** (polos fora do círculo unitário)
            - ❌ **Fase não-linear** (distorção de fase)
            - ❌ Sensíveis a erros de quantização
            
            ### Equação de Diferenças
            
            ```
            y[n] = b₀x[n] + b₁x[n-1] + ... + bₘx[n-M] 
                   - a₁y[n-1] - a₂y[n-2] - ... - aₙy[n-N]
            ```
            
            A presença dos termos `aₖy[n-k]` introduz feedback, resultando em resposta infinita.
            
            ### Tipos de Filtros IIR
            
            **1. Butterworth**
            - Resposta maximamente plana na banda passante
            - Sem ondulações (ripple)
            - Transição suave entre bandas
            - Ideal para aplicações que requerem resposta plana
            
            **2. Chebyshev Tipo I**
            - Ondulação na banda passante
            - Transição mais abrupta que Butterworth
            - Melhor rejeição na banda de transição
            
            **3. Chebyshev Tipo II**
            - Ondulação na banda rejeitada
            - Resposta plana na banda passante
            - Transição menos abrupta que Tipo I
            
            **4. Elíptico (Cauer)**
            - Ondulação em **ambas** as bandas
            - Transição **mais abrupta** de todos
            - Melhor desempenho ordem/especificação
            - Mais complexo de projetar
            
            ### Estabilidade
            
            Um filtro IIR é estável se e somente se todos os seus **polos** estiverem dentro do círculo unitário:
            
            ```
            |pₖ| < 1  para todos os polos pₖ
            ```
            
            ### Como Aplicar um Filtro IIR
            
            ```python
            # Projeto do filtro (retorna SOS para estabilidade)
            sos = SignalProcessor.design_iir(
                order=4,              # Ordem do filtro
                cutoff_hz=1000,       # Frequência de corte
                fs=8000,              # Taxa de amostragem
                btype='lowpass',      # 'lowpass' ou 'highpass'
                ftype='butter',       # 'butter', 'cheby1', 'cheby2', 'ellip'
                rp=1.0,               # Ripple banda passante (dB)
                rs=40.0               # Atenuação banda rejeitada (dB)
            )
            
            # Aplicação do filtro
            filtered_signal = SignalProcessor.apply_filter(
                signal=noisy_signal,
                coeffs=sos,           # Second-Order Sections
                zero_phase=False
            )
            ```
            
            ### SOS (Second-Order Sections)
            
            Para evitar problemas numéricos, filtros IIR são implementados como cascata de seções de 2ª ordem:
            
            ```
            H(z) = H₁(z) × H₂(z) × ... × Hₙ(z)
            ```
            
            Isso melhora drasticamente a estabilidade numérica.
            """)
        
        # Filtragem com Fase Zero
        with st.expander("🔄 **Filtragem com Fase Zero (filtfilt)**"):
            st.markdown("""
            ### O que é Filtragem com Fase Zero?
            
            A filtragem com fase zero aplica o filtro **duas vezes**:
            1. Forward (normal): `y₁ = filter(x)`
            2. Backward (reverso): `y = filter(reverse(y₁))`
            
            **Resultado:**
            - ✅ **Zero distorção de fase** (sinal mantém forma temporal)
            - ✅ Magnitude do filtro é **elevada ao quadrado** (H² em vez de H)
            - ❌ **Não-causal** (não pode ser usado em tempo real)
            
            ### Quando usar?
            
            - ✅ Análise offline de sinais
            - ✅ Quando fase linear é crítica
            - ❌ **NÃO** usar para processamento em tempo real
            """)
        
        # Filtros Adaptativos LMS
        with st.expander("🧠 **Filtros Adaptativos - LMS (Least Mean Squares)**"):
            st.markdown("""
            ### O que é o Algoritmo LMS?
            
            O LMS é um algoritmo adaptativo que ajusta os coeficientes de um filtro para **minimizar o erro quadrático médio** 
            entre a saída do filtro e um sinal desejado.
            
            ### Aplicação: Cancelamento de Ruído
            
            **Configuração:**
            - `x[n]`: Sinal de referência (ruído correlacionado)
            - `d[n]`: Sinal desejado (sinal limpo + ruído)
            - `y[n]`: Saída do filtro (estimativa do ruído)
            - `e[n]`: Erro = d[n] - y[n] (sinal limpo estimado)
            
            ```
            Ruído (x) ──→ [Filtro LMS] ──→ Estimativa do Ruído (y)
                                  ↑
                                  │ Atualização de Pesos
                                  │
            Sinal Ruidoso (d) ─→ (−) ──→ Erro/Sinal Limpo (e)
                                  ↑
                            Estimativa (y)
            ```
            
            ### Equações do LMS
            
            **1. Saída do filtro:**
            ```
            y[n] = wₙᵀ × x[n] = Σ wᵢ[n] × x[n-i]
            ```
            
            **2. Erro:**
            ```
            e[n] = d[n] - y[n]
            ```
            
            **3. Atualização dos pesos:**
            ```
            w[n+1] = w[n] + 2μ × e[n] × x[n]
            ```
            
            Onde:
            - `w`: vetor de pesos do filtro
            - `μ`: taxa de aprendizado (learning rate)
            - `x[n]`: vetor regressor [x[n], x[n-1], ..., x[n-M+1]]
            
            ### Parâmetros Importantes
            
            **Taxa de Aprendizado (μ)**
            - μ **pequeno**: Convergência lenta, mas mais estável
            - μ **grande**: Convergência rápida, mas pode ser instável
            - Tipicamente: `0.001 < μ < 0.1`
            
            **Número de Taps (M)**
            - M **pequeno**: Menos parâmetros, convergência rápida, menor capacidade
            - M **grande**: Mais parâmetros, maior capacidade, convergência lenta
            - Tipicamente: `8 ≤ M ≤ 128`
            
            ### Como Usar o LMS
            
            ```python
            # Criar filtro LMS
            lms = LMSFilter(num_taps=32, mu=0.01)
            
            # Executar adaptação
            y, e, weights, mse = lms.run(
                x=noise_reference,    # Sinal de referência (ruído)
                d=noisy_signal        # Sinal desejado (sinal + ruído)
            )
            
            # e[n] contém o sinal limpo estimado
            # mse contém o histórico do erro quadrático
            ```
            
            ### Vantagens do LMS
            
            - ✅ Adapta-se automaticamente às características do ruído
            - ✅ Não requer conhecimento prévio do sinal ou ruído
            - ✅ Computacionalmente eficiente
            - ✅ Funciona bem para ruídos correlacionados
            
            ### Limitações
            
            - ❌ Requer sinal de **referência correlacionado** com o ruído
            - ❌ Convergência depende da escolha de μ
            - ❌ Performance degrada com ruído não-correlacionado
            """)
        
        # Análise Espectral
        with st.expander("🌊 **Análise Espectral (FFT/STFT/CWT)**"):
            st.markdown("""
            ### 1. FFT (Fast Fourier Transform)
            
            A FFT decompõe um sinal no domínio do tempo em suas componentes de frequência.
            
            **Uso:** Análise global do conteúdo frequencial do sinal.
            
            **Limitação:** Não fornece informação temporal (quando cada frequência ocorre).
            
            ### 2. STFT (Short-Time Fourier Transform)
            
            Aplica FFT em **janelas deslizantes** do sinal, gerando um **espectrograma**.
            
            **Parâmetros:**
            - **Tamanho da janela**: Trade-off entre resolução temporal e frequencial
              - Janela grande: boa resolução frequencial, má resolução temporal
              - Janela pequena: boa resolução temporal, má resolução frequencial
            - **Sobreposição**: Percentual de overlap entre janelas consecutivas (tipicamente 50-75%)
            
            **Uso:** Visualizar como o espectro muda ao longo do tempo.
            
            ### 3. CWT (Continuous Wavelet Transform)
            
            Decompõe o sinal usando **wavelets** em diferentes escalas.
            
            **Vantagem sobre STFT:**
            - Resolução variável: alta resolução temporal para altas frequências, 
              alta resolução frequencial para baixas frequências
            
            **Uso:** Análise de sinais não-estacionários, detecção de transientes.
            """)
        
        # Métricas de Qualidade
        with st.expander("📊 **Métricas de Qualidade de Sinal**"):
            st.markdown("""
            ### SNR (Signal-to-Noise Ratio)
            
            Mede a relação entre a potência do sinal e a potência do ruído:
            
            ```
            SNR (dB) = 10 × log₁₀(P_signal / P_noise)
            ```
            
            - **SNR alto** (> 20 dB): Sinal limpo, ruído baixo
            - **SNR médio** (10-20 dB): Ruído perceptível
            - **SNR baixo** (< 10 dB): Ruído dominante
            
            ### PSNR (Peak Signal-to-Noise Ratio)
            
            Similar ao SNR, mas usa o pico do sinal em vez da potência média:
            
            ```
            PSNR (dB) = 10 × log₁₀(peak² / MSE)
            ```
            
            ### MSE (Mean Squared Error)
            
            Erro quadrático médio entre sinal original e processado:
            
            ```
            MSE = (1/N) × Σ(original[n] - processed[n])²
            ```
            
            Quanto menor o MSE, melhor a reconstrução.
            
            ### Fator de Crista (Crest Factor)
            
            Razão entre o valor de pico e o valor RMS:
            
            ```
            CF = Peak / RMS
            ```
            
            Indica a "impulsividade" do sinal.
            """)
        
        st.divider()
        st.success("""
        💡 **Dica**: Use esta aba como referência enquanto explora as funcionalidades da ferramenta! 
        Os conceitos aqui apresentados são aplicados nas outras abas.
        """)

    # =================================================================
    # TAB 1: EDA & TIME DOMAIN
    # =================================================================
    with tab_eda:
        st.subheader("📈 Análise Exploratória de Dados e Domínio do Tempo")
        
        has_mix = 'mixed_signal' in st.session_state
        has_filt = 'filtered_signal' in st.session_state
        
        n_cols = 1
        if has_mix:
            n_cols = 2
        if has_filt:
            n_cols = 3
        
        cols = st.columns(n_cols)
        
        # Column 1: Original Signal
        with cols[0]:
            st.markdown("#### 🎵 Sinal Original")
            fig_orig = create_time_domain_plot(
                st.session_state.signal,
                st.session_state.sampling_rate,
                "Sinal Original",
                "#00CC96",
                show_envelope=True
            )
            st.plotly_chart(fig_orig, use_container_width=True)
            
            img_bytes = fig_to_bytes(fig_orig)
            st.download_button(
                "💾 Baixar Imagem",
                data=img_bytes,
                file_name="sinal_original.png",
                mime="image/png",
                use_container_width=True
            )
            
            safe_audio(st.session_state.signal, st.session_state.sampling_rate, "Original")
        
        # Column 2: Noisy Signal
        if has_mix:
            with cols[1]:
                st.markdown(f"#### 🔊 Sinal com Ruído (SNR = {target_snr} dB)")
                fig_noisy = create_time_domain_plot(
                    st.session_state.mixed_signal,
                    st.session_state.sampling_rate,
                    f"Sinal Ruidoso",
                    "#EF553B",
                    show_envelope=True
                )
                st.plotly_chart(fig_noisy, use_container_width=True)
                
                img_bytes = fig_to_bytes(fig_noisy)
                st.download_button(
                    "💾 Baixar Imagem",
                    data=img_bytes,
                    file_name="sinal_ruidoso.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                safe_audio(st.session_state.mixed_signal, st.session_state.sampling_rate, "Ruidoso")
        
        # Column 3: Filtered Signal
        if has_filt:
            with cols[2]:
                st.markdown("#### ✨ Sinal Filtrado")
                fig_filt = create_time_domain_plot(
                    st.session_state.filtered_signal,
                    st.session_state.sampling_rate,
                    "Sinal Filtrado",
                    "#AB63FA",
                    show_envelope=True
                )
                st.plotly_chart(fig_filt, use_container_width=True)
                
                img_bytes = fig_to_bytes(fig_filt)
                st.download_button(
                    "💾 Baixar Imagem",
                    data=img_bytes,
                    file_name="sinal_filtrado.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                safe_audio(st.session_state.filtered_signal, st.session_state.sampling_rate, "Filtrado")
        
        # Metrics Section
        st.divider()
        st.markdown("### 📋 Métricas Quantitativas")
        
        # Original Signal Metrics
        st.markdown("#### Sinal Original")
        m_orig = SignalProcessor.compute_signal_metrics(
            st.session_state.signal,
            st.session_state.signal,
            st.session_state.sampling_rate
        )
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMS", f"{m_orig['rms_clean']:.4f}")
        col2.metric("Pico", f"{m_orig['peak_clean']:.4f}")
        col3.metric("Fator de Crista", f"{m_orig['crest_clean']:.2f}")
        col4.metric("Energia", f"{np.sum(st.session_state.signal**2):.2e}")
        
        # Noisy Signal Metrics
        if has_mix:
            st.markdown("#### Comparação: Original vs Ruidoso")
            m_mix = SignalProcessor.compute_signal_metrics(
                st.session_state.signal,
                st.session_state.mixed_signal,
                st.session_state.sampling_rate
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "RMS (Ruidoso)",
                f"{m_mix['rms_proc']:.4f}",
                delta=f"{((m_mix['rms_proc']-m_mix['rms_clean'])/m_mix['rms_clean']*100):.1f}%"
            )
            col2.metric(
                "SNR Real",
                f"{m_mix['snr']:.2f} dB",
                help="SNR calculado entre sinal original e ruidoso"
            )
            col3.metric(
                "PSNR",
                f"{m_mix['psnr']:.2f} dB",
                help="Peak Signal-to-Noise Ratio"
            )
            col4.metric(
                "MSE",
                f"{m_mix['mse']:.2e}",
                help="Mean Squared Error"
            )
        
        # Filtered Signal Metrics
        if has_filt:
            st.markdown("#### Comparação: Original vs Filtrado")
            m_filt = SignalProcessor.compute_signal_metrics(
                st.session_state.signal,
                st.session_state.filtered_signal,
                st.session_state.sampling_rate
            )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "RMS (Filtrado)",
                f"{m_filt['rms_proc']:.4f}"
            )
            col2.metric(
                "SNR Melhorado",
                f"{m_filt['snr']:.2f} dB",
                delta=f"+{m_filt['snr'] - m_mix['snr']:.2f} dB" if has_mix else None,
                help="Melhoria em SNR após filtragem"
            )
            col3.metric(
                "PSNR",
                f"{m_filt['psnr']:.2f} dB"
            )
            col4.metric(
                "MSE",
                f"{m_filt['mse']:.2e}"
            )

    # =================================================================
    # TAB 2: SPECTRAL ANALYSIS
    # =================================================================
    with tab_spec:
        st.subheader("🌊 Análise Espectral Multidimensional")
        
        # Signal selector
        signal_options = ["Original"]
        if has_mix:
            signal_options.append("Ruidoso")
        if has_filt:
            signal_options.append("Filtrado")
        
        target_name = st.radio(
            "Selecione o sinal para análise espectral:",
            signal_options,
            horizontal=True
        )
        
        sig_map = {
            "Original": st.session_state.signal,
            "Ruidoso": st.session_state.get('mixed_signal'),
            "Filtrado": st.session_state.get('filtered_signal')
        }
        
        curr_sig = sig_map.get(target_name, st.session_state.signal)
        if curr_sig is None:
            curr_sig = st.session_state.signal
        
        curr_sig = np.nan_to_num(curr_sig)
        fs = st.session_state.sampling_rate
        
        subtab1, subtab2, subtab3 = st.tabs([
            "FFT (Magnitude & Fase)",
            "STFT (Espectrograma)",
            "CWT (Wavelet)"
        ])
        
        # --- FFT Analysis ---
        with subtab1:
            st.markdown("### Transformada Rápida de Fourier (FFT)")
            
            window_type = st.selectbox(
                "Janela para FFT",
                ["hann", "hamming", "blackman", "rect"],
                help="Tipo de janela aplicada antes da FFT"
            )
            
            # CACHE: FFT cacheada
            xf, mag, mag_db, phase = cached_compute_fft(curr_sig, fs, window_type)
            
            # MINMAX: downsampling pareado (frequência x magnitude)
            xf_ds, m_ds = downsample_xy(xf, mag, max_points=MAX_PLOT_POINTS, method='minmax')
            xf_ds2, m_db_ds = downsample_xy(xf, mag_db, max_points=MAX_PLOT_POINTS, method='minmax')
            xf_ds3, p_ds = downsample_xy(xf, phase, max_points=MAX_PLOT_POINTS, method='minmax')
            
            # Magnitude Plot (Linear)
            fig_mag = go.Figure()
            fig_mag.add_trace(go.Scattergl(
                x=xf_ds, y=m_ds,
                name="Magnitude",
                line=dict(color='#00CC96', width=1.5)
            ))
            fig_mag.update_layout(
                title="Espectro de Magnitude (Linear)",
                xaxis_title="Frequência (Hz)",
                yaxis_title="Magnitude",
                template="plotly_dark",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig_mag, use_container_width=True)
            st.download_button(
                "💾 Baixar Magnitude Linear",
                data=fig_to_bytes(fig_mag),
                file_name="fft_magnitude_linear.png",
                mime="image/png"
            )
            
            # Magnitude Plot (dB)
            fig_mag_db = go.Figure()
            fig_mag_db.add_trace(go.Scattergl(
                x=xf_ds2, y=m_db_ds,
                name="Magnitude (dB)",
                line=dict(color='#FFA15A', width=1.5)
            ))
            fig_mag_db.update_layout(
                title="Densidade Espectral de Potência (dB)",
                xaxis_title="Frequência (Hz)",
                yaxis_title="Magnitude (dB)",
                template="plotly_dark",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig_mag_db, use_container_width=True)
            st.download_button(
                "💾 Baixar Magnitude dB",
                data=fig_to_bytes(fig_mag_db),
                file_name="fft_magnitude_db.png",
                mime="image/png"
            )
            
            # Phase Plot
            fig_phs = go.Figure()
            fig_phs.add_trace(go.Scattergl(
                x=xf_ds3, y=p_ds,
                name="Fase",
                line=dict(color='#AB63FA', width=1.5)
            ))
            fig_phs.update_layout(
                title="Espectro de Fase",
                xaxis_title="Frequência (Hz)",
                yaxis_title="Fase (rad)",
                template="plotly_dark",
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig_phs, use_container_width=True)
            st.download_button(
                "💾 Baixar Fase",
                data=fig_to_bytes(fig_phs),
                file_name="fft_fase.png",
                mime="image/png"
            )
        
        # --- STFT Analysis ---
        with subtab2:
            st.markdown("### Transformada de Fourier de Tempo Curto (STFT)")
            
            col1, col2 = st.columns(2)
            with col1:
                n_win = st.select_slider(
                    "Tamanho da Janela (samples)",
                    options=[64, 128, 256, 512, 1024, 2048],
                    value=256,
                    help="Tamanho da janela para STFT - compromisso entre resolução temporal e frequencial"
                )
            with col2:
                overlap_pct = st.slider(
                    "Sobreposição (%)",
                    min_value=0,
                    max_value=90,
                    value=50,
                    step=10,
                    help="Percentual de sobreposição entre janelas"
                )
            
            noverlap = int(n_win * overlap_pct / 100)
            
            # CACHE: STFT cacheada
            f, t, Zxx = cached_compute_stft(
                curr_sig, fs,
                nperseg=n_win,
                noverlap=noverlap
            )
            mag_spec = 20 * np.log10(np.abs(Zxx) + 1e-12)
            
            # Downsample temporal do espectrograma
            if len(t) > MAX_SPEC_TIME_BINS:
                step = len(t) // MAX_SPEC_TIME_BINS
                t = t[::step]
                mag_spec = mag_spec[:, ::step]
            
            fig_stft = go.Figure(data=go.Heatmap(
                z=mag_spec,
                x=t,
                y=f,
                colorscale='Magma',
                colorbar=dict(title="Magnitude (dB)")
            ))
            fig_stft.update_layout(
                title=f"Espectrograma (Janela={n_win}, Overlap={overlap_pct}%)",
                xaxis_title="Tempo (s)",
                yaxis_title="Frequência (Hz)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_stft, use_container_width=True)
            st.download_button(
                "💾 Baixar Espectrograma",
                data=fig_to_bytes(fig_stft),
                file_name="espectrograma.png",
                mime="image/png"
            )
            
            st.info(f"""
            **Resolução Temporal**: {n_win/fs*1000:.2f} ms  
            **Resolução Frequencial**: {fs/n_win:.2f} Hz  
            **Número de Frames**: {len(t)}
            """)
            
            # Libera matriz pesada
            del Zxx, mag_spec
            gc.collect()
        
        # --- CWT Analysis ---
        with subtab3:
            st.markdown("### Transformada Wavelet Contínua (CWT)")
            
            st.info("💡 A CWT permite visualizar componentes de frequência com resolução variável no tempo")
            
            col1, col2 = st.columns(2)
            with col1:
                wavelet = st.selectbox(
                    "Família Wavelet",
                    ["cmor1.5-1.0", "morl", "mexh", "gaus1"],
                    index=0,
                    help="Escolha a wavelet mãe para a transformada"
                )
            with col2:
                max_scale = st.slider(
                    "Número de Escalas",
                    min_value=32,
                    max_value=256,
                    value=128,
                    step=32,
                    help="Número de escalas para análise"
                )
            
            # Downsample signal if too long for CWT
            if len(curr_sig) > MAX_CWT_PTS:
                step = len(curr_sig) // MAX_CWT_PTS
                sig_cwt = curr_sig[::step]
                fs_cwt = fs / step
                st.caption(f"⚠️ Sinal decimado por fator {step}x para performance ({len(sig_cwt)} amostras)")
            else:
                sig_cwt = curr_sig
                fs_cwt = fs
            
            # CACHE: CWT cacheada
            coefs, freqs = cached_compute_cwt(sig_cwt, fs_cwt, wavelet=wavelet, num_scales=max_scale)
            power = np.abs(coefs)**2
            
            t_vals = np.linspace(0, len(curr_sig)/fs, len(sig_cwt))
            
            fig_cwt = go.Figure(data=go.Heatmap(
                z=power,
                x=t_vals,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title="Potência")
            ))
            fig_cwt.update_layout(
                title=f"Escalograma CWT (Wavelet: {wavelet})",
                xaxis_title="Tempo (s)",
                yaxis_title="Frequência (Hz)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_cwt, use_container_width=True)
            st.download_button(
                "💾 Baixar Escalograma",
                data=fig_to_bytes(fig_cwt),
                file_name="escalograma_cwt.png",
                mime="image/png"
            )
            
            # Libera matrizes pesadas da CWT
            del coefs, power
            gc.collect()

    # =================================================================
    # TAB 3: STATISTICS
    # =================================================================
    with tab_stats:
        st.subheader("📈 Estatísticas Avançadas e Distribuições")
        
        signal_options = ["Original"]
        if has_mix:
            signal_options.append("Ruidoso")
        if has_filt:
            signal_options.append("Filtrado")
        
        target_stat = st.selectbox(
            "Selecione o sinal para análise estatística:",
            signal_options
        )
        
        s_stat_map = {
            "Original": st.session_state.signal,
            "Ruidoso": st.session_state.get('mixed_signal'),
            "Filtrado": st.session_state.get('filtered_signal')
        }
        
        curr_stat_sig = s_stat_map.get(target_stat, st.session_state.signal)
        
        if curr_stat_sig is not None:
            curr_stat_sig = np.nan_to_num(curr_stat_sig)
            
            # Basic Statistics
            st.markdown("### 📊 Estatísticas Descritivas")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Média", f"{np.mean(curr_stat_sig):.4f}")
            col2.metric("Mediana", f"{np.median(curr_stat_sig):.4f}")
            col3.metric("Desvio Padrão", f"{np.std(curr_stat_sig):.4f}")
            col4.metric("Variância", f"{np.var(curr_stat_sig):.4e}")
            col5.metric("Curtose", f"{np.mean((curr_stat_sig - np.mean(curr_stat_sig))**4) / (np.var(curr_stat_sig)**2):.2f}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Histograma de Amplitude")
                
                bins = st.slider("Número de bins", 20, 200, 100, key="hist_bins")
                
                # CACHE: histograma cacheado
                counts, bin_centers = cached_compute_histogram(curr_stat_sig, bins=bins)
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=bin_centers,
                    y=counts,
                    name='PDF',
                    marker=dict(color='#00CC96', line=dict(color='#00CC96', width=1))
                ))
                fig_hist.update_layout(
                    title="Função Densidade de Probabilidade (PDF)",
                    xaxis_title="Amplitude",
                    yaxis_title="Densidade",
                    template="plotly_dark",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.download_button(
                    "💾 Baixar Histograma",
                    data=fig_to_bytes(fig_hist),
                    file_name="histograma.png",
                    mime="image/png"
                )
                
                st.info(f"""
                **Assimetria (Skewness)**: {np.mean((curr_stat_sig - np.mean(curr_stat_sig))**3) / (np.std(curr_stat_sig)**3):.3f}  
                **Amplitude**: [{np.min(curr_stat_sig):.4f}, {np.max(curr_stat_sig):.4f}]
                """)
            
            with col2:
                st.markdown("#### Função de Autocorrelação")
                
                max_lag = st.slider(
                    "Lag Máximo (samples)",
                    100, 5000, 1000,
                    step=100,
                    key="acf_lag"
                )
                
                # CACHE: autocorrelação cacheada
                lags, corr = cached_compute_autocorrelation(curr_stat_sig, max_lag=max_lag)
                lags_time = lags / st.session_state.sampling_rate
                
                # MINMAX: downsampling pareado
                l_ds, c_ds = downsample_xy(lags_time, corr, max_points=2000, method='minmax')
                
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Scattergl(
                    x=l_ds, y=c_ds,
                    mode='lines',
                    name='ACF',
                    line=dict(color='cyan', width=1.5)
                ))
                fig_acf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_acf.update_layout(
                    title="Autocorrelação Normalizada",
                    xaxis_title="Lag (s)",
                    yaxis_title="Correlação",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_acf, use_container_width=True)
                st.download_button(
                    "💾 Baixar Autocorrelação",
                    data=fig_to_bytes(fig_acf),
                    file_name="autocorrelacao.png",
                    mime="image/png"
                )
                
                zero_crossings = np.where(np.diff(np.sign(corr)))[0]
                if len(zero_crossings) > 0:
                    first_zero = lags_time[zero_crossings[0]]
                    st.info(f"**Primeiro Zero**: {first_zero*1000:.2f} ms")

    # =================================================================
    # TAB 4: DIGITAL FILTERS (FIR/IIR)
    # =================================================================
    with tab_filter:
        st.subheader("🛠️ Projeto e Aplicação de Filtros Digitais")
        
        col_params, col_viz = st.columns([1, 2])
        
        # --- FILTER DESIGN PARAMETERS ---
        with col_params:
            st.markdown("### Parâmetros do Filtro")
            
            arch = st.radio(
                "Arquitetura",
                ["FIR", "IIR"],
                horizontal=True,
                help="FIR: Fase linear, sempre estável | IIR: Mais eficiente, pode ser instável"
            )
            
            fs = st.session_state.sampling_rate
            nyq = fs / 2
            
            filter_type = st.selectbox(
                "Tipo de Filtro",
                ["Passa-Baixas", "Passa-Altas"],
                help="Tipo de resposta em frequência"
            )
            pass_zero = (filter_type == "Passa-Baixas")
            btype = 'lowpass' if pass_zero else 'highpass'
            
            cutoff = st.number_input(
                "Frequência de Corte (Hz)",
                min_value=1.0,
                max_value=float(nyq - 1),
                value=min(1000.0, nyq - 100),
                step=10.0,
                help=f"Frequência de corte (Nyquist: {nyq:.0f} Hz)"
            )
            
            st.divider()
            
            # FIR specific parameters
            if arch == "FIR":
                st.markdown("#### Parâmetros FIR")
                
                fir_method = st.selectbox(
                    "Método de Projeto",
                    ["Janelamento", "Parks-McClellan"],
                    help="Janelamento: simples, rápido | Parks-McClellan: otimizado"
                )
                
                taps = st.number_input(
                    "Número de Taps",
                    min_value=11,
                    max_value=501,
                    value=51,
                    step=2,
                    help="Número de coeficientes (ordem + 1)"
                )
                
                if fir_method == "Janelamento":
                    window_fir = st.selectbox(
                        "Tipo de Janela",
                        ["rect", "hann", "hamming", "blackman"],
                        help="Janela aplicada aos coeficientes"
                    )
                    method_param = 'window'
                else:
                    window_fir = 'hamming'
                    method_param = 'parks-mcclellan'
                
                if st.button("⚙️ Projetar Filtro FIR", use_container_width=True):
                    with st.spinner("Projetando filtro FIR..."):
                        try:
                            # CACHE: design cacheado
                            b = cached_design_fir(
                                int(taps),
                                cutoff,
                                fs,
                                window=window_fir,
                                pass_zero=pass_zero,
                                method=method_param
                            )
                            st.session_state.coeffs = (b, np.array([1.0]))
                            st.session_state.coeffs_key = _make_coeffs_key(
                                "FIR", taps=taps, cutoff=cutoff, fs=fs,
                                window=window_fir, method=method_param
                            )
                            st.session_state.filter_type = f"FIR-{fir_method}"
                            st.session_state.filter_order = len(b) - 1
                            st.success(f"✅ Filtro FIR projetado ({taps} taps, ordem {len(b)-1})")
                        except Exception as e:
                            st.error(f"❌ Erro: {e}")
            
            # IIR specific parameters
            else:
                st.markdown("#### Parâmetros IIR")
                
                ftype_display = st.selectbox(
                    "Modelo de Filtro",
                    ["Butterworth", "Chebyshev I", "Chebyshev II", "Elíptico"],
                    help="Tipo de aproximação da resposta ideal"
                )
                
                ftype_map = {
                    "Butterworth": "butter",
                    "Chebyshev I": "cheby1",
                    "Chebyshev II": "cheby2",
                    "Elíptico": "ellip"
                }
                ftype = ftype_map[ftype_display]
                
                order = st.number_input(
                    "Ordem do Filtro",
                    min_value=1,
                    max_value=20,
                    value=4,
                    help="Ordem do filtro (inclinação da resposta)"
                )
                
                rp = 1.0
                rs = 40.0
                
                if ftype in ['cheby1', 'ellip']:
                    rp = st.number_input(
                        "Ripple na Banda Passante (dB)",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        help="Ondulação permitida na banda passante"
                    )
                
                if ftype in ['cheby2', 'ellip']:
                    rs = st.number_input(
                        "Atenuação na Banda Rejeitada (dB)",
                        min_value=10.0,
                        max_value=100.0,
                        value=40.0,
                        step=5.0,
                        help="Atenuação mínima na banda rejeitada"
                    )
                
                if st.button("⚙️ Projetar Filtro IIR", use_container_width=True):
                    with st.spinner("Projetando filtro IIR..."):
                        try:
                            # CACHE: design cacheado
                            sos = cached_design_iir(
                                int(order),
                                cutoff,
                                fs,
                                btype=btype,
                                ftype=ftype,
                                rp=rp,
                                rs=rs
                            )
                            st.session_state.coeffs = sos
                            st.session_state.coeffs_key = _make_coeffs_key(
                                "IIR", order=order, cutoff=cutoff, fs=fs,
                                btype=btype, ftype=ftype, rp=rp, rs=rs
                            )
                            st.session_state.filter_type = f"IIR-{ftype_display}"
                            st.session_state.filter_order = order
                            st.success(f"✅ Filtro IIR projetado (Ordem {order})")
                        except Exception as e:
                            st.error(f"❌ Erro: {e}")
            
            st.divider()
            
            # Apply filter button
            zero_phase = st.checkbox(
                "Filtragem com Fase Zero (filtfilt)",
                value=False,
                help="Aplica filtro duas vezes (forward e backward) para eliminar distorção de fase"
            )
            
            if st.button("🎯 APLICAR FILTRO AO SINAL", use_container_width=True, type="primary"):
                if 'coeffs' not in st.session_state:
                    st.error("❌ Projete um filtro primeiro!")
                else:
                    with st.spinner("Aplicando filtro..."):
                        try:
                            target = st.session_state.get('mixed_signal', st.session_state.signal)
                            
                            # CACHE: apply_filter cacheado
                            coeffs_key = st.session_state.get('coeffs_key', 'unknown')
                            filt = cached_apply_filter(
                                target,
                                coeffs_key,
                                st.session_state.coeffs,
                                zero_phase=zero_phase
                            )
                            
                            st.session_state.filtered_signal = filt
                            st.success("✅ Filtro aplicado! Veja os resultados na aba 'EDA'")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao aplicar filtro: {e}")
        
        # --- FILTER VISUALIZATION ---
        with col_viz:
            st.markdown("### Análise do Filtro")
            
            if 'coeffs' in st.session_state:
                try:
                    # CACHE: análise do filtro cacheada
                    coeffs_key = st.session_state.get('coeffs_key', 'unknown')
                    res = cached_analyze_filter(coeffs_key, st.session_state.coeffs, fs)
                    
                    st.info(f"**Ordem do Filtro**: {st.session_state.get('filter_order', 'N/A')}")
                    
                    # Magnitude Response
                    st.markdown("#### Resposta em Magnitude")
                    
                    fig_mag = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Magnitude (Linear)", "Magnitude (dB)"),
                        vertical_spacing=0.12
                    )
                    
                    fig_mag.add_trace(
                        go.Scattergl(
                            x=res['w'],
                            y=res['magnitude'],
                            name="Magnitude",
                            line=dict(color='#00CC96', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    fig_mag.add_trace(
                        go.Scattergl(
                            x=res['w'],
                            y=res['magnitude_db'],
                            name="Magnitude (dB)",
                            line=dict(color='#FFA15A', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig_mag.add_vline(x=cutoff, line_dash="dash", line_color="red", opacity=0.5, row="all")
                    
                    fig_mag.update_xaxes(title_text="Frequência (Hz)", row=2, col=1)
                    fig_mag.update_yaxes(title_text="Magnitude", row=1, col=1)
                    fig_mag.update_yaxes(title_text="Magnitude (dB)", row=2, col=1)
                    
                    fig_mag.update_layout(
                        template="plotly_dark",
                        height=500,
                        showlegend=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_mag, use_container_width=True)
                    st.download_button(
                        "💾 Baixar Resposta em Magnitude",
                        data=fig_to_bytes(fig_mag),
                        file_name="filtro_magnitude.png",
                        mime="image/png"
                    )
                    
                    # Passband detail (zoom)
                    st.markdown("#### Detalhe da Banda Passante (Ripple)")
                    
                    if pass_zero:
                        pb_mask = res['w'] <= cutoff * 1.2
                    else:
                        pb_mask = res['w'] >= cutoff * 0.8
                    
                    fig_ripple = go.Figure()
                    fig_ripple.add_trace(go.Scattergl(
                        x=res['w'][pb_mask],
                        y=res['magnitude_db'][pb_mask],
                        name="Magnitude (dB)",
                        line=dict(color='#FFA15A', width=2)
                    ))
                    fig_ripple.add_vline(x=cutoff, line_dash="dash", line_color="red", opacity=0.5)
                    fig_ripple.update_layout(
                        title="Ondulação na Banda Passante",
                        xaxis_title="Frequência (Hz)",
                        yaxis_title="Magnitude (dB)",
                        template="plotly_dark",
                        height=300,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_ripple, use_container_width=True)
                    st.download_button(
                        "💾 Baixar Ripple",
                        data=fig_to_bytes(fig_ripple),
                        file_name="filtro_ripple.png",
                        mime="image/png"
                    )
                    
                    # Phase Response and Group Delay
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Resposta de Fase")
                        
                        fig_phase = go.Figure()
                        fig_phase.add_trace(go.Scattergl(
                            x=res['w'],
                            y=res['phase_deg'],
                            name="Fase",
                            line=dict(color='#AB63FA', width=2)
                        ))
                        fig_phase.add_vline(x=cutoff, line_dash="dash", line_color="red", opacity=0.5)
                        fig_phase.update_layout(
                            title="Resposta de Fase",
                            xaxis_title="Frequência (Hz)",
                            yaxis_title="Fase (graus)",
                            template="plotly_dark",
                            height=300,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_phase, use_container_width=True)
                        st.download_button(
                            "💾 Baixar Fase",
                            data=fig_to_bytes(fig_phase),
                            file_name="filtro_fase.png",
                            mime="image/png",
                            key="download_phase"
                        )
                    
                    with col2:
                        st.markdown("#### Atraso de Grupo")
                        
                        fig_gd = go.Figure()
                        fig_gd.add_trace(go.Scattergl(
                            x=res['w_gd'],
                            y=res['gd'],
                            name="Group Delay",
                            line=dict(color='#EF553B', width=2)
                        ))
                        fig_gd.add_vline(x=cutoff, line_dash="dash", line_color="red", opacity=0.5)
                        fig_gd.update_layout(
                            title="Atraso de Grupo",
                            xaxis_title="Frequência (Hz)",
                            yaxis_title="Atraso (samples)",
                            template="plotly_dark",
                            height=300,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_gd, use_container_width=True)
                        st.download_button(
                            "💾 Baixar Group Delay",
                            data=fig_to_bytes(fig_gd),
                            file_name="filtro_group_delay.png",
                            mime="image/png",
                            key="download_gd"
                        )
                    
                    # Pole-Zero Plot and Impulse Response
                    st.markdown("### Análise no Domínio Z e Temporal")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Diagrama Polo-Zero")
                        
                        if len(res['p']) > 0 or len(res['z']) > 0:
                            fig_pz = go.Figure()
                            
                            theta = np.linspace(0, 2*np.pi, 100)
                            fig_pz.add_trace(go.Scattergl(
                                x=np.cos(theta),
                                y=np.sin(theta),
                                mode='lines',
                                name='Círculo Unitário',
                                line=dict(color='gray', dash='dash')
                            ))
                            
                            if len(res['z']) > 0:
                                fig_pz.add_trace(go.Scattergl(
                                    x=np.real(res['z']),
                                    y=np.imag(res['z']),
                                    mode='markers',
                                    name='Zeros',
                                    marker=dict(symbol='circle-open', size=12, color='blue', line=dict(width=2))
                                ))
                            
                            if len(res['p']) > 0:
                                fig_pz.add_trace(go.Scattergl(
                                    x=np.real(res['p']),
                                    y=np.imag(res['p']),
                                    mode='markers',
                                    name='Polos',
                                    marker=dict(symbol='x', size=12, color='red', line=dict(width=2))
                                ))
                            
                            fig_pz.update_layout(
                                title="Diagrama Polo-Zero",
                                xaxis_title="Parte Real",
                                yaxis_title="Parte Imaginária",
                                template="plotly_dark",
                                height=400,
                                yaxis=dict(scaleanchor="x", scaleratio=1)
                            )
                            st.plotly_chart(fig_pz, use_container_width=True)
                            st.download_button(
                                "💾 Baixar Polo-Zero",
                                data=fig_to_bytes(fig_pz),
                                file_name="filtro_polo_zero.png",
                                mime="image/png",
                                key="download_pz"
                            )
                    
                    with col2:
                        st.markdown("#### Resposta ao Impulso")
                        
                        h = SignalProcessor.compute_impulse_response(st.session_state.coeffs, num_samples=100)
                        n = np.arange(len(h))
                        
                        fig_ir = go.Figure()
                        fig_ir.add_trace(go.Scatter(
                            x=n,
                            y=h,
                            mode='markers+lines',
                            name='h[n]',
                            marker=dict(size=6, color='#00CC96'),
                            line=dict(width=1, color='#00CC96')
                        ))
                        fig_ir.update_layout(
                            title="Resposta ao Impulso (100 amostras)",
                            xaxis_title="n (amostras)",
                            yaxis_title="Amplitude",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_ir, use_container_width=True)
                        st.download_button(
                            "💾 Baixar Impulso",
                            data=fig_to_bytes(fig_ir),
                            file_name="filtro_impulso.png",
                            mime="image/png",
                            key="download_ir"
                        )
                    
                    # Stability Analysis
                    st.divider()
                    st.markdown("#### Análise de Estabilidade")
                    
                    max_pole = np.max(np.abs(res['p'])) if len(res['p']) > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Polo Máximo", f"{max_pole:.6f}")
                    col2.metric("Número de Zeros", len(res['z']))
                    col3.metric("Número de Polos", len(res['p']))
                    
                    if max_pole > 1.000001:
                        st.error(f"⚠️ **FILTRO INSTÁVEL**: Polo com magnitude {max_pole:.6f} > 1")
                        st.warning("Filtro pode oscilar ou divergir. Considere reduzir a ordem ou usar FIR.")
                    else:
                        st.success(f"✅ **FILTRO ESTÁVEL**: Todos os polos dentro do círculo unitário")
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise do filtro: {e}")
            else:
                st.info("👈 Projete um filtro para visualizar suas características")
            
            # Residual Analysis
            if 'filtered_signal' in st.session_state and 'mixed_signal' in st.session_state:
                st.divider()
                st.markdown("### 🔍 Análise de Resíduos")
                st.caption("Resíduo = Sinal Ruidoso - Sinal Filtrado (componentes removidos)")
                
                orig = st.session_state.mixed_signal
                filt = st.session_state.filtered_signal
                
                L = min(len(orig), len(filt))
                residual = orig[:L] - filt[:L]
                
                fig_res = create_time_domain_plot(
                    residual,
                    st.session_state.sampling_rate,
                    "Resíduo (Componentes Removidas)",
                    "#EF553B"
                )
                st.plotly_chart(fig_res, use_container_width=True)
                st.download_button(
                    "💾 Baixar Resíduo",
                    data=fig_to_bytes(fig_res),
                    file_name="residuo.png",
                    mime="image/png",
                    key="download_residual"
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMS do Resíduo", f"{np.sqrt(np.mean(residual**2)):.4f}")
                col2.metric("Energia Removida", f"{np.sum(residual**2):.2e}")
                col3.metric("Pico do Resíduo", f"{np.max(np.abs(residual)):.4f}")
                col4.metric("% Energia Original", f"{(np.sum(residual**2)/np.sum(orig**2)*100):.1f}%")

    # =================================================================
    # TAB 5: ADAPTIVE FILTERS (LMS)
    # =================================================================
    with tab_lms:
        st.subheader("🧠 Filtros Adaptativos - Least Mean Squares (LMS)")
        
        st.markdown("""
        O algoritmo LMS é um filtro adaptativo que aprende a estimar e remover ruído de um sinal.
        Ele requer um **sinal de referência** (ruído) e um **sinal desejado** (sinal ruidoso).
        """)
        
        if 'mixed_signal' not in st.session_state or 'noise_ref' not in st.session_state:
            st.warning("""
            ⚠️ **Pré-requisito não atendido**: 
            Para usar o filtro LMS, você precisa primeiro:
            1. Carregar ou gerar um sinal original
            2. Aplicar ruído na seção "Adição de Ruído" da barra lateral
            
            Isso criará o sinal de referência (ruído) necessário para o LMS.
            """)
        else:
            col_params, col_viz = st.columns([1, 2])
            
            # --- LMS PARAMETERS ---
            with col_params:
                st.markdown("### Parâmetros do LMS")
                
                mu = st.number_input(
                    "Taxa de Aprendizado (μ)",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                    help="Controla velocidade de convergência vs. estabilidade"
                )
                
                taps = st.number_input(
                    "Número de Taps",
                    min_value=4,
                    max_value=128,
                    value=32,
                    help="Ordem do filtro adaptativo"
                )
                
                st.info(f"""
                **Taxa de Amostragem**: {st.session_state.sampling_rate:,.0f} Hz  
                **Amostras**: {len(st.session_state.mixed_signal):,}  
                **Duração**: {len(st.session_state.mixed_signal)/st.session_state.sampling_rate:.2f}s
                """)
                
                if st.button("🚀 Executar LMS", use_container_width=True, type="primary"):
                    with st.spinner("Executando algoritmo LMS..."):
                        try:
                            d = st.session_state.mixed_signal
                            x = st.session_state.noise_ref
                            
                            min_len = min(len(d), len(x))
                            d = d[:min_len]
                            x = x[:min_len]
                            
                            norm_fact = np.max(np.abs(d)) + 1e-9
                            
                            # CACHE: LMS cacheado
                            y, e, w, mse_history = cached_lms_filter(
                                x/norm_fact, d/norm_fact,
                                int(taps), mu
                            )
                            
                            st.session_state.lms_output = e * norm_fact
                            st.session_state.lms_noise_est = y * norm_fact
                            st.session_state.lms_mse = mse_history * (norm_fact**2)
                            st.session_state.lms_weights = w
                            
                            st.success("✅ LMS convergiu com sucesso!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao executar LMS: {e}")
            
            # --- LMS VISUALIZATION ---
            with col_viz:
                st.markdown("### Resultados do LMS")
                
                if 'lms_output' in st.session_state:
                    # Output signal (cleaned)
                    st.markdown("#### Sinal Limpo (Erro do LMS)")
                    fig_lms = create_time_domain_plot(
                        st.session_state.lms_output,
                        st.session_state.sampling_rate,
                        "Sinal Processado pelo LMS",
                        "#00CC96"
                    )
                    st.plotly_chart(fig_lms, use_container_width=True)
                    st.download_button(
                        "💾 Baixar Sinal LMS",
                        data=fig_to_bytes(fig_lms),
                        file_name="lms_sinal_limpo.png",
                        mime="image/png",
                        key="download_lms_signal"
                    )
                    safe_audio(st.session_state.lms_output, st.session_state.sampling_rate, "LMS")
                    
                    # MSE Convergence
                    st.markdown("#### Curva de Convergência (MSE)")
                    
                    mse_db = 10 * np.log10(st.session_state.lms_mse + 1e-12)
                    
                    # MINMAX: downsampling pareado
                    n_full = np.linspace(0, len(st.session_state.lms_mse), len(mse_db))
                    n_ds, mse_ds = downsample_xy(n_full, mse_db, max_points=MAX_PLOT_POINTS, method='minmax')
                    
                    fig_mse = go.Figure()
                    fig_mse.add_trace(go.Scattergl(
                        x=n_ds,
                        y=mse_ds,
                        mode='lines',
                        name='MSE (dB)',
                        line=dict(color='#FFA15A', width=1.5)
                    ))
                    fig_mse.update_layout(
                        title="Erro Quadrático Médio ao Longo do Tempo",
                        xaxis_title="Iteração (amostra)",
                        yaxis_title="MSE (dB)",
                        template="plotly_dark",
                        height=300,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_mse, use_container_width=True)
                    st.download_button(
                        "💾 Baixar Convergência MSE",
                        data=fig_to_bytes(fig_mse),
                        file_name="lms_convergencia.png",
                        mime="image/png",
                        key="download_lms_mse"
                    )
                    
                    # Performance Metrics
                    st.markdown("#### Métricas de Performance")
                    
                    # CACHE: SNR cacheado
                    snr_lms, psnr_lms = cached_compute_snr(
                        st.session_state.signal[:len(st.session_state.lms_output)],
                        st.session_state.lms_output
                    )
                    
                    snr_init, psnr_init = cached_compute_snr(
                        st.session_state.signal[:len(st.session_state.mixed_signal)],
                        st.session_state.mixed_signal
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "SNR Inicial",
                        f"{snr_init:.2f} dB"
                    )
                    col2.metric(
                        "SNR Final (LMS)",
                        f"{snr_lms:.2f} dB",
                        delta=f"+{snr_lms - snr_init:.2f} dB"
                    )
                    col3.metric(
                        "PSNR Final",
                        f"{psnr_lms:.2f} dB"
                    )
                    col4.metric(
                        "MSE Final",
                        f"{st.session_state.lms_mse[-1]:.2e}"
                    )
                    
                    # Weight vector visualization
                    st.markdown("#### Coeficientes do Filtro Adaptativo")
                    
                    fig_weights = go.Figure()
                    fig_weights.add_trace(go.Bar(
                        x=np.arange(len(st.session_state.lms_weights)),
                        y=st.session_state.lms_weights,
                        name='Pesos',
                        marker=dict(color='#AB63FA')
                    ))
                    fig_weights.update_layout(
                        title="Vetor de Pesos Final",
                        xaxis_title="Índice do Tap",
                        yaxis_title="Valor do Peso",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                    st.download_button(
                        "💾 Baixar Pesos LMS",
                        data=fig_to_bytes(fig_weights),
                        file_name="lms_pesos.png",
                        mime="image/png",
                        key="download_lms_weights"
                    )
                    
                else:
                    st.info("👈 Configure os parâmetros e execute o LMS para ver os resultados")